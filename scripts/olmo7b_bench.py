"""Cost of the recommended attribution pipeline at 7B (Olmo-3.1-7B-RL-Zero-Math).

Measures wall time, tokens/s and peak GPU memory of each Bergson stage of the two-stage
setup in docs/06_attribution_methods.md, on the real model with GRPO-style rows:

  trackstar_p32 / trackstar_p64   autocorrelation Hessian + query index + scoring of all items
  ekfac_factors                   EK-FAC factors on a subset of items (a block of layers)
  ekfac_rerank                    EK-FAC influence of the top candidates for one GRPO query

Rows are Dolci-RL-Zero-Math prompts rendered with the RL-Zero template, followed by
FineWeb-Edu text of a fixed length standing in for a rollout, with random advantages. Cost
depends on lengths and shapes, not on the text, so this needs no generation. EK-FAC is run
on a block of layers and scaled per layer, since full 7B factors (~70 GB in fp32) do not fit
next to the model on a 40 GB GPU.

    python scripts/olmo7b_bench.py prepare && python scripts/olmo7b_bench.py run
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reachtrace.attrib import bergson_yaml as by

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("BENCH_ART", ROOT / "artifacts" / "olmo7b_bench")).resolve()
OUT = ROOT / "results" / "olmo7b_bench"
MODEL = "allenai/Olmo-3.1-7B-RL-Zero-Math"
N_ITEMS = int(os.environ.get("BENCH_ITEMS", 256))
COMPLETION_TOKENS = int(os.environ.get("BENCH_LEN", 2048))
N_QUERY = 8
EKFAC_LAYERS = int(os.environ.get("BENCH_EKFAC_LAYERS", 4))
N_LAYERS = 32


def prepare() -> None:
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    from reachtrace.data.dolci import load_dolci
    from reachtrace.prompts import get_template

    tok = AutoTokenizer.from_pretrained(MODEL)
    tpl = get_template("rlzero31_math")
    dolci = load_dolci().sample(N_ITEMS + N_QUERY, random_state=0).reset_index(drop=True)
    web = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
                       streaming=True)
    filler = []
    for r in web:
        ids = tok(r["text"], add_special_tokens=False).input_ids
        while len(ids) >= COMPLETION_TOKENS and len(filler) < N_ITEMS + N_QUERY:
            filler.append(ids[:COMPLETION_TOKENS])
            ids = ids[COMPLETION_TOKENS:]
        if len(filler) >= N_ITEMS + N_QUERY:
            break
    rng = np.random.default_rng(0)
    rows = []
    for prompt, comp in zip(dolci.prompt, filler):
        p = tok(tpl.render(prompt), add_special_tokens=False).input_ids
        rows.append({"input_ids": p + comp, "labels": [-100] * len(p) + comp,
                     "advantage": float(rng.choice([-0.5, 0.5]))})
    ART.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows[:N_ITEMS]).save_to_disk(str(ART / "items"))
    Dataset.from_list(rows[N_ITEMS: N_ITEMS + N_QUERY]).save_to_disk(str(ART / "query"))
    n_tok = sum(len(r["input_ids"]) for r in rows[:N_ITEMS])
    print(f"{N_ITEMS} items, {n_tok} tokens; {N_QUERY} query rows", flush=True)
    # The streaming FineWeb reader's threads can abort the interpreter at shutdown.
    os._exit(0)


class PeakMemory:
    """Poll nvidia-smi for the peak used memory while a stage runs."""

    def __init__(self):
        self.peak, self._stop = 0, threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True,
                                 text=True).stdout.split()
            if out:
                self.peak = max(self.peak, int(out[0]))
            time.sleep(1)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()


def _run(name: str, steps: list[dict]) -> dict:
    y = by.write_pipeline(ART / f"{name}.yaml", str(ART / name), steps)
    cmd = [sys.executable, "-m", "bergson", str(y)]
    print("+", " ".join(cmd), flush=True)
    with PeakMemory() as mem:
        t0 = time.time()
        proc = subprocess.run(cmd)
        sec = time.time() - t0
    return {"stage": name, "ok": proc.returncode == 0, "seconds": sec,
            "peak_gpu_gb": mem.peak / 1024}


def run() -> None:
    from datasets import load_from_disk

    items = by.data_cfg(str(ART / "items"), advantage=True)
    query = by.data_cfg(str(ART / "query"), advantage=True)
    n_tok = sum(map(len, load_from_disk(str(ART / "items"))["input_ids"]))
    common = {"precision": "bf16", "token_batch_size": COMPLETION_TOKENS + 512,
              "max_batch_size": None}
    res = []
    for p in (32, 64):
        rp = str(ART / f"trackstar_p{p}")
        r = _run(f"trackstar_p{p}", [by.trackstar_step(
            rp, MODEL, items, query, projection_dim=p, stats_sample_size=min(256, N_ITEMS),
            **common)])
        res.append({**r, "items": N_ITEMS, "tokens": n_tok})
        _write(res)
    # EK-FAC on the last EKFAC_LAYERS layers: exclude every other layer's modules.
    keep = list(range(N_LAYERS - EKFAC_LAYERS, N_LAYERS))
    drop = [i for i in range(N_LAYERS) if i not in keep]
    # Comma-separated fnmatch patterns over collector module names ("layers.3.mlp.up_proj").
    filt = ",".join(["lm_head", *(f"layers.{i}.*" for i in drop)])
    sub = min(64, N_ITEMS)
    shared = ART / "ekfac"
    r = _run("ekfac_factors", [by.kfac_hessian_step(
        str(shared), MODEL, items, subset=sub, filter_modules=filt, **common)])
    res.append({**r, "items": sub, "tokens": n_tok * sub / N_ITEMS, "layers": EKFAC_LAYERS})
    _write(res)
    cand = ART / "candidates.csv"
    k = min(32, N_ITEMS)
    pd.DataFrame({"query": 0, "direction": "Top", "result": "", "result_index": range(k),
                  "score": 0.0}).to_csv(cand, index=False)
    r = _run("ekfac_rerank", [by.influence_step(
        str(shared), MODEL, items, query, candidates=by.candidates_cfg(str(cand), k),
        filter_modules=filt, **common)])
    res.append({**r, "items": k, "tokens": n_tok * k / N_ITEMS, "layers": EKFAC_LAYERS})
    _write(res)


def _write(res: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(res)
    df["tokens_per_s"] = df.tokens / df.seconds
    if "layers" in df:
        # Scale the layer-block EK-FAC stages to all 32 layers (cost is per module).
        scale = np.where(df.layers.notna(), N_LAYERS / df.layers.fillna(N_LAYERS), 1.0)
        df["tokens_per_s_all_layers"] = df.tokens_per_s / scale
    df.to_csv(OUT / "bench.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps(
        {"model": MODEL, "items": N_ITEMS, "completion_tokens": COMPLETION_TOKENS,
         "ekfac_layers": EKFAC_LAYERS, "gpu": _gpu()}, indent=1))
    print(df.to_string(index=False))


def _gpu() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    {"prepare": prepare, "run": run}[sys.argv[1]]()
