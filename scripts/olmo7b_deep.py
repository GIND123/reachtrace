"""Few problems, many rollouts: is the 7B GRPO contrast reliable within a problem?

Follow-up to scripts/olmo7b_midtrain.py after the Bergson maintainers' feedback on its
limits result: (1) different data may well support different problems, so reliability is
measured within a problem (two halves of its own rollouts) and compared with agreement
between problems; (2) long documents are scored in chunks and the chunk scores averaged,
instead of truncated; (3) collect more data on a smaller set of things. The first run's
incorrect rollouts mostly hit the token cap, so the contrast is also taken between finished
solutions only, with and without a length control.

Stages (artifacts under $DEEP_ART, results in results/olmo7b_deep/):

  problems   the first run's problems with both a finished correct and a finished
             incorrect rollout (artifacts/olmo7b_midtrain)
  rollouts   N_SAMPLES RL-Zero samples per problem, one vLLM engine per GPU
  queries    every rollout is one query row (plain gradient, no advantage weighting:
             TrackStar's column is then cos(q_i, d) and any contrast is applied afterwards)
  docs       N_DOCS full documents per Dolmino source, split into MAX_DOC-token chunks
  attribute  TrackStar p32 over all chunks, one column per rollout
  evaluate   within-problem split-half reliability of each contrast against a shuffled-label
             null, agreement between problems, chunk-mean against first-chunk scores
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from olmo7b_midtrain import (  # noqa: E402
    BASE,
    DOLMINO,
    GROUP,
    MIDTRAIN_END,
    N_SHARDS,
    RL,
    _bergson,
    _scores,
    _tok,
    _write_json,
)

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("DEEP_ART", ROOT / "artifacts" / "olmo7b_deep")).resolve()
MID = Path(os.environ.get("MID_ART", ROOT / "artifacts" / "olmo7b_midtrain")).resolve()
OUT = ROOT / "results" / "olmo7b_deep"
SEED = 20261104  # not the first run's seed: new samples
N_SAMPLES = int(os.environ.get("DEEP_SAMPLES", 64))
MAX_NEW = 7680  # as in the first run, so prompt + rollout fit the stage-2 context
MAX_DOC = 1024  # chunk length
MAX_CHUNKS = 8  # chunks per document (8,192 tokens)
N_DOCS = int(os.environ.get("DEEP_DOCS", 60))
TOKEN_BATCH = 8192
N_NULL = int(os.environ.get("DEEP_NULL", 30))  # shuffled-label null draws


# ------------------------------------------------------------------------------- problems


def problems() -> None:
    q = pd.read_parquet(MID / "queries_meta.parquet")
    fin = q.finish_reason == "stop"
    both = q[fin].groupby("problem_id").correct.agg(lambda s: s.any() and not s.all())
    keep = sorted(both[both].index)
    probs = pd.read_parquet(MID / "problems.parquet")
    probs = probs[probs.problem_id.isin(keep)]
    ART.mkdir(parents=True, exist_ok=True)
    probs.to_parquet(ART / "problems.parquet")
    print(f"{len(probs)} problems: {keep}")


# ------------------------------------------------------------------------------- rollouts


def rollouts() -> None:
    from reachtrace.prompts import get_template

    probs = pd.read_parquet(ART / "problems.parquet")
    t = get_template("rlzero31_math")
    probs.assign(prompt=[t.render(x) for x in probs.question]).to_parquet(
        ART / "prompts.parquet")
    n_gpu = int(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
                .stdout.count("GPU "))
    vllm_py = str(Path.home() / "vllm_env" / "bin" / "python")
    procs = [subprocess.Popen([vllm_py, __file__, "rollouts_vllm", str(i), str(n_gpu)],
                              env={**os.environ, "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": str(i)})
             for i in range(n_gpu)]
    if any(p.wait() for p in procs):
        sys.exit("vLLM worker failed")
    r = pd.concat([pd.read_parquet(ART / f"rollouts_{i}.parquet") for i in range(n_gpu)])
    r.sort_values(["problem_id", "sample_idx"]).to_parquet(ART / "rollouts.parquet")


def rollouts_vllm(shard: str, n_shard: str) -> None:
    # Runs in ~/vllm_env, which has vLLM but not this package: no reachtrace imports here.
    from vllm import LLM, SamplingParams

    probs = pd.read_parquet(ART / "prompts.parquet").iloc[int(shard)::int(n_shard)]
    # Full multi-head KV (32 heads) limits concurrency at 7.7k tokens; an fp8 KV cache
    # doubles it. Falls back to bf16 if this vLLM/GPU combination does not support it.
    try:
        llm = LLM(model=RL, dtype="bfloat16", max_model_len=8192, seed=SEED,
                  gpu_memory_utilization=0.9, kv_cache_dtype="fp8")
        kv = "fp8"
    except Exception as e:  # noqa: BLE001
        print(f"fp8 KV cache unavailable ({e}); using bf16", flush=True)
        llm = LLM(model=RL, dtype="bfloat16", max_model_len=8192, seed=SEED,
                  gpu_memory_utilization=0.9)
        kv = "bf16"
    sp = SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0, max_tokens=MAX_NEW,
                        seed=SEED + int(shard), stop=["\nSolve the following"])
    t0 = time.time()
    outs = llm.generate(probs.prompt.tolist(), sp)
    recs = []
    for pid, o in zip(probs.problem_id, outs):
        for j, c in enumerate(o.outputs):
            recs.append({"problem_id": pid, "sample_idx": j,
                         "prompt_ids": list(o.prompt_token_ids),
                         "completion_ids": list(c.token_ids), "text": c.text,
                         "finish_reason": c.finish_reason, "kv_cache": kv})
    pd.DataFrame(recs).to_parquet(ART / f"rollouts_{shard}.parquet")
    n_tok = sum(len(r["completion_ids"]) for r in recs)
    print(f"vLLM shard {shard} ({kv} KV): {len(recs)} rollouts, {n_tok} tokens in "
          f"{time.time() - t0:.0f}s ({n_tok / (time.time() - t0):.0f} tok/s)", flush=True)


def queries() -> None:
    from datasets import Dataset

    from reachtrace.sample.grade import grade

    probs = pd.read_parquet(ART / "problems.parquet").set_index("problem_id")
    r = pd.read_parquet(ART / "rollouts.parquet")
    r["correct"] = [grade(t, probs.answer[p]).correct for t, p in zip(r.text, r.problem_id)]
    r["n_tokens"] = r.completion_ids.map(len)
    Dataset.from_dict({
        "input_ids": [list(a) + list(b) for a, b in zip(r.prompt_ids, r.completion_ids)],
        "labels": [[-100] * len(a) + list(b) for a, b in zip(r.prompt_ids, r.completion_ids)],
    }).save_to_disk(str(ART / "queries"))
    r.drop(columns=["prompt_ids", "completion_ids"]).to_parquet(ART / "queries_meta.parquet")
    r["finished"] = r.finish_reason == "stop"
    s = r.groupby("problem_id").apply(lambda g: pd.Series({
        "n": len(g), "correct": g.correct.sum(),
        "finished_correct": (g.finished & g.correct).sum(),
        "finished_incorrect": (g.finished & ~g.correct).sum(),
        "capped_incorrect": (~g.finished & ~g.correct).sum(),
        "median_tokens_finished": g.n_tokens[g.finished].median()}))
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "queries.csv")
    print(s.to_string())


# ------------------------------------------------------------------------------- docs


def docs() -> None:
    """As olmo7b_midtrain.docs, but each document is kept whole (up to MAX_CHUNKS chunks of
    MAX_DOC tokens) and split into chunks that are scored separately."""
    import io

    import zstandard
    from datasets import Dataset
    from huggingface_hub import HfApi, hf_hub_download

    tree = [f for f in HfApi().list_repo_tree(DOLMINO, repo_type="dataset", path_in_repo="data",
                                              recursive=True)
            if getattr(f, "size", None) and f.path.endswith(".jsonl.zst")]
    tok = _tok()
    rng = np.random.default_rng(SEED)
    docs_rows, chunks = [], []
    for src, grp in GROUP.items():
        files = [f for f in tree if f.path.startswith(f"data/{src}/")]
        size = np.array([f.size for f in files], dtype=float)
        k = min(N_SHARDS, len(files))
        shards = [files[i].path for i in rng.choice(len(files), k, replace=False,
                                                    p=size / size.sum())]
        texts = []
        for shard in shards:
            path = hf_hub_download(DOLMINO, shard, repo_type="dataset")
            with open(path, "rb") as fh:
                reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh),
                                          encoding="utf-8")
                n0 = len(texts)
                for line in reader:
                    t = json.loads(line).get("text", "")
                    if len(t) >= 400:
                        texts.append((shard, t))
                    if len(texts) - n0 >= 5 * N_DOCS:
                        break
        for i in rng.choice(len(texts), size=min(N_DOCS, len(texts)), replace=False):
            shard, text = texts[i]
            ids = tok(text, add_special_tokens=False).input_ids
            d = len(docs_rows)
            n_chunks = min(MAX_CHUNKS, -(-len(ids) // MAX_DOC))
            docs_rows.append({"source": src, "group": grp, "shard": shard,
                              "text": text[:4000], "n_tokens_full": len(ids),
                              "n_chunks": n_chunks})
            for c in range(n_chunks):
                piece = ids[c * MAX_DOC:(c + 1) * MAX_DOC]
                if len(piece) >= 64 or c == 0:
                    chunks.append({"doc": d, "chunk": c, "input_ids": piece})
        print(f"{src}: {len(texts)} read from {k} of {len(files)} files", flush=True)
    ch = pd.DataFrame(chunks)
    ART.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"input_ids": ch.input_ids.tolist(),
                       "labels": ch.input_ids.tolist()}).save_to_disk(str(ART / "chunks"))
    ch.drop(columns=["input_ids"]).assign(n_tokens=ch.input_ids.map(len)).to_parquet(
        ART / "chunks_meta.parquet")
    pd.DataFrame(docs_rows).to_parquet(ART / "docs_meta.parquet")
    print(f"{len(docs_rows)} documents, {len(ch)} chunks, "
          f"{ch.input_ids.map(len).sum()} tokens")


# ------------------------------------------------------------------------------- attribute


def attribute() -> None:
    from reachtrace.attrib import bergson_yaml as by

    if (ART / "trackstar_p32" / "scores").exists():
        return
    common = {"precision": "bf16", "token_batch_size": TOKEN_BATCH, "max_batch_size": None,
              "revision": MIDTRAIN_END}
    # Plain query rows (no advantage column): each column is cos(q_i, chunk).
    secs = _bergson_here("trackstar_p32", [by.trackstar_step(
        str(ART / "trackstar_p32"), BASE, by.data_cfg(str(ART / "chunks")),
        by.data_cfg(str(ART / "queries")), projection_dim=32, **common)])
    _write_json(ART / "timing.json", {"trackstar_p32": secs})


def _bergson_here(name: str, steps: list[dict]) -> float:
    import olmo7b_midtrain as mid

    mid.ART = ART  # its helper writes the YAML under ART
    return _bergson(name, steps)


# ------------------------------------------------------------------------------- evaluate


def _load_scores(path: Path) -> np.ndarray:
    try:
        return _scores(path)
    except ImportError:  # no bergson locally: read the structured memmap directly
        info = json.loads((path / "info.json").read_text())
        raw = np.memmap(path / "scores.bin", dtype=np.dtype(info["dtype"]), mode="r")
        return np.stack([raw[f"score_{j}"].astype(np.float64)
                         for j in range(info["num_scores"])], 1)


def evaluate() -> None:
    from scipy.stats import spearmanr

    OUT.mkdir(parents=True, exist_ok=True)
    qm = pd.read_parquet(ART / "queries_meta.parquet")
    cm = pd.read_parquet(ART / "chunks_meta.parquet")
    dm = pd.read_parquet(ART / "docs_meta.parquet")
    C = _load_scores(ART / "trackstar_p32" / "scores")  # chunks x rollouts: cos
    doc = cm["doc"].to_numpy()
    n_doc = len(dm)
    M = np.zeros((n_doc, len(cm)))
    M[doc, np.arange(len(cm))] = 1
    D_mean = (M @ C) / M.sum(1, keepdims=True)  # Lucia: average the chunk scores
    D_first = C[cm.chunk.to_numpy() == 0][np.argsort(doc[cm.chunk.to_numpy() == 0])]
    pids = qm.problem_id.to_numpy()
    probs = sorted(set(pids))
    correct = qm.correct.to_numpy()
    fin = (qm.finish_reason == "stop").to_numpy()
    ntok = qm.n_tokens.to_numpy().astype(float)
    src = dm.source.to_numpy()
    sources = list(GROUP)
    rng = np.random.default_rng(0)

    def by_source(x):
        return np.array([x[src == s].mean() for s in sources])

    def diff(D, i, lab):
        return D[:, i[lab]].mean(1) - D[:, i[~lab]].mean(1)

    def lenctl(D, i, lab):
        """Coefficient of correctness in a per-document regression of cos on
        [1, correct, length] over the problem's rollouts."""
        L = ntok[i]
        X = np.stack([np.ones(len(i)), lab.astype(float), (L - L.mean()) / (L.std() + 1e-9)], 1)
        return (D[:, i] @ np.linalg.pinv(X).T)[:, 1]

    def plain(D, i, lab):
        return D[:, i].mean(1)

    # Contrast definitions: which rollouts, which labels, which estimator.
    defs = {
        "contrast_all": (np.ones_like(fin), diff),       # the first run's contrast
        "contrast_finished": (fin, diff),                # cut-off rollouts dropped
        "contrast_finished_lenctl": (fin, lenctl),       # and length regressed out
        "finished_vs_capped": (None, diff),              # label = finished, the confound
        "plain_rollout": (np.ones_like(fin), plain),     # no contrast
    }

    def rows_labels(name, p):
        mask, _ = defs[name]
        i = np.flatnonzero(pids == p)
        if name == "finished_vs_capped":
            return i, fin[i]
        i = i[mask[i]]
        return i, correct[i]

    def half_split(lab):
        a = np.zeros(len(lab), bool)
        for v in (True, False):
            j = rng.permutation(np.flatnonzero(lab == v))
            a[j[: len(j) // 2]] = True
        return a

    def within(D, name, p, n=200, shuffle=False):
        i, lab = rows_labels(name, p)
        est = defs[name][1]
        if lab.sum() < 4 or (~lab).sum() < 4:
            return np.nan, np.nan
        rd, rs = [], []
        lb = rng.permutation(lab) if shuffle else lab
        for _ in range(n):
            a = half_split(lb)
            x, y = est(D, i[a], lb[a]), est(D, i[~a], lb[~a])
            rd.append(spearmanr(x, y)[0])
            rs.append(spearmanr(by_source(x), by_source(y))[0])
        return float(np.mean(rd)), float(np.mean(rs))

    rec, full = [], {}
    for chunking, D in [("chunk_mean", D_mean), ("first_chunk", D_first)]:
        for name in defs:
            for p in probs:
                i, lab = rows_labels(name, p)
                rd, rs = within(D, name, p)
                # Null: correctness labels shuffled once within the problem, then the
                # same split procedure.
                nd = [within(D, name, p, n=20, shuffle=True)[0] for _ in range(N_NULL)] \
                    if name != "plain_rollout" else [np.nan]
                few = lab.sum() < 2 or (~lab).sum() < 2
                full[(chunking, name, p)] = (
                    np.full(n_doc, np.nan) if few and name != "plain_rollout"
                    else defs[name][1](D, i, lab))
                rec.append({"chunking": chunking, "query": name, "problem": p,
                            "n_pos": int(lab.sum()), "n_neg": int((~lab).sum()),
                            "within_doc_rho": rd, "within_src_rho": rs,
                            "null_doc_rho": float(np.mean(nd)), "null_doc_sd": float(np.std(nd)),
                            "null_z": (rd - np.mean(nd)) / np.std(nd),
                            "perm_p": (np.nan if np.isnan(rd) or np.isnan(nd).any() else
                                       float((np.sum(np.array(nd) >= rd) + 1) / (len(nd) + 1)))})
    per = pd.DataFrame(rec)
    per["spearman_brown_doc"] = 2 * per.within_doc_rho / (1 + per.within_doc_rho)
    per.to_csv(OUT / "per_problem.csv", index=False)

    summ = {"n_problems": len(probs), "n_rollouts": len(qm), "n_docs": n_doc,
            "n_chunks": len(cm), "tokens_docs": int(cm.n_tokens.sum()),
            "tokens_rollouts": int(qm.n_tokens.sum())}
    for (chunking, name), g in per.groupby(["chunking", "query"]):
        key = f"{chunking}/{name}"
        cr = [spearmanr(full[(chunking, name, p)], full[(chunking, name, q)])[0]
              for a, p in enumerate(probs) for q in probs[a + 1:]
              if not (np.isnan(full[(chunking, name, p)]).any()
                      or np.isnan(full[(chunking, name, q)]).any())]
        summ[key] = {"within_doc_rho": g.within_doc_rho.mean(),
                     "within_src_rho": g.within_src_rho.mean(),
                     "null_doc_rho": g.null_doc_rho.mean(),
                     "cross_problem_doc_rho": float(np.mean(cr)) if cr else None,
                     "n_problems": int(g.within_doc_rho.notna().sum())}
    # How much of the first run's contrast is the finished/cut-off confound, per problem.
    summ["contrast_all_vs_finished_vs_capped_rho"] = float(np.nanmean([
        spearmanr(full[("chunk_mean", "contrast_all", p)],
                  full[("chunk_mean", "finished_vs_capped", p)])[0] for p in probs
        if 0 < fin[pids == p].sum() < (pids == p).sum()]))
    summ["chunk_mean_vs_first_chunk_plain_rho"] = float(np.mean([
        spearmanr(full[("chunk_mean", "plain_rollout", p)],
                  full[("first_chunk", "plain_rollout", p)])[0] for p in probs]))

    # Sources per problem for the length-controlled finished contrast (z over documents),
    # with bootstrap CIs over documents.
    rows = []
    for p in probs:
        x = full[("chunk_mean", "contrast_finished_lenctl", p)]
        if np.isnan(x).any():
            continue
        z = (x - x.mean()) / x.std()
        for s in sources:
            v = z[src == s]
            bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            rows.append({"problem": p, "source": s, "group": GROUP[s], "z": v.mean(),
                         "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
    pd.DataFrame(rows).to_csv(OUT / "sources_by_problem.csv", index=False)
    _write_json(OUT / "summary.json", summ)
    figure(per, summ)
    print(per.to_string())
    print(json.dumps(summ, indent=1, default=float))


def figure(per: pd.DataFrame, summ: dict) -> None:
    """Within-problem split-half reliability per problem against the shuffled-label null and
    against agreement between problems (chunk-mean scores)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = ["#2a78d6", "#1f9e89", "#8e5bd0", "#eb6834"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})
    names = {"contrast_all": "correct − incorrect\nall rollouts\n(as in the first run)",
             "contrast_finished": "correct − incorrect\nfinished only",
             "contrast_finished_lenctl": "correct − incorrect\nfinished only,\nlength controlled",
             "finished_vs_capped": "finished − cut off\n(the confound)",
             "plain_rollout": "plain rollout query\n(no contrast)"}
    p = per[per.chunking == "chunk_mean"]
    probs = sorted(p.problem.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [4, 1.3]})
    for ax, qs in [(axes[0], list(names)[:4]), (axes[1], ["plain_rollout"])]:
        for k, q in enumerate(qs):
            g = p[p["query"] == q].set_index("problem")
            if q != "plain_rollout":
                lo = (g.null_doc_rho - 2 * g.null_doc_sd).min()
                hi = (g.null_doc_rho + 2 * g.null_doc_sd).max()
                ax.fill_between([k - 0.35, k + 0.35], lo, hi, color=grid, lw=0)
            for j, pr in enumerate(probs):
                if pr in g.index and not np.isnan(g.within_doc_rho[pr]):
                    ax.plot(k - 0.24 + 0.16 * j, g.within_doc_rho[pr], "o", color=colors[j],
                            ms=7, mec="white", mew=1.2,
                            label=pr.split("/")[0] + "/" + pr.split("/")[-1].replace(".json", "")
                            if k == 0 else None)
            cr = summ[f"chunk_mean/{q}"]["cross_problem_doc_rho"]
            ax.plot([k - 0.32, k + 0.32], [cr, cr], color=ink, lw=1.6,
                    label="agreement between problems" if k == 0 else None)
        ax.set_xticks(range(len(qs)), [names[q] for q in qs], fontsize=7.5)
        ax.axhline(0, color=muted, lw=0.8)
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Spearman over 1,440 documents")
    axes[0].set_title("Does a problem's contrast replicate across halves of its own rollouts?\n"
                      "dots: within-problem split-half; grey: shuffled-label null (±2 SD); "
                      "line: between problems", fontsize=8.5, loc="left", color=ink)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7,
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5)
    fig.savefig(OUT / "reliability.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


STAGES = {"problems": problems, "rollouts": rollouts, "rollouts_vllm": rollouts_vllm,
          "queries": queries, "docs": docs, "attribute": attribute, "evaluate": evaluate}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: {sys.argv[0]} {{{','.join(STAGES)}}}")
    STAGES[sys.argv[1]](*sys.argv[2:])
