"""Modal jobs for the GPU stages: weight download, vLLM sampling, teacher-forced scoring.

Everything tokenizer-dependent runs remotely, so the local side only handles text, token
IDs and grading. Weights and outputs live on the `reachtrace-vol` volume:

  /vol/hf                     Hugging Face cache (weights downloaded once, CPU-only)
  /vol/artifacts/<run>/...    generation shards and ledger scores

Prompts are passed as token IDs: the rendered prompt is tokenized on its own and any prefix
(an RL cue or a partial RL solution, for the handoff map) is appended as the exact token IDs
the RL model generated, so nothing is re-tokenized across a boundary.

Usage from Python (see scripts/pilot.py):
    with modal.enable_output(), app.run():
        download.remote("allenai/Olmo-3-1025-7B", "main")
"""

from __future__ import annotations

import time

import modal

APP_NAME = "reachtrace"
VOL_PATH = "/vol"
GPU = "H100"
H100_USD_PER_S = 0.001097  # modal.com/pricing, checked 2026-10-26

vol = modal.Volume.from_name("reachtrace-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.32.0",
        "pandas>=2.1",
        "pyarrow>=15",
        "scipy>=1.11",
        "pydantic>=2.6",
        "pyyaml>=6",
        "math-verify>=0.8",
        "huggingface_hub>=0.34",
    )
    # The pip wheel ships no FlashInfer cubins and the image has no nvcc to JIT them, so the
    # FlashInfer top-k/top-p sampler fails at engine warm-up; use vLLM's PyTorch sampler.
    .env(
        {
            "HF_HOME": f"{VOL_PATH}/hf",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .add_local_python_source("reachtrace")
)
app = modal.App(APP_NAME, image=image)
secrets = [modal.Secret.from_name("hf-token")]


def _local_path(repo: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*"],
    )


@app.function(cpu=4, memory=8192, volumes={VOL_PATH: vol}, secrets=secrets, timeout=3600)
def download(repo: str, revision: str) -> dict:
    """Fetch weights into the volume cache. Checks the token exists without printing it."""
    import os

    assert os.environ.get("HF_TOKEN"), "hf-token secret has no HF_TOKEN key"
    t0 = time.time()
    path = _local_path(repo, revision)
    vol.commit()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
    return {"repo": repo, "revision": revision, "path": path, "gb": size / 1e9,
            "seconds": time.time() - t0}


def _llm(repo: str, revision: str, max_model_len: int, eager: bool):
    from vllm import LLM

    path = _local_path(repo, revision)
    return LLM(
        model=path,
        tokenizer=path,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=True,
        enforce_eager=eager,
        max_num_seqs=256,
    )


@app.function(gpu=GPU, volumes={VOL_PATH: vol}, secrets=secrets, timeout=4 * 3600)
def generate(
    repo: str,
    revision: str,
    requests: list[dict],
    out_path: str,
    max_tokens: int = 7680,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_model_len: int = 8192,
    stop: list[str] | None = None,
) -> dict:
    """Sample continuations and write one parquet row per sample.

    requests: dicts with key, prompt (rendered text), n, seed and optional prefix_ids (token
    IDs appended after the tokenized prompt). Output columns: key, sample_idx, text,
    token_ids, n_tokens, finish_reason, prompt_len, prefix_len.
    """
    import os

    import pandas as pd
    from vllm import SamplingParams, TokensPrompt

    t0 = time.time()
    llm = _llm(repo, revision, max_model_len, eager=False)
    tok = llm.get_tokenizer()
    t_load = time.time() - t0
    prompts, params, meta = [], [], []
    for r in requests:
        p_ids = tok.encode(r["prompt"], add_special_tokens=False)
        pre = list(r.get("prefix_ids") or [])
        budget = max(1, min(max_tokens, max_model_len - len(p_ids) - len(pre)))
        prompts.append(TokensPrompt(prompt_token_ids=p_ids + pre))
        params.append(
            SamplingParams(
                n=r["n"], temperature=temperature, top_p=top_p, max_tokens=budget,
                seed=r["seed"], stop=stop or ["\nSolve the following"],
            )
        )
        meta.append((r["key"], len(p_ids), len(pre)))
    outs = llm.generate(prompts, params)
    rows, n_tok = [], 0
    for (key, plen, prelen), out in zip(meta, outs):
        for j, c in enumerate(out.outputs):
            ids = list(c.token_ids)
            n_tok += len(ids)
            rows.append(dict(key=key, sample_idx=j, text=c.text, token_ids=ids, n_tokens=len(ids),
                             finish_reason=c.finish_reason, prompt_len=plen, prefix_len=prelen))
    full = f"{VOL_PATH}/artifacts/{out_path}"
    os.makedirs(os.path.dirname(full), exist_ok=True)
    pd.DataFrame(rows).to_parquet(full, compression="zstd")
    vol.commit()
    dt = time.time() - t0
    return {"out": out_path, "samples": len(rows), "gen_tokens": n_tok, "load_s": t_load,
            "seconds": dt, "tok_per_s": n_tok / max(dt - t_load, 1e-9),
            "usd": dt * H100_USD_PER_S}


@app.function(gpu=GPU, volumes={VOL_PATH: vol}, secrets=secrets, timeout=2 * 3600)
def ledger_scores(
    repo: str, revision: str, checkpoint: str, items: list[dict], opening: int = 16,
    max_model_len: int = 8192,
) -> dict:
    """Teacher-forced per-token log-likelihoods of RL traces under one checkpoint.

    items: dicts with trace_id, prompt (rendered text), completion_ids. Returns one record
    per trace with mean log-likelihood on the opening / content / answer masks.
    """
    from vllm import SamplingParams, TokensPrompt

    from reachtrace.ledger.core import answer_start_token, trace_scores

    t0 = time.time()
    llm = _llm(repo, revision, max_model_len, eager=True)
    tok = llm.get_tokenizer()
    t_load = time.time() - t0
    prompts, meta = [], []
    for it in items:
        p_ids = tok.encode(it["prompt"], add_special_tokens=False)
        c_ids = list(it["completion_ids"])
        if len(p_ids) + len(c_ids) > max_model_len:
            continue
        prompts.append(TokensPrompt(prompt_token_ids=p_ids + c_ids))
        meta.append((it["trace_id"], len(p_ids), c_ids))
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate(prompts, sp)
    records, n_tok = [], 0
    for (tid, plen, c_ids), out in zip(meta, outs):
        plp = out.prompt_logprobs
        lps = [plp[plen + i][c_ids[i]].logprob for i in range(len(c_ids))]
        a = answer_start_token(c_ids, lambda ids: tok.decode(list(ids)))
        rec = trace_scores(lps, a, opening)
        rec.update(trace_id=tid, checkpoint=checkpoint, answer_start=a, n_completion=len(c_ids))
        records.append(rec)
        n_tok += plen + len(c_ids)
    dt = time.time() - t0
    return {"records": records, "skipped": len(items) - len(meta), "tokens": n_tok,
            "load_s": t_load, "seconds": dt, "usd": dt * H100_USD_PER_S}
