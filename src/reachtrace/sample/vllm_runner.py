"""vLLM sampling with deterministic per-(problem, batch) seeds and resumable shards."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ..cache import gen_key, sample_seed
from ..config import DecodeConfig
from ..prompts import PromptTemplate, prompt_sha
from .adaptive import Job
from .grade import grade


def load_llm(model_id: str, revision_sha: str, decode: DecodeConfig, tensor_parallel_size: int = 1):
    from vllm import LLM

    return LLM(
        model=model_id,
        revision=revision_sha,
        dtype="bfloat16",
        max_model_len=decode.max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=True,
    )


def run_jobs(
    llm,
    jobs: list[Job],
    problems: pd.DataFrame,
    *,
    model_id: str,
    revision_sha: str,
    template: PromptTemplate,
    decode: DecodeConfig,
    base_seed: int,
    batch: int,
    cue: str = "",
) -> pd.DataFrame:
    """Generate and grade. `problems` is indexed by problem_id with question/answer/problem_hash."""
    import vllm
    from vllm import SamplingParams

    prompts, params = [], []
    for job in jobs:
        q = problems.loc[job.problem_id]
        prompts.append(template.render(q.question, cue=cue))
        params.append(
            SamplingParams(
                n=job.n,
                temperature=decode.temperature,
                top_p=decode.top_p,
                max_tokens=decode.max_tokens,
                stop=template.stop,
                seed=sample_seed(base_seed, job.problem_id, job.batch_idx),
                logprobs=0,
            )
        )
    outputs = llm.generate(prompts, params)

    run_id, now = uuid4().hex[:12], datetime.now(UTC).isoformat()
    records = []
    for job, prompt, sp, out in zip(jobs, prompts, params, outputs):
        q = problems.loc[job.problem_id]
        psha = prompt_sha(prompt)
        for j, comp in enumerate(out.outputs):
            sample_idx = job.batch_idx * batch + j
            g = grade(comp.text, q.answer)
            n_tok = len(comp.token_ids)
            records.append(
                dict(
                    gen_key=gen_key(
                        model_id=model_id,
                        revision_sha=revision_sha,
                        prompt_sha=psha,
                        problem_hash=q.problem_hash,
                        sample_idx=sample_idx,
                        seed=sp.seed,
                        temperature=decode.temperature,
                        top_p=decode.top_p,
                        max_tokens=decode.max_tokens,
                        stop=template.stop,
                        vllm_version=vllm.__version__,
                    ),
                    model_id=model_id,
                    revision_sha=revision_sha,
                    problem_id=job.problem_id,
                    problem_hash=q.problem_hash,
                    prompt_template_id=template.template_id,
                    prompt_sha=psha,
                    cue=cue,
                    sample_idx=sample_idx,
                    seed=sp.seed,
                    temperature=decode.temperature,
                    top_p=decode.top_p,
                    max_tokens=decode.max_tokens,
                    text=comp.text,
                    n_tokens=n_tok,
                    finish_reason=comp.finish_reason,
                    extracted_answer=g.extracted,
                    correct=g.correct,
                    grade_method=g.method,
                    mean_logprob=(comp.cumulative_logprob / n_tok)
                    if comp.cumulative_logprob is not None and n_tok
                    else None,
                    vllm_version=vllm.__version__,
                    run_id=run_id,
                    timestamp=now,
                )
            )
    return pd.DataFrame.from_records(records)


def write_shard(df: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{df.run_id.iloc[0]}.parquet"
    df.to_parquet(path, compression="zstd")
    return path


def load_counts(gen_dir: str | Path, problem_ids: list[str]) -> pd.DataFrame:
    """Per-problem (n, c) over all shards in a model's generation directory."""
    files = sorted(Path(gen_dir).glob("*.parquet"))
    if not files:
        return pd.DataFrame({"problem_id": problem_ids, "n": 0, "c": 0})
    gens = pd.concat(
        [pd.read_parquet(f, columns=["gen_key", "problem_id", "correct"]) for f in files]
    ).drop_duplicates("gen_key")
    agg = gens.groupby("problem_id").correct.agg(n="size", c="sum")
    return agg.reindex(problem_ids, fill_value=0).rename_axis("problem_id").reset_index()
