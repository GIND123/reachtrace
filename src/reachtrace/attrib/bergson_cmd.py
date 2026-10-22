"""Bergson command lines (pinned to bergson>=2.3.0; flags verified against v2.4.0 source).

Two scoring routes:
  * stream: `bergson build` a small *query* index, then `bergson score` streams the corpus
    against it and writes scores.bin (no corpus gradient store, so Q x D floats on disk).
  * store:  `bergson build` the corpus index once, then `bergson query`. Use this for the
    norm-only control on a subsample, since streamed cosine scores discard gradient norms.

Prompt/completion rows are wrapped by `tokenizer.apply_chat_template` with user/assistant
roles. The RL-Zero chat template would then drop the "Remember to put your answer..."
line and add stray newlines, so the gradients would be taken in a context the model never
trained on. Always pass `--tokenizer` pointing at `write_identity_tokenizer(...)`
together with a pre-rendered prompt column.
"""

from __future__ import annotations

from pathlib import Path

from ..config import BergsonIndexConfig

IDENTITY_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"


def write_identity_tokenizer(model_id: str, revision: str, out_dir: str | Path) -> Path:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tok.chat_template = IDENTITY_CHAT_TEMPLATE
    tok.save_pretrained(out_dir)
    return Path(out_dir)


def _data_flags(cfg: BergsonIndexConfig) -> list[str]:
    flags = ["--dataset", cfg.dataset, "--prompt_column", cfg.prompt_column]
    if cfg.completion_column:
        flags += ["--completion_column", cfg.completion_column]
    if cfg.reward_column and cfg.advantage_column:
        raise ValueError("reward_column and advantage_column are mutually exclusive in Bergson")
    if cfg.reward_column:
        flags += ["--reward_column", cfg.reward_column]
    if cfg.advantage_column:
        flags += ["--advantage_column", cfg.advantage_column]
    if cfg.truncation:
        flags.append("--truncation")
    return flags


def _model_flags(cfg: BergsonIndexConfig) -> list[str]:
    flags = ["--model", cfg.model, "--revision", cfg.revision, "--precision", cfg.precision]
    return flags + (["--tokenizer", cfg.tokenizer] if cfg.tokenizer else [])


def build_cmd(cfg: BergsonIndexConfig, run_root: str | Path) -> list[str]:
    return [
        "bergson",
        "build",
        str(Path(run_root) / cfg.name),
        *_model_flags(cfg),
        *_data_flags(cfg),
        "--token_batch_size",
        str(cfg.token_batch_size),
        "--projection_dim",
        str(cfg.projection_dim),
        *cfg.extra,
    ]


def score_cmd(
    corpus: BergsonIndexConfig,
    query_index: str | Path,
    run_root: str | Path,
    unit_normalize: bool = True,
    hessian_path: str | None = None,
) -> list[str]:
    """Stream `corpus` against a query index. projection_dim must match the query index."""
    cmd = [
        "bergson",
        "score",
        str(Path(run_root) / f"{corpus.name}__scores"),
        *_model_flags(corpus),
        *_data_flags(corpus),
        "--token_batch_size",
        str(corpus.token_batch_size),
        "--projection_dim",
        str(corpus.projection_dim),
        "--query_path",
        str(query_index),
        "--score",
        "individual",
    ]
    if unit_normalize:
        cmd.append("--unit_normalize")
    if hessian_path:
        cmd += ["--hessian_path", hessian_path]
    return cmd + corpus.extra


def query_cmd(index: str | Path, unit_norm: bool = True, faiss: bool = False) -> list[str]:
    cmd = ["bergson", "query", "--index", str(index)]
    if unit_norm:
        cmd.append("--unit_norm")
    if faiss:
        cmd.append("--faiss")
    return cmd
