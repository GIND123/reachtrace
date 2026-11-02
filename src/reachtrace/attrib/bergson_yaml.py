"""Bergson pipeline YAMLs for the methods recommended for this project (bergson 2.7.0).

Each builder returns a `steps:` entry and starts from the matching reproducible YAML in
Bergson's `examples/` (leaderboard: GPT-2 fine-tuned on WikiText):

* TrackStar: `gpt2_wikitext_leaderboard/trackstar.yaml`. Always autocorrelation
  preconditioning and unit normalization; projection 32 or 64 (LDS 0.19 / 0.28 there,
  against 0.12 at the 16 used by Matlin et al. and in our plan v3).
* EKFAC: `gpt2_wikitext_leaderboard/ekfac.yaml` (LDS 0.45). The factors may be fit on a
  subset of the training data to save compute (a separate `hessian` step on `train[:N]`,
  then `influence` with `resume: true`, as in `pretraining_scale_attribution/build.yaml`).
* EKFAC + ASTRA: `ekfac_astra.yaml` (LDS 0.61; its step size must be tuned per setting) and
  the conjugate-gradient solver `ekfac_astra_cg.yaml` (LDS 0.69).
* Two-stage retrieval: `pretraining_scale_attribution/`: a TrackStar pass over everything,
  then EKFAC on the top candidates only (`score_cfg.candidates`).

Queries are GRPO queries (`attrib.grpo_query`): pre-tokenized rows with an `advantage`
column. Bergson's experimental `query.contrast` is not used.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def data_cfg(dataset: str, split: str = "train", advantage: bool = False) -> dict:
    """A pre-tokenized dataset (input_ids, labels[, advantage]) saved with save_to_disk."""
    cfg = {"dataset": dataset, "split": split}
    if advantage:
        cfg["advantage_column"] = "advantage"
    return cfg


def index_cfg(
    run_path: str, model: str, data: dict, projection_dim: int = 0,
    token_batch_size: int = 4096, precision: str = "fp32",
    filter_modules: str | None = "lm_head", max_batch_size: int | None = 64,
    overwrite: bool = True, **extra,
) -> dict:
    cfg = {"run_path": run_path, "model": model, "precision": precision,
           "token_batch_size": token_batch_size, "overwrite": overwrite, "data": data}
    if projection_dim:
        cfg["projection_dim"] = projection_dim
    if filter_modules:
        cfg["filter_modules"] = filter_modules
    if max_batch_size:
        cfg["max_batch_size"] = max_batch_size
    return {**cfg, **extra}


def trackstar_step(
    run_path: str, model: str, train: dict, query: dict, projection_dim: int = 64,
    mix_hessians: bool = False, stats_sample_size: int | None = 10_000, **index_kw,
) -> dict:
    """TrackStar with one score column per query row (reweight by |A| afterwards).

    `mix_hessians: false` preconditions with the training autocorrelation alone, the
    documented choice when the query set is too small for its own Hessian estimate (a few
    hundred rollouts here).
    """
    return {"trackstar": {
        "index_cfg": index_cfg(run_path, model, train, projection_dim, **index_kw),
        "trackstar_cfg": {
            "stats_sample_size": stats_sample_size,
            "mix_hessians": mix_hessians,
            "query": {**query, "aggregation": "none"},
            "preprocess_cfg": {"unit_normalize": True, "aggregation": "none"},
            "score_cfg": {"batch_size": 1024},
        },
    }}


def grad_cosine_steps(
    run_path: str, model: str, train: dict, query: dict, projection_dim: int = 64, **index_kw,
) -> list[dict]:
    """Projected gradient cosine: TrackStar without the preconditioner (an ablation)."""
    q_path = f"{run_path}/query"
    return [
        {"build": {"index_cfg": index_cfg(q_path, model, {**query}, projection_dim, **index_kw),
                   "preprocess_cfg": {"unit_normalize": True}}},
        {"score": {"score_cfg": {"query_path": q_path, "score": "individual",
                                 "batch_size": 1024},
                   "index_cfg": index_cfg(f"{run_path}/scores", model, train, projection_dim,
                                          **index_kw),
                   "preprocess_cfg": {"unit_normalize": True}}},
    ]


def kfac_hessian_step(run_path: str, model: str, train: dict, subset: int | None = None,
                      **index_kw) -> dict:
    """EK-FAC factors (KFAC + eigenvalue correction), optionally on the first `subset` rows.

    Writes `{run_path}/hessian/kfac`, where `influence` steps with the same run path (or a
    symlinked `hessian/`) read them back with `resume: true`.
    """
    data = dict(train)
    if subset:
        data["split"] = f"{data.get('split', 'train').split('[')[0]}[:{subset}]"
    return {"hessian": {
        "index_cfg": index_cfg(f"{run_path}/hessian/kfac", model, data, **index_kw),
        "hessian_cfg": {"method": "kfac", "ev_correction": True},
    }}


def influence_step(
    run_path: str, model: str, train: dict, query: dict, damping: float = 0.1,
    astra: dict | None = None, candidates: dict | None = None, aggregation: str = "sum",
    score_kw: dict | None = None, **index_kw,
) -> dict:
    """EK-FAC influence of every training row (or only `candidates`) on one query set.

    `score_kw` goes into `score_cfg`, e.g. `query_batch_size` (one pass over the training
    data per batch of queries, for query rows too large to hold on the GPU together).
    """
    score_cfg = {"batch_size": 1024, **(score_kw or {})}
    if candidates:
        score_cfg["candidates"] = candidates
    pipe = {"resume": True, "query": {**query, "aggregation": aggregation},
            "inversion_cfg": {"damping_factor": damping}}
    if astra:
        pipe["astra"] = astra
    return {"influence": {
        "index_cfg": index_cfg(run_path, model, train, overwrite=False, **index_kw),
        "hessian_cfg": {"method": "kfac", "ev_correction": True},
        "score_cfg": score_cfg,
        "preprocess_cfg": {"unit_normalize": False},
        "hessian_pipeline_cfg": pipe,
    }}


# ASTRA settings from ekfac_astra.yaml (momentum SGD) and ekfac_astra_cg.yaml (CG).
ASTRA_SGD = {"num_steps": 300, "momentum": 0.9, "batch_size": 16, "micro_batch_size": 8,
             "lr_decay": 0.9, "lr_decay_interval": 100}
ASTRA_CG = {"solver": "cg", "num_steps": 20, "batch_size": 256, "micro_batch_size": 8}
ASTRA_CG_DAMPING = 10.0  # ekfac_astra_cg.yaml


def astra_cfg(lr: float, **over) -> dict:
    return {**ASTRA_SGD, "lr": lr, **over}


def candidates_cfg(csv_path: str, top_k: int) -> dict:
    """Rows from a `query --record`-format CSV (query, direction, result, result_index,
    score); the EK-FAC store then holds one row per candidate (candidates.npy)."""
    return {"scores": csv_path, "top_k": top_k}


def merge_query_indices(dirs: list[str | Path], out: str | Path) -> Path:
    """Stack single-row aggregated query indices (one per problem) into one index.

    Each input is a `build` with `aggregation: sum`, so its saved preprocess config marks it
    as already aggregated and scoring will not aggregate the stacked rows again. An
    `influence` step pointed at the result (`query.path`, `aggregation: none`) applies the
    inverse Hessian to every row and scores all of them in one pass over the training data.
    """
    import json
    import shutil

    import numpy as np
    from bergson.data import load_gradients
    from datasets import Dataset

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(dirs[0], out)
    rows = np.concatenate([np.asarray(load_gradients(d)) for d in dirs], axis=0)
    info = json.loads((out / "info.json").read_text())
    info.update(num_items=len(dirs), num_grads=len(dirs))
    (out / "info.json").write_text(json.dumps(info))
    mm = np.memmap(out / "gradients.bin", dtype=rows.dtype, mode="w+", shape=rows.shape)
    mm[:] = rows
    mm.flush()
    shutil.rmtree(out / "data.hf", ignore_errors=True)
    Dataset.from_dict({"query_index": list(range(len(dirs)))}).save_to_disk(str(out / "data.hf"))
    return out


def link_hessian(run_dir: str | Path, shared_run_path: str | Path) -> None:
    """Point `{run_dir}/hessian` at the factors fit once under `shared_run_path`."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    link = run_dir / "hessian"
    if not link.exists():
        link.symlink_to(Path(shared_run_path).resolve() / "hessian", target_is_directory=True)


def write_pipeline(path: str | Path, run_path: str, steps: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"run_path": run_path, "steps": copy.deepcopy(steps)},
                                   sort_keys=False))
    return path
