"""Synthetic gate v3: which Bergson method recovers ground truth under a CPT -> GRPO flow?

Stages (run in order; each reads the previous stage's artifacts under $GATE_ART):

  prepare     planted corpus (A-D operations, M name decoys, F format-only, W worked
              traces) plus generic FineWeb-Edu text; RL / held-out task sets
  cpt         continued pretraining of SmolLM2 on the corpus (deterministic, weighted)
  grpo        GRPO from the CPT model on A-then-B and C-then-D prompts; every rollout logged
  queries     RL-model rollouts on held-out A-then-B problems -> GRPO query rows
  attribute   Bergson methods (TrackStar p16/32/64, gradient cosine, EK-FAC full-fit and
              subset-fit, EK-FAC+ASTRA, EK-FAC+CG, two-stage TrackStar->EK-FAC) and BM25
  groundtruth dose-design retrains (LDS), top-k removal retrains (QLD), leave-family-out
  evaluate    tables and figures into results/gate_v3/

    GATE_ART=artifacts/gate_v3 python scripts/gate_v3.py prepare
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reachtrace.synthetic.tasks import make_ops
from reachtrace.synthetic.v3 import FAMILIES, disjoint_task_sets, make_corpus_v3

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("GATE_ART", ROOT / "artifacts" / "gate_v3")).resolve()
OUT = ROOT / "results" / ART.name  # results/gate_v3, results/gate_v3_smooth, ...
CFG = {
    "model": os.environ.get("GATE_MODEL", "HuggingFaceTB/SmolLM2-135M"),
    "seed": 0,
    "docs_per_family": int(os.environ.get("GATE_DOCS", 600)),
    "n_generic": int(os.environ.get("GATE_GENERIC", 3000)),
    "max_len": 256,
    "train_lengths": [3, 4],
    "eval_lengths": [5, 6],
    "n_rl_ab": 1000,
    "n_rl_single": 1000,
    "n_rl_cd": 500,
    "n_heldout": 64,
    "grpo_steps": int(os.environ.get("GATE_GRPO_STEPS", 400)),
    "grpo_lr": float(os.environ.get("GATE_GRPO_LR", 1e-5)),
    "query_samples": 8,
    "max_queries": int(os.environ.get("GATE_MAX_QUERIES", 16)),
}


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CFG["model"])


def _tasks() -> dict[str, list[dict]]:
    return json.loads((ART / "tasks.json").read_text())


# ------------------------------------------------------------------------------- prepare


def generic_texts(n: int, seed: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
                      streaming=True).shuffle(seed=seed, buffer_size=10_000)
    out = []
    for r in ds:
        if len(r["text"]) >= 400:
            out.append(r["text"][:2000])
        if len(out) >= n:
            break
    return out


def prepare() -> None:
    from datasets import Dataset

    from reachtrace.synthetic.cpt import tokenize_docs

    ops = make_ops(CFG["seed"])
    planted = make_corpus_v3(ops, CFG["docs_per_family"], seed=CFG["seed"])
    generic = [{"doc_id": f"G-{i}", "family": "G", "text": t}
               for i, t in enumerate(generic_texts(CFG["n_generic"], CFG["seed"]))]
    docs = planted + generic
    random.Random(CFG["seed"]).shuffle(docs)
    tok = _tok()
    ids = tokenize_docs(tok, [d["text"] for d in docs], CFG["max_len"])
    ART.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({
        "input_ids": ids, "labels": ids,
        "doc_id": [d["doc_id"] for d in docs], "family": [d["family"] for d in docs],
    }).save_to_disk(str(ART / "corpus"))
    pd.DataFrame(docs).to_json(ART / "corpus.jsonl", orient="records", lines=True)
    n_tok = sum(map(len, ids))
    print(f"{len(docs)} docs ({len(planted)} planted, {len(generic)} generic), {n_tok} tokens")
    print(pd.Series([d["family"] for d in docs]).value_counts().to_dict())
    make_tasks()


def make_tasks() -> None:
    """RL prompts and held-out sets (disjoint inputs within each chain).

    GRPO only learns from groups with mixed rewards. After CPT, A-then-B is solved ~1% of
    the time, so A-then-B prompts alone give almost no signal; the single operations A and B
    sit at the edge of competence (~10% and ~50%) and carry it, as in curricula that train
    at the edge of competence. C-then-D (solved almost always after CPT) is the trained but
    off-target RL item type.
    """
    ops = make_ops(CFG["seed"])
    lt, le = CFG["train_lengths"], CFG["eval_lengths"]
    ab = disjoint_task_sets(ops, "AB", lt, {"rl": CFG["n_rl_ab"], "heldout_short":
                                            CFG["n_heldout"]}, CFG["seed"] + 1)
    ab_long = disjoint_task_sets(ops, "AB", le, {"heldout_long": CFG["n_heldout"]},
                                 CFG["seed"] + 2)
    cd = disjoint_task_sets(ops, "CD", lt, {"rl": CFG["n_rl_cd"], "heldout_short":
                                            CFG["n_heldout"]}, CFG["seed"] + 3)
    tasks = {"rl_ab": ab["rl"], "rl_cd": cd["rl"], "heldout_ab_short": ab["heldout_short"],
             "heldout_ab_long": ab_long["heldout_long"], "heldout_cd_short": cd["heldout_short"]}
    for i, f in enumerate("ABCD"):
        s = disjoint_task_sets(ops, f, lt, {"rl": CFG["n_rl_single"], "x": 64},
                               CFG["seed"] + 10 + i)
        tasks[f"single_{f}"] = s["x"]
        if f in "AB":
            tasks[f"rl_{f.lower()}"] = s["rl"]
    _write_json(ART / "tasks.json", tasks)
    print({k: len(v) for k, v in tasks.items()})


def _rl_prompts() -> list[dict]:
    t = _tasks()
    rl = t["rl_ab"] + t["rl_a"] + t["rl_b"] + t["rl_cd"]
    random.Random(CFG["seed"]).shuffle(rl)
    return rl


def _corpus():
    from datasets import load_from_disk

    return load_from_disk(str(ART / "corpus"))


# ------------------------------------------------------------------------------- CPT + skill


def skill_check(model_dir: str, label: str, n: int = 8) -> pd.DataFrame:
    """pass@1 / pass@n on single operations and on held-out compositions."""
    import torch
    from transformers import AutoModelForCausalLM

    from reachtrace.synthetic.rollout import sample_rollouts

    tok = _tok()
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(_device())
    rows = []
    for split, tasks in _tasks().items():
        if split.startswith("rl_"):
            continue
        ro = sample_rollouts(model, tok, tasks[:64], n=n, seed=CFG["seed"])
        g = ro.groupby("task_id").reward
        rows.append({"model": label, "split": split, "pass1": g.mean().mean(),
                     f"pass{n}": g.max().mean(), "finished": ro.finished.mean()})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return df


def cpt() -> None:
    from reachtrace.synthetic.cpt import CPTConfig, train_cpt

    ds = _corpus()
    t0 = time.time()
    _, log = train_cpt(CFG["model"], ds["input_ids"], np.ones(len(ds)), CPTConfig(),
                       out_dir=ART / "models" / "cpt", device=_device(),
                       verbose=True)
    _tok().save_pretrained(ART / "models" / "cpt")
    print(f"CPT {log[-1]['step'] + 1} steps in {time.time() - t0:.0f}s; "
          f"loss {log[0]['loss']:.3f} -> {log[-1]['loss']:.3f}")
    base = skill_check(CFG["model"], "base")
    after = skill_check(str(ART / "models" / "cpt"), "cpt")
    pd.concat([base, after]).to_csv(ART / "skill_cpt.csv", index=False)


# ------------------------------------------------------------------------------- GRPO


def grpo() -> None:
    from reachtrace.synthetic.train import grpo as run_grpo

    rl = _rl_prompts()
    t0 = time.time()
    run_grpo(
        ART / "models" / "cpt", rl, ART / "models" / "rl", ART / "rl" / "rollouts.jsonl",
        seed=CFG["seed"], max_steps=CFG["grpo_steps"], lr=CFG["grpo_lr"],
        max_completion_length=64, save_steps=min(100, CFG["grpo_steps"]),
        per_device_train_batch_size=64,
        lr_scheduler_type="constant", bf16=True,
    )
    print(f"GRPO {CFG['grpo_steps']} steps in {time.time() - t0:.0f}s")
    final = ART / "models" / "rl" / f"checkpoint-{CFG['grpo_steps']}"
    _tok().save_pretrained(final)
    df = skill_check(str(final), "rl")
    df.to_csv(ART / "skill_rl.csv", index=False)


def skill() -> None:
    """Accuracy of base, CPT and final RL models on every held-out set."""
    dfs = [skill_check(CFG["model"], "base"), skill_check(str(ART / "models" / "cpt"), "cpt"),
           skill_check(str(rl_final()), "rl")]
    pd.concat(dfs).to_csv(ART / "skill.csv", index=False)


def rl_final() -> Path:
    return ART / "models" / "rl" / f"checkpoint-{CFG['grpo_steps']}"


# ------------------------------------------------------------------------------- queries


def queries() -> None:
    """GRPO query rows: RL-model rollouts on held-out A-then-B problems with mixed rewards."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM

    from reachtrace.attrib.grpo_query import query_rows
    from reachtrace.synthetic.rollout import sample_rollouts

    tok = _tok()
    t = _tasks()
    splits = os.environ.get("GATE_QUERY_SPLITS", "heldout_ab_long,heldout_ab_short")
    pool = [x for sp in splits.split(",") for x in t[sp]]
    rl = AutoModelForCausalLM.from_pretrained(rl_final(), dtype=torch.bfloat16).to(_device())
    ro = sample_rollouts(rl, tok, pool, n=CFG["query_samples"], seed=CFG["seed"] + 5)
    ro["problem_id"] = ro.task_id
    rows = query_rows(ro)
    # Prefer long-input problems (base-unreachable, the G analogue), then short ones.
    order = {tid: i for i, tid in enumerate(x["task_id"] for x in pool)}
    keep = sorted(rows.problem_id.unique(), key=order.get)[: CFG["max_queries"]]
    rows = rows[rows.problem_id.isin(keep)].sort_values(["problem_id", "sample_idx"])
    rows = rows.reset_index(drop=True)
    Dataset.from_dict({
        "input_ids": rows.input_ids.tolist(), "labels": rows.labels.tolist(),
        "advantage": rows.advantage.astype(float).tolist(),
        "problem_id": rows.problem_id.tolist(), "reward": rows.reward.tolist(),
    }).save_to_disk(str(ART / "queries"))
    rows.drop(columns=["input_ids", "labels"]).to_json(ART / "queries.jsonl", orient="records",
                                                        lines=True)
    ro.drop(columns=["prompt_ids"]).to_json(ART / "query_pool_rollouts.jsonl",
                                           orient="records", lines=True)
    k = rows.groupby("problem_id").reward.agg(["sum", "size"])
    print(f"{len(keep)} query problems, {len(rows)} rows; correct per problem "
          f"{k['sum'].tolist()} of {k['size'].tolist()}")


# ------------------------------------------------------------------------------- attribution


# GATE_QUERY_SET=plain attributes the plain objective (correct rollouts only, advantage +1)
# instead of the GRPO query, into scores_plain/; ground truth always uses the GRPO rows.
QS = "_plain" if os.environ.get("GATE_QUERY_SET") == "plain" else ""


def _query_meta() -> pd.DataFrame:
    return pd.read_json(ART / f"queries{QS}.jsonl", lines=True)


def _problem_ranges() -> list[tuple[str, int, int]]:
    """Row range of each problem in the (problem-sorted) query dataset."""
    q = _query_meta()
    out, start = [], 0
    for pid, n in q.groupby("problem_id", sort=False).size().items():
        out.append((pid, start, start + n))
        start += n
    return out


def _bergson(yaml_path: Path) -> float:
    cmd = [sys.executable, "-m", "bergson", str(yaml_path)]
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def _load(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    from bergson.data import load_scores

    s = load_scores(path)
    return np.asarray(s[:], dtype=np.float64).reshape(len(s), -1), s.candidates


def _save_scores(name: str, scores: np.ndarray, seconds: float, **meta) -> None:
    d = ART / f"scores{QS}"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}.npy", scores)
    _write_json(d / f"{name}.json", {"seconds": seconds, **meta})
    print(f"[{name}] scores {scores.shape} in {seconds:.0f}s", flush=True)


def _done(name: str) -> bool:
    return (ART / f"scores{QS}" / f"{name}.npy").exists()


def attribute(only: str = "") -> None:
    from reachtrace.attrib import bergson_yaml as by
    from reachtrace.attrib.grpo_query import per_problem, reweight_columns
    from reachtrace.attrib.lexical import BM25

    model = str(ART / "models" / "cpt")
    root = ART / f"bergson{QS}"
    train = by.data_cfg(str(ART / "corpus"))
    query = by.data_cfg(str(ART / f"queries{QS}"), advantage=True)
    qm = _query_meta()
    ranges = _problem_ranges()
    n_docs = len(_corpus())
    want = set(only.split(",")) if only else None
    def run(name: str) -> bool:
        return (want is None or name in want) and not _done(name)

    tbs = int(os.environ.get("GATE_TOKEN_BATCH", 4096))
    mbs = int(os.environ.get("GATE_EKFAC_BATCH", 16))

    # BM25 on text: the problem statement plus its correct rollouts.
    if run("bm25"):
        t0 = time.time()
        docs = pd.read_json(ART / "corpus.jsonl", lines=True).text.tolist()
        tok = _tok()
        bm = BM25(docs)
        cols = []
        for _, a, b in ranges:
            rows = qm.iloc[a:b]
            prompt = tok.decode(rows.iloc[0].prompt_ids)
            text = prompt + " " + " ".join(rows[rows.reward > 0].completion)
            cols.append(bm.scores(text))
        _save_scores("bm25", np.column_stack(cols), time.time() - t0)

    # TrackStar and its unpreconditioned ablation: one column per rollout, |A| reweighted.
    for p, mix in [(16, False), (32, False), (64, False), (64, True)]:
        name = f"trackstar_p{p}" + ("_mix" if mix else "")
        if not run(name):
            continue
        rp = str(root / name)
        y = by.write_pipeline(root / f"{name}.yaml", rp, [by.trackstar_step(
            rp, model, train, query, projection_dim=p, mix_hessians=mix,
            token_batch_size=tbs)])
        sec = _bergson(y)
        raw, _ = _load(root / name / "scores")
        sc, _ = per_problem(reweight_columns(raw, qm.advantage.to_numpy()), qm.problem_id)
        _save_scores(name, sc, sec, projection_dim=p, mix_hessians=mix)
    if run("gradcos_p64"):
        rp = str(root / "gradcos_p64")
        y = by.write_pipeline(root / "gradcos_p64.yaml", rp, by.grad_cosine_steps(
            rp, model, train, query, 64, token_batch_size=tbs))
        sec = _bergson(y)
        raw, _ = _load(root / "gradcos_p64" / "scores")
        sc, _ = per_problem(reweight_columns(raw, qm.advantage.to_numpy()), qm.problem_id)
        _save_scores("gradcos_p64", sc, sec, projection_dim=64)

    # EK-FAC. Each problem's query gradient q_p = sum_i A_i g_i is built once (a `build` with
    # aggregation "sum"); the 16 rows are merged into one query index, so every EK-FAC
    # variant scores all problems in a single pass over the training data (the same scores
    # as one influence run per problem: EK-FAC scores are linear in each query column).
    qdir = root / "query_rows"

    def merged_query() -> str:
        out = qdir / "merged"
        if not (out / "info.json").exists():
            steps = [{"build": {
                "index_cfg": by.index_cfg(str(qdir / f"p{j:02d}"), model, by.data_cfg(
                    str(ART / f"queries{QS}"), split=f"train[{a}:{b}]", advantage=True),
                    token_batch_size=tbs, max_batch_size=mbs),
                "preprocess_cfg": {"aggregation": "sum", "unit_normalize": False}}}
                for j, (_, a, b) in enumerate(ranges)]
            _bergson(by.write_pipeline(root / "query_rows.yaml", str(qdir), steps))
            by.merge_query_indices([qdir / f"p{j:02d}" for j in range(len(ranges))], out)
        return str(out)

    def ekfac_family(name: str, subset: int | None, astra: dict | None = None,
                     damping: float = 0.1, hessian_from: str | None = None,
                     candidates: str | None = None, top_k: int = 0,
                     own: dict[int, np.ndarray] | None = None) -> None:
        shared = root / (hessian_from or f"{name}_factors")
        sec = 0.0
        if not (shared / "hessian" / "kfac").exists():
            y = by.write_pipeline(root / f"{shared.name}.yaml", str(shared), [
                by.kfac_hessian_step(str(shared), model, train, subset=subset,
                                     token_batch_size=tbs, max_batch_size=mbs)])
            sec += _bergson(y)
        t0 = time.time()
        q = {"path": merged_query()}
        rp = root / name
        by.link_hessian(rp, shared)
        cand = by.candidates_cfg(candidates, top_k) if candidates else None
        y = by.write_pipeline(root / f"{name}.yaml", str(rp), [by.influence_step(
            str(rp), model, train, q, damping=damping, astra=astra, candidates=cand,
            aggregation="none", token_batch_size=tbs, max_batch_size=mbs)])
        _bergson(y)
        sec += time.time() - t0  # includes the shared query build on its first use
        s, cand_rows = _load(rp / "scores")
        if cand_rows is not None:  # two-stage: each problem keeps EK-FAC on its own candidates
            full = np.full((n_docs, s.shape[1]), np.nan)
            full[cand_rows] = s
            for j in range(full.shape[1]):
                mask = np.ones(n_docs, bool)
                mask[own[j]] = False
                full[mask, j] = np.nan
            s = full
        _save_scores(name, s, sec, subset=subset, astra=astra, damping=damping, top_k=top_k)

    if run("ekfac"):
        ekfac_family("ekfac", None)
    sub = int(os.environ.get("GATE_EKFAC_SUBSET", n_docs // 4))
    if run("ekfac_sub"):
        ekfac_family("ekfac_sub", sub)
    if run("ekfac_cg"):
        ekfac_family("ekfac_cg", None, astra=by.ASTRA_CG, damping=by.ASTRA_CG_DAMPING,
                     hessian_from="ekfac_factors")
    for lr in [float(x) for x in os.environ.get("GATE_ASTRA_LRS", "1e-5,3e-6,1e-6").split(",")]:
        name = f"ekfac_astra_lr{lr:.0e}"
        if run(name):
            ekfac_family(name, None, hessian_from="ekfac_factors", astra=by.astra_cfg(
                lr, num_steps=int(os.environ.get("GATE_ASTRA_STEPS", 300))))

    # Two-stage retrieval: TrackStar over everything, EK-FAC on each problem's top 10%.
    # The first stage is p32 by default: p64's per-module autocorrelation Hessians for the
    # 210 modules of SmolLM2-135M need more than a 40 GB GPU.
    first = os.environ.get("GATE_TWOSTAGE_FIRST", "trackstar_p32")
    if run("twostage") and _done(first):
        ts = np.load(ART / f"scores{QS}" / f"{first}.npy")
        k = max(1, n_docs // 10)
        own = {j: np.argsort(-ts[:, j])[:k] for j in range(ts.shape[1])}
        path = root / "twostage_candidates.csv"
        pd.concat([pd.DataFrame({"query": j, "direction": "Top", "result": "",
                                 "result_index": top, "score": ts[top, j]})
                   for j, top in own.items()]).to_csv(path, index=False)
        ekfac_family("twostage", None, hessian_from="ekfac_factors", candidates=str(path),
                     top_k=k, own=own)


# ------------------------------------------------------------------------------- ground truth

GROUPS = list(FAMILIES) + ["G"]


def _query_rows() -> tuple[list[dict], np.ndarray, np.ndarray]:
    from datasets import load_from_disk

    q = load_from_disk(str(ART / "queries"))
    rows = [{"input_ids": r["input_ids"], "labels": r["labels"]} for r in q]
    return rows, np.asarray(q["advantage"]), np.asarray(q["problem_id"])


def _score_J(model, return_lp: bool = False):
    """J per problem; with `return_lp`, also every query row's summed log-prob, so other
    objectives over the same rollouts (e.g. correct rollouts only) can be computed later."""
    from reachtrace.synthetic.cpt import grpo_objective, sequence_logprobs

    rows, adv, prob = _query_rows()
    lp, _ = sequence_logprobs(model, rows, device=_device())
    J = grpo_objective(lp, adv, prob)
    return (J, lp) if return_lp else J


def _retrain_J(weights: np.ndarray, name: str, kind: str) -> dict | None:
    """Retrain CPT with `weights` and record J per problem (cached on disk by name).

    Before the query rollouts exist, the retrained weights are saved instead and
    `groundtruth finalize` scores them later, so retraining can overlap GRPO.
    """
    import torch

    from reachtrace.synthetic.cpt import CPTConfig, train_cpt

    path = ART / "groundtruth" / kind / f"{name}.json"
    pending = ART / "retrained" / kind / f"{name}.pt"
    if path.exists() or pending.exists():
        return json.loads(path.read_text()) if path.exists() else None
    t0 = time.time()
    model, _ = train_cpt(CFG["model"], _corpus()["input_ids"], weights, CPTConfig(),
                         device=_device())
    meta = {"name": name, "kept": float((weights != 0).mean()), "seconds": time.time() - t0}
    if not (ART / "queries").exists():
        pending.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "meta": meta}, pending)
        print(f"[{kind}/{name}] saved for scoring ({meta['seconds']:.0f}s)", flush=True)
        return None
    J, lp = _score_J(model, return_lp=True)
    out = {**meta, "J": J.to_dict(), "lp": lp.tolist()}
    _write_json(path, out)
    print(f"[{kind}/{name}] J_total {J.sum():.3f} ({out['seconds']:.0f}s)", flush=True)
    return out


def finalize() -> None:
    """Score J for retrained models saved before the queries existed, then delete them."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(CFG["model"], dtype=torch.float32).to(_device())
    for pt in sorted((ART / "retrained").glob("*/*.pt")):
        blob = torch.load(pt, map_location=_device())
        model.load_state_dict(blob["state"])
        model.eval()
        J, lp = _score_J(model, return_lp=True)
        _write_json(ART / "groundtruth" / pt.parent.name / f"{pt.stem}.json",
                    {**blob["meta"], "J": J.to_dict(), "lp": lp.tolist()})
        pt.unlink()
        print(f"[{pt.parent.name}/{pt.stem}] J_total {J.sum():.3f}", flush=True)


def _shard(items: list, shard: str) -> list:
    i, n = (int(x) for x in shard.split("/"))
    return items[i::n]


def groundtruth(kind: str, shard: str = "0/1") -> None:
    from reachtrace.synthetic.v3 import dose_design, dose_weights

    fam = np.asarray(_corpus()["family"])
    n = len(fam)
    if kind == "noise":  # determinism: the full run, twice
        for r in _shard(["full_a", "full_b"], shard):
            _retrain_J(np.ones(n), r, "noise")
    elif kind == "dose":
        design = dose_design(GROUPS, int(os.environ.get("GATE_DOSE_SUBSETS", 40)),
                             seed=CFG["seed"])
        np.save(ART / "dose_design.npy", design)
        jobs = list(range(len(design)))
        for s in _shard(jobs, shard):
            w = dose_weights(fam, dict(zip(GROUPS, design[s])), seed=CFG["seed"])
            _retrain_J(w, f"s{s:02d}", "dose")
    elif kind == "leave1":  # Bergson leaderboard protocol: random leave-1%-out retrains
        rng = np.random.default_rng(CFG["seed"] + 7)
        k = max(1, round(0.01 * n))
        masks = np.ones((int(os.environ.get("GATE_LEAVE1_SUBSETS", 40)), n), dtype=bool)
        for row in masks:
            row[rng.choice(n, k, replace=False)] = False
        np.save(ART / "leave1_masks.npy", masks)
        for s in _shard(list(range(len(masks))), shard):
            _retrain_J(masks[s].astype(float), f"l{s:02d}", "leave1")
        if shard.startswith("0/") and not (ART / "groundtruth" / "leave1" / "full.json").exists():
            _retrain_J(np.ones(n), "full", "leave1")  # reference with per-row log-probs
    elif kind == "smooth":  # is J a smooth function of the data weights in this regime?
        from reachtrace.synthetic.cpt import CPTConfig

        c = CPTConfig()
        tag = f"lr{c.lr:g}_m{c.micro_docs}"
        rng = np.random.default_rng(123)
        jobs = [("full", np.ones(n))]
        for i in range(2):  # drop one random document
            w = np.ones(n)
            w[rng.integers(n)] = 0.0
            jobs.append((f"minus1_{i}", w))
        for name, w in _shard(jobs, shard):
            _retrain_J(w, f"{tag}_{name}", "smooth")
    elif kind == "random":  # TRAK-style random half subsets, document level
        rng = np.random.default_rng(CFG["seed"] + 99)
        masks = rng.random((int(os.environ.get("GATE_RANDOM_SUBSETS", 24)), n)) < 0.5
        np.save(ART / "random_masks.npy", masks)
        for s in _shard(list(range(len(masks))), shard):
            _retrain_J(masks[s].astype(float), f"r{s:02d}", "random")
    elif kind == "qld":
        frac = float(os.environ.get("GATE_QLD_FRAC", 0.01))
        k = max(1, round(frac * n))
        tag = f"top{frac:g}"
        jobs = [("random", i) for i in range(3)]
        jobs += [(p.stem, 0) for p in sorted((ART / "scores").glob("*.npy"))]
        for method, i in _shard(jobs, shard):
            if method == "random":
                drop = np.random.default_rng(1000 + i).choice(n, k, replace=False)
            else:
                s = np.load(ART / "scores" / f"{method}.npy")
                agg = np.nan_to_num(s, nan=-np.inf).sum(1) if method == "twostage" else s.sum(1)
                drop = np.argsort(-agg)[:k]
            w = np.ones(n)
            w[drop] = 0.0
            _retrain_J(w, f"{tag}_{method}_{i}", "qld")
    elif kind == "lfo":  # leave-family-out CPT then GRPO: each family's role in the RL gain
        lfo(shard)
    elif kind == "finalize":
        finalize()
    else:
        raise ValueError(kind)


def lfo(shard: str = "0/1") -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from reachtrace.synthetic.cpt import CPTConfig, train_cpt
    from reachtrace.synthetic.rollout import sample_rollouts
    from reachtrace.synthetic.train import grpo as run_grpo

    fam = np.asarray(_corpus()["family"])
    t = _tasks()
    steps = int(os.environ.get("GATE_LFO_STEPS", CFG["grpo_steps"]))
    tok = _tok()
    # Leave a CPT family out (then the usual GRPO), or keep CPT and leave an RL prompt type
    # out of GRPO (ground truth for RL-item attribution).
    jobs = [("family", f) for f in ["none", "A", "B", "C", "D", "M", "F", "W"]]
    cpt_only = os.environ.get("GATE_LFO_CPT_ONLY") == "1"  # J at the CPT checkpoint only
    if not cpt_only:
        jobs += [("rltype", c) for c in ["AB", "A", "B"]]
    for mode, f in _shard(jobs, shard):
        tag = f"minus_{f}" if mode == "family" else f"rl_minus_{f}"
        path = ART / "groundtruth" / "lfo" / f"{tag}.json"
        if path.exists():
            continue
        t0 = time.time()
        d = ART / "lfo" / tag
        if mode == "family":
            w = (fam != f).astype(float)
            train_cpt(CFG["model"], _corpus()["input_ids"], w, CPTConfig(), out_dir=d / "cpt",
                      device=_device())
            tok.save_pretrained(d / "cpt")
            cpt_dir, rl = d / "cpt", _rl_prompts()
        else:
            cpt_dir = ART / "models" / "cpt"
            rl = [x for x in _rl_prompts() if x["chain"] != f]
        stages = [("cpt", cpt_dir)]
        if not cpt_only:
            run_grpo(cpt_dir, rl, d / "rl", d / "rollouts.jsonl", seed=CFG["seed"],
                     max_steps=steps, lr=CFG["grpo_lr"], max_completion_length=64,
                     save_steps=steps, per_device_train_batch_size=64,
                     lr_scheduler_type="constant", bf16=True)
            stages.append(("rl", d / "rl" / f"checkpoint-{steps}"))
        res = {"mode": mode, "removed": f}
        for stage, md in stages:
            m = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float32).to(_device())
            for split in ["heldout_ab_short", "heldout_ab_long", "heldout_cd_short",
                          "single_A", "single_B"]:
                ro = sample_rollouts(m, tok, t[split][:64], n=8, seed=CFG["seed"] + 7)
                res[f"{stage}_{split}"] = float(ro.reward.mean())
            if (ART / "queries").exists():
                res[f"{stage}_J"] = float(_score_J(m).sum())
            del m
        res["seconds"] = time.time() - t0
        _write_json(path, res)
        shutil.rmtree(d / "rl", ignore_errors=True)  # checkpoints are large and not reused
        print(f"[lfo] {tag}: {res}", flush=True)


# ------------------------------------------------------------------------------- RL items


RL_WINDOW = (50, 150)  # GRPO steps whose rollouts are attributed; A-then-B is learned here
RL_CKPT = 100          # their gradients are taken at this checkpoint (TracIn-CP style)


def rl_items() -> None:
    """RL-stage items: logged GRPO rollouts (with their advantages) from RL_WINDOW, plus
    placebo rollouts sampled at the same checkpoint on prompts GRPO never trained on."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM

    from reachtrace.attrib.grpo_query import dr_grpo_advantage
    from reachtrace.synthetic.rollout import sample_rollouts

    tok = _tok()
    t = _tasks()
    prompts = {x["task_id"]: x for k in ("rl_ab", "rl_a", "rl_b", "rl_cd") for x in t[k]}
    log = pd.read_json(ART / "rl" / "rollouts.jsonl", lines=True)
    log = log[(log.step >= RL_WINDOW[0]) & (log.step < RL_WINDOW[1])].copy()
    log["group"] = log.step.astype(str) + "/" + log.call.astype(str) + "/" + log.task_id
    log["advantage"] = dr_grpo_advantage(log.reward, log.group)
    log = log[log.advantage != 0]
    log["type"] = log.task_id.map(lambda i: prompts[i]["chain"])
    log["prompt_ids"] = log.task_id.map(
        lambda i: tok(prompts[i]["prompt"], add_special_tokens=False).input_ids)
    ckpt = ART / "models" / "rl" / f"checkpoint-{RL_CKPT}"
    m = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).to(_device())
    placebo = []
    for split, typ in [("heldout_ab_short", "placebo_AB"), ("single_A", "placebo_A"),
                       ("single_B", "placebo_B")]:
        ro = sample_rollouts(m, tok, t[split], n=8, seed=CFG["seed"] + 11)
        ro["group"] = "placebo/" + ro.task_id
        ro["advantage"] = dr_grpo_advantage(ro.reward, ro.group)
        ro["type"] = typ
        placebo.append(ro[ro.advantage != 0])
    items = pd.concat([log, *placebo], ignore_index=True)
    items["input_ids"] = [list(p) + list(c) for p, c in zip(items.prompt_ids,
                                                             items.completion_ids)]
    items["labels"] = [[-100] * len(p) + list(c) for p, c in zip(items.prompt_ids,
                                                                  items.completion_ids)]
    Dataset.from_dict({"input_ids": items.input_ids.tolist(), "labels": items.labels.tolist(),
                       "advantage": items.advantage.astype(float).tolist()}).save_to_disk(
        str(ART / "rl_items"))
    items[["group", "type", "advantage", "reward", "task_id"]].to_json(
        ART / "rl_items.jsonl", orient="records", lines=True)
    print(items.groupby("type").agg(n=("advantage", "size"), mean_reward=("reward", "mean")))


def rl_attribute(only: str = "") -> None:
    """Score RL items for the GRPO queries at checkpoint RL_CKPT: TrackStar p32 with the
    advantage restored after unit normalization, sign-only and unweighted variants from
    the same run, and EK-FAC with the advantage applied natively."""
    from reachtrace.attrib import bergson_yaml as by
    from reachtrace.attrib.grpo_query import per_problem, reweight_columns, unit_items_reweight

    model = str(ART / "models" / "rl" / f"checkpoint-{RL_CKPT}")
    _tok().save_pretrained(model)
    root = ART / "bergson_rl"
    items = by.data_cfg(str(ART / "rl_items"), advantage=True)
    query = by.data_cfg(str(ART / "queries"), advantage=True)
    qm = _query_meta()
    im = pd.read_json(ART / "rl_items.jsonl", lines=True)
    tbs = int(os.environ.get("GATE_TOKEN_BATCH", 4096))
    mbs = int(os.environ.get("GATE_EKFAC_BATCH", 16))
    out = ART / "scores_rl"
    out.mkdir(parents=True, exist_ok=True)
    want = set(only.split(",")) if only else None

    if (want is None or "trackstar_p32" in want) and not (out / "ts_weighted.npy").exists():
        rp = str(root / "trackstar_p32")
        y = by.write_pipeline(root / "trackstar_p32.yaml", rp, [by.trackstar_step(
            rp, model, items, query, projection_dim=32, token_batch_size=tbs)])
        sec = _bergson(y)
        raw, _ = _load(root / "trackstar_p32" / "scores")          # items x query rows
        q_w, _ = per_problem(reweight_columns(raw, qm.advantage.to_numpy()), qm.problem_id)
        a = im.advantage.to_numpy()
        np.save(out / "ts_sign_only.npy", q_w)                       # Bergson as stored
        np.save(out / "ts_weighted.npy", unit_items_reweight(q_w, a))  # |A_i| restored
        np.save(out / "ts_unweighted.npy", q_w * np.sign(a)[:, None])  # no advantage at all
        _write_json(out / "trackstar_p32.json", {"seconds": sec})
    if (want is None or "ekfac" in want) and not (out / "ekfac.npy").exists():
        shared = root / "ekfac_factors"
        sec = 0.0
        if not (shared / "hessian" / "kfac").exists():
            y = by.write_pipeline(root / "ekfac_factors.yaml", str(shared), [
                by.kfac_hessian_step(str(shared), model, items, token_batch_size=tbs,
                                     max_batch_size=mbs)])
            sec += _bergson(y)
        # One aggregated query row per problem, at this checkpoint, merged into one index.
        qdir = root / "query_rows"
        ranges = _problem_ranges()
        sec += _bergson(by.write_pipeline(root / "query_rows.yaml", str(qdir), [{"build": {
            "index_cfg": by.index_cfg(str(qdir / f"p{j:02d}"), model, by.data_cfg(
                str(ART / "queries"), split=f"train[{a0}:{b0}]", advantage=True),
                token_batch_size=tbs, max_batch_size=mbs),
            "preprocess_cfg": {"aggregation": "sum", "unit_normalize": False}}}
            for j, (_, a0, b0) in enumerate(ranges)]))
        merged = by.merge_query_indices([qdir / f"p{j:02d}" for j in range(len(ranges))],
                                        qdir / "merged")
        rp = root / "ekfac"
        by.link_hessian(rp, shared)
        sec += _bergson(by.write_pipeline(root / "ekfac.yaml", str(rp), [by.influence_step(
            str(rp), model, items, {"path": str(merged)}, aggregation="none",
            token_batch_size=tbs, max_batch_size=mbs)]))
        np.save(out / "ekfac.npy", _load(rp / "scores")[0])
        _write_json(out / "ekfac.json", {"seconds": sec})


# ------------------------------------------------------------------------------- ledger (G0)


def ledger() -> None:
    """Margin of the query rollouts along the flow: base -> CPT -> GRPO checkpoints.

    M = mean per-token log-prob of correct minus incorrect rollouts (the Olmo 3 ledger's
    margin) and J (the attributed objective), per problem.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from reachtrace.synthetic.cpt import grpo_objective, sequence_logprobs

    rows, adv, prob = _query_rows()
    reward = np.asarray(_query_meta().reward)
    ckpts = [("base", CFG["model"]), ("cpt", str(ART / "models" / "cpt"))]
    ckpts += [(p.name.replace("checkpoint-", "rl_"), str(p))
              for p in sorted((ART / "models" / "rl").glob("checkpoint-*"),
                              key=lambda p: int(p.name.split("-")[1]))]
    out = []
    for name, path in ckpts:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(_device())
        lp, cnt = sequence_logprobs(m, rows, device=_device())
        per_tok = pd.DataFrame({"p": prob, "c": reward > 0, "lp": lp / cnt})
        M = per_tok.groupby(["p", "c"]).lp.mean().unstack()
        J = grpo_objective(lp, adv, prob)
        for p in M.index:
            out.append({"checkpoint": name, "problem": p, "M": M.loc[p, True] - M.loc[p, False],
                        "L_correct": M.loc[p, True], "L_incorrect": M.loc[p, False],
                        "J": J[p]})
        del m
    df = pd.DataFrame(out)
    df.to_csv(ART / "ledger.csv", index=False)
    print(df.groupby("checkpoint", sort=False)[["M", "L_correct", "L_incorrect", "J"]].mean())


# ------------------------------------------------------------------------------- evaluate

LABELS = {
    "bm25": "BM25", "gradcos_p64": "Gradient cosine (p64, no preconditioner)",
    "trackstar_p16": "TrackStar p16", "trackstar_p32": "TrackStar p32",
    "trackstar_p64": "TrackStar p64", "trackstar_p64_mix": "TrackStar p64 (mixed Hessian)",
    "ekfac": "EK-FAC", "ekfac_sub": "EK-FAC (factors on 25% subset)",
    "ekfac_cg": "EK-FAC + ASTRA (CG)", "twostage": "Two-stage: TrackStar p32 -> EK-FAC top 10%",
}


def _label(m: str) -> str:
    if m.startswith("ekfac_astra_lr"):
        return f"EK-FAC + ASTRA (SGD, lr {m.removeprefix('ekfac_astra_lr')})"
    return LABELS.get(m, m)


def _gt(kind: str) -> dict[str, dict]:
    d = ART / "groundtruth" / kind
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.exists() \
        else {}


def _fmt(v: float, lo: float | None = None, hi: float | None = None) -> str:
    return f"{v:.3f}" if lo is None else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def evaluate() -> None:
    from reachtrace.attrib.evaluate import (
        family_recovery,
        lds_summary,
        precision_at_k,
        topk_agreement,
    )
    from reachtrace.synthetic.v3 import dose_weights

    OUT.mkdir(parents=True, exist_ok=True)
    fam = np.asarray(_corpus()["family"])
    n = len(fam)
    problems = sorted(_query_meta().problem_id.unique())
    methods = sorted(p.stem for p in (ART / "scores").glob("*.npy"))
    scores = {m: np.load(ART / "scores" / f"{m}.npy") for m in methods}
    secs = {m: json.loads((ART / "scores" / f"{m}.json").read_text())["seconds"]
            for m in methods}

    def outcomes(kind: str, names: list[str]) -> np.ndarray:
        g = _gt(kind)
        return np.array([[g[s]["J"][p] for p in problems] for s in names])

    dose = _gt("dose")
    design = np.load(ART / "dose_design.npy") if dose else None
    dose_names = sorted(dose)
    W_dose = (np.stack([dose_weights(fam, dict(zip(GROUPS, design[int(s[1:])])),
                                     seed=CFG["seed"]) for s in dose_names])
              if dose else None)
    Y_dose = outcomes("dose", dose_names) if dose else None
    rnd = _gt("random")
    rnd_names = sorted(rnd)
    W_rnd = np.load(ART / "random_masks.npy")[[int(s[1:]) for s in rnd_names]].astype(float) \
        if rnd else None
    Y_rnd = outcomes("random", rnd_names) if rnd else None
    l1 = {k: v for k, v in _gt("leave1").items() if k.startswith("l")}
    l1_names = sorted(l1)
    W_l1 = (np.load(ART / "leave1_masks.npy")[[int(s[1:]) for s in l1_names]].astype(float)
            if l1 else None)
    Y_l1 = outcomes("leave1", l1_names) if l1 else None
    qld_runs = _gt("qld")
    # Family-level ground truth from leave-family-out CPT: dJ_f = J(full) - J(without f).
    lfo_runs = _gt("lfo")
    # Capability ground truth: RL accuracy on long A-then-B lost when a family is removed
    # from CPT (effects of 0 vs ~0.6, far above retraining noise).
    dAcc_fam = {}
    base_acc = lfo_runs.get("minus_none", {}).get("rl_heldout_ab_long")
    if base_acc is not None:
        for f in FAMILIES:
            r_f = lfo_runs.get(f"minus_{f}")
            if r_f and "rl_heldout_ab_long" in r_f:
                dAcc_fam[f] = base_acc - r_f["rl_heldout_ab_long"]
    dJ_fam = {}
    if "minus_none" in lfo_runs and "cpt_J" in lfo_runs["minus_none"]:
        for f in FAMILIES:
            r_f = lfo_runs.get(f"minus_{f}")
            if r_f and "cpt_J" in r_f:
                dJ_fam[f] = lfo_runs["minus_none"]["cpt_J"] - r_f["cpt_J"]
    noise = _gt("noise")
    full_J = None
    if noise:
        full_J = np.array([noise["full_a"]["J"][p] for p in problems]).sum()
    k1 = max(1, round(0.01 * n))

    rows = []
    for m in methods:
        s = scores[m]
        r = {"method": _label(m), "key": m, "seconds": secs[m]}
        finite = np.isfinite(s).all()
        if finite and W_dose is not None:
            d = lds_summary(s, W_dose, Y_dose)
            r.update({"LDS (dose)": _fmt(d["lds"], d["lds_lo"], d["lds_hi"]),
                      "lds_dose": d["lds"]})
        if finite and W_l1 is not None:
            d = lds_summary(s, W_l1, Y_l1)
            r.update({"LDS (leave-1%)": _fmt(d["lds"], d["lds_lo"], d["lds_hi"]),
                      "lds_leave1": d["lds"]})
        if finite and W_rnd is not None:
            d = lds_summary(s, W_rnd, Y_rnd)
            r.update({"LDS (random half)": _fmt(d["lds"], d["lds_lo"], d["lds_hi"]),
                      "lds_random": d["lds"]})
        s_fill = np.where(np.isfinite(s), s, np.nanmin(s) - 1.0)
        rec = family_recovery(s_fill, fam)
        r.update({k: v for k, v in rec.items()})
        agg = s_fill.sum(1)
        r["P@1% causal"] = precision_at_k(agg, np.isin(fam, ["A", "B"]), k1)
        if len(dAcc_fam) >= 3:
            from scipy.stats import spearmanr

            fs = sorted(dAcc_fam)
            fam_mean = [np.nanmean(np.where(np.isfinite(s), s, np.nan)[fam == f].sum(1))
                        for f in fs]
            r["capability rank corr"] = float(spearmanr(fam_mean, [dAcc_fam[f] for f in fs])
                                              .statistic)
        if finite and len(dJ_fam) >= 3:
            from scipy.stats import spearmanr

            fs = sorted(dJ_fam)
            mass = [agg[fam == f].sum() for f in fs]
            truth = [dJ_fam[f] for f in fs]
            r["family rank corr (leave-family-out)"] = float(spearmanr(mass, truth).statistic)
            r["family sign agreement"] = float(np.mean(np.sign(mass) == np.sign(truth)))
        q = qld_runs.get(f"top0.01_{m}_0")
        rq = [v for k, v in qld_runs.items() if k.startswith("top0.01_random_")]
        if q and rq and full_J is not None:
            jt = sum(q["J"].values())
            jr = np.mean([sum(v["J"].values()) for v in rq])
            r["QLD top 1%"] = full_J - jt
            r["QLD - random"] = (full_J - jt) - (full_J - jr)
        rows.append(r)
    res = pd.DataFrame(rows)
    order = [c for c in ["lds_leave1", "lds_random", "lds_dose", "auc_causal"] if c in res]
    res = res.sort_values(order[0] if order else "method", ascending=False)
    res.to_csv(OUT / "methods.csv", index=False)

    # Two-stage retrieval against full EK-FAC: does the cheap first stage keep the top 1%?
    tw = []
    if "ekfac" in scores:
        for m in ["trackstar_p16", "trackstar_p32", "trackstar_p64", "twostage", "ekfac_sub",
                  "bm25", "gradcos_p64"]:
            if m not in scores:
                continue
            s = np.where(np.isfinite(scores[m]), scores[m], -np.inf)
            agree = [topk_agreement(s[:, j], scores["ekfac"][:, j], k1)
                     for j in range(len(problems))]
            tw.append({"method": _label(m), "top-1% overlap with full EK-FAC":
                       float(np.mean(agree)), "seconds": secs[m]})
    pd.DataFrame(tw).to_csv(OUT / "twostage.csv", index=False)

    # Retrain noise: identical full runs should give identical J.
    meta = {"n_docs": n, "n_problems": len(problems), "n_dose": len(dose_names),
            "n_leave1": len(l1_names),
            "n_random": len(rnd_names), "top1pct_docs": k1}
    if noise and len(noise) == 2:
        a, b = (np.array([noise[x]["J"][p] for p in problems]) for x in ("full_a", "full_b"))
        meta["retrain_noise_max_abs_J"] = float(np.abs(a - b).max())
        if Y_dose is not None:
            meta["dose_outcome_sd_mean"] = float(Y_dose.std(0).mean())
    _write_json(OUT / "meta.json", meta)
    for f in ["skill.csv", "ledger.csv"]:
        if (ART / f).exists():
            pd.read_csv(ART / f).to_csv(OUT / f, index=False)
    lfo = _gt("lfo")
    if lfo:
        pd.DataFrame(list(lfo.values())).to_csv(OUT / "leave_family_out.csv", index=False)
    if dJ_fam:
        _write_json(OUT / "leave_family_out_dJ.json", dJ_fam)
    if dAcc_fam:
        _write_json(OUT / "leave_family_out_capability.json",
                    {"rl_ab_long_full": base_acc, "drop_when_removed": dAcc_fam})
    show = [c for c in ["method", "LDS (leave-1%)", "LDS (random half)", "LDS (dose)",
                        "capability rank corr",
                        "family rank corr (leave-family-out)", "family sign agreement",
                        "auc_causal", "z_A", "z_B",
                        "z_M", "z_F", "z_W", "P@1% causal", "QLD - random", "seconds"]
            if c in res]
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(tw).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(meta)
    evaluate_plain(problems, {"leave1": (l1, l1_names, W_l1), "random": (rnd, rnd_names, W_rnd)})
    evaluate_rl()
    figures(res)


def evaluate_plain(problems: list[str], designs: dict) -> None:
    """Diagnostic: the plain objective (log-likelihood of the correct rollouts) attributed with
    plain queries, against the same retrains, from the per-row log-probs they stored."""
    from reachtrace.attrib.evaluate import lds_summary

    d = ART / "scores_plain"
    q = pd.read_json(ART / "queries.jsonl", lines=True)
    correct, prob = q.reward.to_numpy() > 0, q.problem_id.to_numpy()
    rows = []
    for m in sorted(p.stem for p in d.glob("*.npy")) if d.exists() else []:
        s = np.load(d / f"{m}.npy")
        r = {"method": _label(m), "key": m}
        for design, (gt, names, W) in designs.items():
            if W is None or not all("lp" in gt[x] for x in names):
                continue
            Y = np.array([[np.asarray(gt[x]["lp"])[(prob == p) & correct].sum() for p in problems]
                          for x in names])
            out = lds_summary(s, W, Y)
            r[f"plain LDS ({design})"] = _fmt(out["lds"], out["lds_lo"], out["lds_hi"])
        rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "methods_plain.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))


def evaluate_rl() -> None:
    """RL items: score mass per item type under each scoring, against leave-type-out GRPO."""
    d = ART / "scores_rl"
    if not d.exists():
        return
    im = pd.read_json(ART / "rl_items.jsonl", lines=True)
    lfo = _gt("lfo")
    ref = lfo.get("minus_none", {}).get("rl_J")
    effect = {t: ref - lfo[f"rl_minus_{t}"]["rl_J"] for t in ("AB", "A", "B")
              if ref is not None and f"rl_minus_{t}" in lfo and "rl_J" in lfo[f"rl_minus_{t}"]}
    rows = []
    for name in ["ts_weighted", "ts_sign_only", "ts_unweighted", "ekfac"]:
        if not (d / f"{name}.npy").exists():
            continue
        s = np.load(d / f"{name}.npy").sum(1)          # summed over query problems
        s = s / np.abs(s).sum()                         # share of total absolute mass
        g = pd.Series(s).groupby(im.type.to_numpy())
        r = {"scoring": name}
        for t, v in g.sum().items():
            r[f"mass_{t}"] = v
        for t, v in g.mean().items():
            r[f"per_item_{t}"] = v
        trained = [t for t in ("AB", "A", "B") if f"mass_{t}" in r]
        if len(effect) >= 2:
            from scipy.stats import spearmanr

            ts = [t for t in trained if t in effect]
            r["rank_agreement_with_removal"] = float(spearmanr(
                [r[f"mass_{t}"] for t in ts], [effect[t] for t in ts]).statistic)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "rl_items.csv", index=False)
    _write_json(OUT / "rl_items_groundtruth.json",
                {"J_full_200_steps": ref, "drop_in_J_when_type_removed": effect,
                 "items_per_type": im.type.value_counts().to_dict()})
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("drop in J when an RL prompt type is removed:", effect)


def figures(res: pd.DataFrame) -> None:
    """Static figures for the results README (palette: dataviz reference instance)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, blue, orange = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})

    key, col, what = next(((k, c, w) for k, c, w in [
        ("lds_leave1", "LDS (leave-1%)", "leave-1%-out retrains"),
        ("lds_random", "LDS (random half)", "random-half retrains"),
        ("lds_dose", "LDS (dose)", "dose design")] if k in res), (None, None, None))
    if key:
        r = res.dropna(subset=[key]).sort_values(key)
        lo = r[col].str.extract(r"\[(-?[\d.]+), (-?[\d.]+)\]").astype(float)
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(r) + 1.0))
        y = np.arange(len(r))
        ax.hlines(y, lo[0], lo[1], color=blue, lw=2)
        ax.plot(r[key], y, "o", color=blue, ms=6, mec="white", mew=1.5)
        ax.axvline(0, color=muted, lw=0.8)
        ax.set_yticks(y, r.method)
        ax.set_xlabel(f"LDS, {what}\n(mean over query problems; 95% CI over subsets)")
        ax.grid(axis="x", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(OUT / "lds.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fams = ["A", "B", "C", "D", "M", "F", "W", "G"]
    cols = [f"z_{f}" for f in fams if f"z_{f}" in res]
    if cols:
        # Two-stage scores exist only for candidates, so their family means are not comparable.
        r = res[res.key != "twostage"].sort_values("method")
        z = r[cols].to_numpy(float)
        lim = min(0.5, np.nanmax(np.abs(z)))  # clipped so small differences stay visible
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("div", [orange, "#f3f2ef", blue])
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(r) + 1.2))
        ax.imshow(z, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        for (i, j), v in np.ndenumerate(z):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color=ink)
        ax.set_xticks(range(len(cols)), [c[2:] for c in cols])
        ax.set_yticks(range(len(r)), r.method)
        ax.set_xlabel("family (A, B taught operations; C, D look-alikes; M name decoys;\n"
                      "F format only; W worked traces; G web). Cell: mean z-score of the "
                      "family's documents")
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "families.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    if (ART / "ledger.csv").exists():
        led = pd.read_csv(ART / "ledger.csv")
        m = led.groupby("checkpoint", sort=False).M.agg(["mean", "sem"])
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        x = np.arange(len(m))
        ax.errorbar(x, m["mean"], yerr=1.96 * m["sem"], color=blue, lw=2, marker="o", ms=6,
                    mec="white", capsize=0)
        ax.axhline(0, color=muted, lw=0.8)
        ax.set_xticks(x, m.index)
        ax.set_ylabel("margin M (nats/token)")
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(OUT / "ledger.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def plain_queries() -> None:
    """Correct rollouts only, advantage +1: the plain log-likelihood objective as a query."""
    from datasets import load_from_disk

    q = load_from_disk(str(ART / "queries"))
    keep = [i for i, r in enumerate(q["reward"]) if r > 0]
    q.select(keep).map(lambda r: {"advantage": 1.0}).save_to_disk(str(ART / "queries_plain"))
    m = pd.read_json(ART / "queries.jsonl", lines=True).iloc[keep].assign(advantage=1.0)
    m.to_json(ART / "queries_plain.jsonl", orient="records", lines=True)
    print(f"{len(keep)} correct rollouts over {m.problem_id.nunique()} problems")


STAGES = {
    "prepare": prepare, "tasks": make_tasks, "cpt": cpt, "skill": skill, "grpo": grpo,
    "queries": queries, "attribute": attribute, "groundtruth": groundtruth, "ledger": ledger,
    "rl_items": rl_items, "rl_attribute": rl_attribute, "evaluate": evaluate,
    "plain_queries": plain_queries,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: gate_v3.py {{{','.join(STAGES)}}}")
    STAGES[sys.argv[1]](*sys.argv[2:])
