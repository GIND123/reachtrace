"""Exploratory pilot of the v3 instruments on Olmo 3 within a ~$15 Modal budget.

This is a pilot, not the pre-registered study: small n, a 7,680-token cap (so every trace
fits the 8,192-token context of stage-1/2 checkpoints), and relaxed G/B definitions.

    .venv/bin/python scripts/pilot.py select      # problems + Dolci 13-gram decon (local)
    .venv/bin/python scripts/pilot.py download    # weights into the Modal volume (CPU)
    .venv/bin/python scripts/pilot.py sample      # base and RL, n=16 each (2 x H100)
    .venv/bin/python scripts/pilot.py escalate    # base +48 on near-zero problems
    .venv/bin/python scripts/pilot.py ledger      # teacher-forced scores, 7 checkpoints
    .venv/bin/python scripts/pilot.py handoff     # base continuations from RL prefixes
    .venv/bin/python scripts/pilot.py report      # tables + figures -> results/pilot/

Every Modal call's cost is appended to artifacts/pilot/costs.jsonl; the script refuses to
start a GPU step once the running total passes BUDGET_USD.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "pilot"
BUDGET_USD = 20.0
BASE = "allenai/Olmo-3-1025-7B"
RL = "allenai/Olmo-3.1-7B-RL-Zero-Math"
MAX_TOKENS = 7680
# Shared-13-gram fraction above which a problem counts as a Dolci near-duplicate. Any single
# shared 13-gram is too strict: competition boilerplate ("where m and n are relatively prime
# positive integers. Find m+n") flags 26 of 101 problems, including 2026 contests.
DECON_FRAC = 0.4
SEED = 20261026
PHASE_IDS = {"base1": 1, "rl1": 2, "base2": 3, "handoff": 4}
N_FIRST = 16
N_ESCALATE = 48

# Ledger checkpoints, in flow order: (name, repo, branch).
LEDGER_CKPTS = [
    ("s1_mid", BASE, "stage1-step707000"),
    ("s1_end", BASE, "stage1-step1413814"),
    ("s2_end", BASE, "stage2-step47684"),
    ("s3_end", BASE, "main"),
    ("rl_0200", RL, "step_0200"),
    ("rl_0800", RL, "step_0800"),
    ("rl_end", RL, "main"),
]
STAGES = [("pretrain_late", "s1_end"), ("midtrain", "s2_end"), ("longctx", "s3_end"),
          ("rl_early", "rl_0200"), ("rl_mid", "rl_0800"), ("rl_late", "rl_end")]


# ---------------------------------------------------------------------- bookkeeping


def _costs() -> float:
    f = ART / "costs.jsonl"
    if not f.exists():
        return 0.0
    return sum(json.loads(line).get("usd", 0.0) for line in f.read_text().splitlines() if line)


def _log_cost(step: str, stats: dict) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    rec = {"step": step, **{k: v for k, v in stats.items() if k != "records"}}
    with open(ART / "costs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{step}] ${stats.get('usd', 0):.2f}; running total ${_costs():.2f}")


def _guard() -> None:
    spent = _costs()
    if spent > BUDGET_USD:
        sys.exit(f"budget guard: ${spent:.2f} spent > ${BUDGET_USD}")


def _fetch(vol_path: str, local: Path) -> Path:
    import modal

    vol = modal.Volume.from_name("reachtrace-vol")
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local, "wb") as f:
        for chunk in vol.read_file(f"artifacts/{vol_path}"):
            f.write(chunk)
    return local


def _sha(repo: str, branch: str) -> str:
    from reachtrace.registry import resolve_sha

    return resolve_sha(repo, branch)


def _grade(df: pd.DataFrame, answers: pd.Series) -> pd.DataFrame:
    from reachtrace.sample.grade import grade, grade_lenient

    gold = df.key.map(answers)
    g = [grade(t, a) for t, a in zip(df.text, gold)]
    return df.assign(
        correct=[x.correct for x in g],
        extracted=[x.extracted for x in g],
        lenient=[grade_lenient(t, a) for t, a in zip(df.text, gold)],
    )


# ---------------------------------------------------------------------- steps


def select() -> None:
    from datasets import load_dataset

    from reachtrace.data.dolci import DOLCI_MATH_ID
    from reachtrace.data.evals import load_eval_set
    from reachtrace.decon.ngram import contamination_report

    parts = [load_eval_set("aime26"), load_eval_set("hmmt_feb26")]
    omni = load_eval_set("omni_math")
    omni = omni[omni.level.astype(float).between(4.0, 6.5)]
    parts.append(omni.sample(60, random_state=0))
    m5 = load_eval_set("math500")
    parts.append(m5[m5.level == "5"].sample(15, random_state=0))
    probs = pd.concat(parts, ignore_index=True).drop_duplicates("problem_hash")
    dolci = load_dataset(DOLCI_MATH_ID, split="train").to_pandas()
    refs = {"dolci": pd.DataFrame({"ref_id": range(len(dolci)), "text": dolci.prompt.astype(str)})}
    rep = contamination_report(probs, refs, n=13, flag_frac=DECON_FRAC)
    probs = probs.merge(rep[["problem_id", "dolci_frac", "dolci_flag"]], on="problem_id")
    n_dup = int(probs.dolci_flag.sum())
    print(f"dolci near-duplicates (13-gram frac > {DECON_FRAC}): {n_dup} dropped")
    print(probs[probs.dolci_flag].groupby("eval_set").size().to_string())
    probs = probs[~probs.dolci_flag].reset_index(drop=True)
    ART.mkdir(parents=True, exist_ok=True)
    probs.to_parquet(ART / "problems.parquet")
    print(probs.groupby("eval_set").size().to_string(), f"\ntotal {len(probs)}")


def download() -> None:
    import modal

    from reachtrace.remote.app import app
    from reachtrace.remote.app import download as dl

    revs = sorted({(r, _sha(r, b)) for _, r, b in LEDGER_CKPTS})
    with modal.enable_output(), app.run():
        for res in dl.starmap(revs):
            print(res["repo"], res["revision"], f"{res['gb']:.1f} GB", f"{res['seconds']:.0f}s")
            _log_cost("download", {"usd": res["seconds"] * 4 * 0.0000131})


def _requests(probs: pd.DataFrame, n: int, phase: str) -> list[dict]:
    from reachtrace.cache import sample_seed
    from reachtrace.prompts import get_template

    t = get_template("rlzero31_math")
    return [
        {"key": pid, "prompt": t.render(q), "n": n,
         "seed": sample_seed(SEED, pid, PHASE_IDS[phase])}
        for pid, q in zip(probs.problem_id, probs.question)
    ]


def _run_generation(jobs: list[tuple[str, str, list[dict], str]], step: str) -> None:
    """jobs: (repo, branch, requests, out_name); runs in parallel containers."""
    import modal

    from reachtrace.remote.app import app, generate

    _guard()
    args = [(repo, _sha(repo, br), reqs, f"pilot/{name}.parquet") for repo, br, reqs, name in jobs]
    with modal.enable_output(), app.run():
        for (*_, name), res in zip(
            jobs, generate.starmap(args, kwargs={"max_tokens": MAX_TOKENS})
        ):
            _log_cost(f"{step}:{name}", res)
            print(json.dumps({k: res[k] for k in ("samples", "gen_tokens", "tok_per_s")}))
            _fetch(f"pilot/{name}.parquet", ART / f"{name}.parquet")


def sample() -> None:
    probs = pd.read_parquet(ART / "problems.parquet")
    _run_generation(
        [
            (BASE, "main", _requests(probs, N_FIRST, "base1"), "gen_base_1"),
            (RL, "main", _requests(probs, N_FIRST, "rl1"), "gen_rl_1"),
        ],
        "sample",
    )


def _load_gens(side: str) -> pd.DataFrame:
    probs = pd.read_parquet(ART / "problems.parquet").set_index("problem_id")
    files = sorted(ART.glob(f"gen_{side}_*.parquet"))
    parts = []
    for i, f in enumerate(files):
        d = pd.read_parquet(f)
        d["sample_idx"] = d.sample_idx + (0 if i == 0 else N_FIRST + (i - 1) * N_ESCALATE)
        parts.append(d)
    return _grade(pd.concat(parts, ignore_index=True), probs.answer)


def _table() -> pd.DataFrame:
    b, r = _load_gens("base"), _load_gens("rl")
    t = b.groupby("key").agg(n_base=("correct", "size"), c_base=("correct", "sum"),
                             c_base_lenient=("lenient", "sum"),
                             base_len=("n_tokens", "mean"),
                             base_cap=("finish_reason", lambda s: (s == "length").mean()))
    first = b[b.sample_idx < N_FIRST].groupby("key").correct.agg(n_base_first="size",
                                                                 c_base_first="sum")
    rr = r.groupby("key").agg(n_rl=("correct", "size"), c_rl=("correct", "sum"),
                              rl_len=("n_tokens", "mean"),
                              rl_cap=("finish_reason", lambda s: (s == "length").mean()))
    return t.join(first).join(rr).rename_axis("problem_id").reset_index()


def escalate() -> None:
    t = _table()
    near = t[(t.c_base <= 2) & (t.c_rl >= 2)]
    print(f"escalating base on {len(near)} problems (base <= 2/16, RL >= 2/16)")
    probs = pd.read_parquet(ART / "problems.parquet")
    probs = probs[probs.problem_id.isin(near.problem_id)]
    _run_generation([(BASE, "main", _requests(probs, N_ESCALATE, "base2"), "gen_base_2")],
                    "escalate")


def groups() -> pd.DataFrame:
    """Pilot groups. G: base 0/64 strict and lenient, RL >= 2/16. B: base rate in (0, 4/64].
    E (easy reference): base rate >= 1/4 and RL >= 1/2."""
    t = _table()
    rate = t.c_base / t.n_base
    rl = t.c_rl / t.n_rl
    g = (t.c_base == 0) & (t.c_base_lenient == 0) & (t.n_base >= 64) & (rl >= 2 / 16)
    b = (t.c_base > 0) & (rate <= 4 / 64) & (t.n_base >= 64) & (rl >= 2 / 16)
    e = (rate >= 0.25) & (rl >= 0.5)
    t["group"] = np.select([g, b, e], ["G", "B", "E"], "other")
    probs = pd.read_parquet(ART / "problems.parquet")[["problem_id", "eval_set", "level"]]
    return t.merge(probs, on="problem_id")


def _pick_traces(r: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Up to k length-matched (correct, incorrect) pairs of finished RL traces per problem."""
    fin = r[(r.finish_reason == "stop") & r.extracted.notna()]
    rows = []
    for pid, d in fin.groupby("key"):
        cor = d[d.correct].sort_values("n_tokens")
        inc = d[~d.correct].sort_values("n_tokens")
        used: set[int] = set()
        for _, c in cor.iterrows():
            if len(rows) and sum(x["key"] == pid for x in rows) >= 2 * k:
                break
            cand = inc[~inc.index.isin(used)]
            if cand.empty:
                break
            j = (cand.n_tokens / c.n_tokens).sub(1).abs().idxmin()
            if not 2 / 3 <= inc.loc[j, "n_tokens"] / c.n_tokens <= 1.5:
                continue
            used.add(j)
            rows += [c.to_dict(), inc.loc[j].to_dict()]
    return pd.DataFrame(rows)


def ledger() -> None:
    import modal

    from reachtrace.prompts import get_template
    from reachtrace.remote.app import app, ledger_scores

    _guard()
    grp = groups()
    keep = grp[grp.group.isin(["G", "B", "E"])].problem_id
    r = _load_gens("rl")
    traces = _pick_traces(r[r.key.isin(keep)])
    traces["trace_id"] = traces.key + "#" + traces.sample_idx.astype(str)
    traces.to_parquet(ART / "ledger_traces.parquet")
    print(traces.groupby("key").size().describe().to_string())
    probs = pd.read_parquet(ART / "problems.parquet").set_index("problem_id")
    t = get_template("rlzero31_math")
    items = [
        {"trace_id": tid, "prompt": t.render(probs.loc[k, "question"]),
         "completion_ids": list(map(int, ids))}
        for tid, k, ids in zip(traces.trace_id, traces.key, traces.token_ids)
    ]
    args = [(repo, _sha(repo, br), name, items) for name, repo, br in LEDGER_CKPTS]
    recs = []
    with modal.enable_output(), app.run():
        for (name, _, _), res in zip(LEDGER_CKPTS, ledger_scores.starmap(args)):
            _log_cost(f"ledger:{name}", res)
            recs += res["records"]
    pd.DataFrame(recs).to_parquet(ART / "ledger_scores.parquet")


def handoff() -> None:
    from reachtrace.cache import sample_seed
    from reachtrace.handoff.core import prefix_lengths
    from reachtrace.prompts import get_template

    grp = groups()
    keep = set(grp[grp.group.isin(["G", "B"])].problem_id)
    tr = pd.read_parquet(ART / "ledger_traces.parquet")
    sc = pd.read_parquet(ART / "ledger_scores.parquet")
    a_start = sc[sc.checkpoint == "rl_end"].set_index("trace_id").answer_start
    cor = tr[tr.correct & tr.key.isin(keep)].groupby("key").head(1)
    probs = pd.read_parquet(ART / "problems.parquet").set_index("problem_id")
    t = get_template("rlzero31_math")
    reqs, meta = [], []
    for _, row in cor.iterrows():
        a = int(a_start[row.trace_id])
        for f, L in prefix_lengths(a, (0.1, 0.25, 0.5, 0.75)).items():
            key = f"{row.key}|{f}"
            reqs.append({"key": key, "prompt": t.render(probs.loc[row.key, "question"]),
                         "n": 8, "seed": sample_seed(SEED, key, PHASE_IDS["handoff"]),
                         "prefix_ids": list(map(int, row.token_ids[:L]))})
            meta.append({"key": key, "problem_id": row.key, "frac": f, "prefix_len": L,
                         "trace_id": row.trace_id})
    pd.DataFrame(meta).to_parquet(ART / "handoff_meta.parquet")
    print(f"{len(reqs)} prefix requests x 8 samples")
    _run_generation([(BASE, "main", reqs, "handoff_base")], "handoff")


if __name__ == "__main__":
    step = sys.argv[1]
    {"select": select, "download": download, "sample": sample, "escalate": escalate,
     "ledger": ledger, "handoff": handoff}[step]()
