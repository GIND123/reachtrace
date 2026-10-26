"""Exploratory pilot of the v3 instruments on Olmo 3 within a $25 Modal budget.

This is a pilot, not the pre-registered study: small n, a 7,680-token cap (so every trace
fits the 8,192-token context of stage-1/2 checkpoints), and relaxed G/B definitions.

    .venv/bin/python scripts/pilot.py select      # problems + Dolci 13-gram decon (local)
    .venv/bin/python scripts/pilot.py download    # weights into the Modal volume (CPU)
    .venv/bin/python scripts/pilot.py sample      # base and RL, n=16 each (2 x H100)
    .venv/bin/python scripts/pilot.py escalate    # base +48 on near-zero problems (not run)
    .venv/bin/python scripts/pilot.py ledger      # matched-length pairs, 5 checkpoints
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
BUDGET_USD = 22.0
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
# Five checkpoints to fit the budget; stage1-step707000 and step_0800 are downloaded but unused.
LEDGER_CKPTS = [
    ("s1_end", BASE, "stage1-step1413814"),
    ("s2_end", BASE, "stage2-step47684"),
    ("s3_end", BASE, "main"),
    ("rl_0200", RL, "step_0200"),
    ("rl_end", RL, "main"),
]
STAGES = [("midtrain", "s2_end"), ("longctx", "s3_end"), ("rl_early", "rl_0200"),
          ("rl_late", "rl_end")]
HANDOFF_MAX_TOKENS = 4096


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


def _run_generation(
    jobs: list[tuple[str, str, list[dict], str]], step: str, max_tokens: int = MAX_TOKENS
) -> None:
    """jobs: (repo, branch, requests, out_name); runs in parallel containers."""
    import modal

    from reachtrace.remote.app import app, generate

    _guard()
    args = [(repo, _sha(repo, br), reqs, f"pilot/{name}.parquet") for repo, br, reqs, name in jobs]
    with modal.enable_output(), app.run():
        for (*_, name), res in zip(
            jobs, generate.starmap(args, kwargs={"max_tokens": max_tokens})
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
    """Pilot groups at n = 16 per model (no escalation: the budget did not allow it).

    G: base 0/16 under both strict and lenient grading, RL >= 2/16.
    B: base 1-2/16, RL >= 2/16.
    E (easy reference): base >= 4/16, RL >= 8/16.
    G here is "unsolved in 16 base samples", a much weaker statement than the plan's 0/256.
    """
    t = _table()
    g = (t.c_base == 0) & (t.c_base_lenient == 0) & (t.c_rl >= 2)
    b = t.c_base.between(1, 2) & (t.c_rl >= 2)
    e = (t.c_base >= 4) & (t.c_rl >= 8)
    t["group"] = np.select([g, b, e], ["G", "B", "E"], "other")
    probs = pd.read_parquet(ART / "problems.parquet")[["problem_id", "eval_set", "level"]]
    return t.merge(probs, on="problem_id")


def _tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(BASE)


def _answer_start(tok, ids) -> int:
    from reachtrace.ledger.core import answer_start_token

    return answer_start_token(list(map(int, ids)), lambda x: tok.decode(list(x)))


def _pick_pairs(r: pd.DataFrame, min_len: int = 256, seed: int = 0) -> pd.DataFrame:
    """One matched-length (correct, incorrect) RL trace pair per problem.

    The correct trace is a finished one with the median length; the incorrect trace is a
    random incorrect sample, finished or truncated (73% of RL samples hit the 7,680-token
    cap, so finished incorrect traces are rare). Both are cut to the same window
    [0, L) with L = min(answer start of the correct trace, length of the incorrect trace),
    so the content margin compares the same number of positions and never includes an
    answer line.
    """
    tok = _tokenizer()
    rng = np.random.default_rng(seed)
    rows = []
    for pid, d in r.groupby("key"):
        cor = d[d.correct & (d.finish_reason == "stop")].sort_values("n_tokens")
        inc = d[~d.correct]
        if cor.empty or inc.empty:
            continue
        c = cor.iloc[len(cor) // 2]
        i = inc.iloc[int(rng.integers(len(inc)))]
        a_c = _answer_start(tok, c.token_ids)
        a_i = _answer_start(tok, i.token_ids)
        win = int(min(a_c, a_i, len(i.token_ids)))
        if win < min_len:
            continue
        for row, lab in ((c, True), (i, False)):
            rows.append({"key": pid, "trace_id": f"{pid}#{row.sample_idx}", "correct": lab,
                         "window": win, "full_len": int(row.n_tokens),
                         "finished": row.finish_reason == "stop",
                         "token_ids": list(map(int, row.token_ids[:win]))})
    return pd.DataFrame(rows)


def ledger() -> None:
    import modal

    from reachtrace.prompts import get_template
    from reachtrace.remote.app import app, ledger_scores

    _guard()
    traces = _pick_pairs(_load_gens("rl"))
    traces.to_parquet(ART / "ledger_traces.parquet")
    n_prob = traces.key.nunique()
    print(f"{n_prob} problems, window tokens: {traces.groupby('key').window.first().describe()}")
    probs = pd.read_parquet(ART / "problems.parquet").set_index("problem_id")
    t = get_template("rlzero31_math")
    items = [
        {"trace_id": tid, "prompt": t.render(probs.loc[k, "question"]),
         "completion_ids": ids, "answer_start": len(ids)}
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
    r = _load_gens("rl")
    cor = r[r.correct & (r.finish_reason == "stop") & r.key.isin(keep)]
    cor = cor.sort_values("n_tokens").groupby("key").nth(0).reset_index(drop=True)
    tok = _tokenizer()
    probs = pd.read_parquet(ART / "problems.parquet").set_index("problem_id")
    t = get_template("rlzero31_math")
    reqs, meta = [], []
    for _, row in cor.iterrows():
        a = _answer_start(tok, row.token_ids)
        for f, L in prefix_lengths(a, (0.25, 0.5, 0.75)).items():
            key = f"{row.key}|{f}"
            reqs.append({"key": key, "prompt": t.render(probs.loc[row.key, "question"]),
                         "n": 8, "seed": sample_seed(SEED, key, PHASE_IDS["handoff"]),
                         "prefix_ids": list(map(int, row.token_ids[:L]))})
            meta.append({"key": key, "problem_id": row.key, "frac": f, "prefix_len": L,
                         "content_len": a})
    pd.DataFrame(meta).to_parquet(ART / "handoff_meta.parquet")
    print(f"{cor.key.nunique()} problems, {len(reqs)} prefix requests x 8 samples")
    _run_generation([(BASE, "main", reqs, "handoff_base")], "handoff",
                    max_tokens=HANDOFF_MAX_TOKENS)


def _boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def _fmt(m: tuple[float, float, float], d: int = 3) -> str:
    return f"{m[0]:.{d}f} [{m[1]:.{d}f}, {m[2]:.{d}f}]"


def report() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reachtrace.handoff.core import composition_signature, success_curve
    from reachtrace.handoff.core import handoff as h_of
    from reachtrace.ledger.core import margins, stage_increments
    from reachtrace.reach.passk import pass_at_k

    out = ROOT / "results" / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Pilot results (exploratory)", ""]
    costs = [json.loads(x) for x in (ART / "costs.jsonl").read_text().splitlines() if x]
    md += [f"Modal spend for this pilot: **${sum(c.get('usd', 0) for c in costs):.2f}** "
           "(GPU time from job wall-clock at $3.95/H100-hour; CPU downloads included).", ""]

    # ---- RQ1: accuracy, format, groups
    b, r = _load_gens("base"), _load_gens("rl")
    grp = groups()
    gmap = grp.set_index("problem_id").group
    rows = []
    for es, d in grp.groupby("eval_set"):
        ids = set(d.problem_id)
        bb, rr = b[b.key.isin(ids) & (b.sample_idx < N_FIRST)], r[r.key.isin(ids)]

        def pk(df, k):
            c = df.groupby("key").correct.agg(["size", "sum"])
            return np.mean([pass_at_k(int(n), int(s), k) for n, s in zip(c["size"], c["sum"])])

        rows.append({
            "set": es, "problems": len(ids),
            "base pass@1": pk(bb, 1), "base pass@16": pk(bb, 16),
            "RL pass@1": pk(rr, 1), "RL pass@16": pk(rr, 16),
            "base no-answer": bb.extracted.isna().mean(),
            "RL no-answer": rr.extracted.isna().mean(),
            "base lenient-strict gap": (bb.lenient & ~bb.correct).mean(),
            "base at cap": (bb.finish_reason == "length").mean(),
            "RL at cap": (rr.finish_reason == "length").mean(),
            "G": int((d.group == "G").sum()), "B": int((d.group == "B").sum()),
            "E": int((d.group == "E").sum()),
        })
    acc = pd.DataFrame(rows)
    acc.to_csv(out / "accuracy_by_set.csv", index=False)
    md += ["## 1. Accuracy, format and groups (first 16 samples per model; 7,680-token cap)", "",
           acc.to_markdown(index=False, floatfmt=".3f"), ""]
    grp.to_csv(out / "problems_groups.csv", index=False)

    # ---- RQ2: ledger
    if (ART / "ledger_scores.parquet").exists():
        sc = pd.read_parquet(ART / "ledger_scores.parquet")
        tr = pd.read_parquet(ART / "ledger_traces.parquet")[["trace_id", "key", "correct"]]
        sc = sc.merge(tr, on="trace_id").rename(columns={"key": "problem_id"})
        order = [n for n, _, _ in LEDGER_CKPTS]
        m = margins(sc, "content")
        m["group"] = m.problem_id.map(gmap)
        lrows = []
        for g_ in ("G", "B", "E"):
            for ck in order:
                x = m[(m.group == g_) & (m.checkpoint == ck)]
                lrows.append({"group": g_, "checkpoint": ck, "problems": len(x),
                              "M (content margin)": _fmt(_boot_ci(x.M)),
                              "L correct": _fmt(_boot_ci(x.L_correct), 2),
                              "L incorrect": _fmt(_boot_ci(x.L_incorrect), 2)})
        led = pd.DataFrame(lrows)
        led.to_csv(out / "ledger_margins.csv", index=False)
        md += ["## 2. Discrimination margin across the flow", "",
               "M = mean per-token log-likelihood of correct minus incorrect RL traces on content "
               "tokens (opening 16 tokens and answer line excluded); mean [95% bootstrap CI] "
               "over problems.", "", led.to_markdown(index=False), ""]
        wide = m.pivot(index="problem_id", columns="checkpoint", values="M")
        inc = stage_increments(wide.dropna(), STAGES, start="s1_end")
        inc["group"] = inc.index.map(gmap)
        stg = [s for s, _ in STAGES]
        summ = inc.groupby("group")[stg + ["total"]].mean()
        summ.to_csv(out / "ledger_stage_increments.csv")
        md += ["Mean stage increments of M (they sum exactly to `total`, the change from "
               "the end of pretraining to the final RL model):", "",
               summ.to_markdown(floatfmt=".3f"), ""]
        op = sc.groupby(["checkpoint", "correct"]).opening.mean().unstack()
        md += ["Opening-token log-likelihood (first 16 tokens, all traces):", "",
               op.reindex(order).to_markdown(floatfmt=".2f"), ""]
        fig, ax = plt.subplots(figsize=(7, 4))
        for g_, c in zip(("G", "B", "E"), ("C3", "C1", "C0")):
            ys = [_boot_ci(m[(m.group == g_) & (m.checkpoint == ck)].M) for ck in order]
            ax.errorbar(range(len(order)), [y[0] for y in ys],
                        yerr=[[y[0] - y[1] for y in ys], [y[2] - y[0] for y in ys]],
                        label=f"{g_}", color=c, marker="o", capsize=3)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(len(order)), order, rotation=30)
        ax.set_ylabel("content margin M (nats/token)")
        ax.set_title("Preference for correct over incorrect RL traces")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "ledger_margin.png", dpi=150)
        md += ["![margin](ledger_margin.png)", ""]

    # ---- RQ3: handoff
    if (ART / "handoff_base.parquet").exists():
        hm = pd.read_parquet(ART / "handoff_meta.parquet")
        hb = pd.read_parquet(ART / "handoff_base.parquet")
        probs = pd.read_parquet(ART / "problems.parquet").set_index("problem_id")
        hb = hb.merge(hm, on="key")
        from reachtrace.sample.grade import grade

        hb["correct"] = [grade(t, probs.loc[p, "answer"]).correct
                         for t, p in zip(hb.text, hb.problem_id)]
        gi = grp.set_index("problem_id")
        base_rate = gi.c_base / gi.n_base
        hrows = []
        for pid, d in hb.groupby("problem_id"):
            # -0.01 is the unaided base (no prefix); 0.0 is the RL trace's first 16 tokens.
            pts = {-0.01: (int(gi.loc[pid, "n_base"]), int(gi.loc[pid, "c_base"]))}
            for f, dd in d.groupby("frac"):
                pts[float(f)] = (len(dd), int(dd.correct.sum()))
            curve = success_curve(pts)
            hrows.append({"problem_id": pid, "group": gmap[pid],
                          **{f"s@{k:g}": v for k, v in curve.items()},
                          "handoff": h_of(curve, 0.25),
                          "composition": composition_signature(curve, base_rate[pid])})
        hd = pd.DataFrame(hrows)
        hd.to_csv(out / "handoff_curves.csv", index=False)
        cols = [c for c in hd.columns if c.startswith("s@")]
        hs = hd.groupby("group")[cols].mean()
        hs["n"] = hd.groupby("group").size()
        hs["share h<=0.1"] = hd.groupby("group").handoff.apply(lambda s: (s <= 0.1).mean())
        hs["share composition"] = hd.groupby("group").composition.mean()
        md += ["## 3. Handoff map", "",
               "Base success rate when given the first fraction of a correct RL trace's content "
               "region (s@-0.01 = no prefix; s@0 = the RL trace's first 16 tokens only).", "",
               hs.to_markdown(floatfmt=".3f"), ""]
        fig, ax = plt.subplots(figsize=(6, 4))
        for g_, c in zip(("G", "B"), ("C3", "C1")):
            if g_ in hs.index:
                xs = [float(k[2:]) for k in cols]
                ax.plot(xs, hs.loc[g_, cols].to_numpy(float), marker="o", color=c, label=g_)
        ax.set_xlabel("fraction of RL content given to the base")
        ax.set_ylabel("base success rate")
        ax.legend()
        ax.set_title("Handoff: how much RL solution the base needs")
        fig.tight_layout()
        fig.savefig(out / "handoff.png", dpi=150)
        md += ["![handoff](handoff.png)", ""]

    (out / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    step = sys.argv[1]
    {"select": select, "download": download, "sample": sample, "escalate": escalate,
     "ledger": ledger, "handoff": handoff, "report": report}[step]()
