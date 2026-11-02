"""Within-stage attribution at 7B: which midtraining sources does RL-Zero's GRPO query use?

Plan v3 §5 (RQ4) for the midtraining stage, with the methods of
docs/06_attribution_methods.md. Olmo-3.1-7B-RL-Zero-Math's rollouts on held-out competition
problems it solves only sometimes become GRPO queries (Dr. GRPO advantages, so each problem's
query is its correct-minus-incorrect contrast). A balanced sample of the Dolmino midtraining
mix is scored at the end of midtraining (`stage2-step47684`), where the query gradients are
taken too: each stage's documents are scored at that stage's own end checkpoint.

The pilot found that midtraining is where correct solutions become likely and where the
model starts to prefer correct over incorrect content, and Ai2's midtraining data-mix arms
make that causal at the mix level (math-code-reasoning mix vs general-QA mix). This asks
which sources inside the mix the RL-gained contrast points to.

Stages (artifacts under $MID_ART, results in results/olmo7b_midtrain/):

  problems   the pilot's problems with mixed RL-Zero success (results/pilot)
  rollouts   8 RL-Zero samples per problem (vLLM in ~/vllm_env if present, else HF generate)
  queries    graded rollouts -> GRPO query rows for problems with mixed rewards
  docs       N documents per Dolmino source (first shard of each), 1,024 tokens
  attribute  TrackStar p32 (one column per rollout) over all documents; EK-FAC on the last
             4 layers (factors on one stratified subset, scores on another)
  evaluate   source and group rankings with permutation controls, split-half reliability,
             TrackStar / EK-FAC agreement, top documents
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

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("MID_ART", ROOT / "artifacts" / "olmo7b_midtrain")).resolve()
OUT = ROOT / "results" / "olmo7b_midtrain"
RL = "allenai/Olmo-3.1-7B-RL-Zero-Math"
BASE = "allenai/Olmo-3-1025-7B"
MIDTRAIN_END = "stage2-step47684"
DOLMINO = "allenai/dolma3_dolmino_mix-100B-1025"
SEED = 20261101
N_SAMPLES = 8
MAX_NEW = 7680  # the pilot's cap: prompt + rollout fit the 8,192-token stage-2 context
MAX_DOC = 1024
N_DOCS = int(os.environ.get("MID_DOCS", 120))
N_SHARDS = 4
EKFAC_FIT, EKFAC_SCORE = 24, 40  # documents per source for EK-FAC factors / scores
# EK-FAC scoring set: "score" (EKFAC_SCORE per source) or "nofit" (every document not used
# for the factors, 96 per source).
EKFAC_SET = os.environ.get("MID_EKFAC_SET", "nofit")
EKFAC_LAYERS, N_LAYERS = 4, 32
TOKEN_BATCH = 8192  # the stage-2 checkpoint's context; prompt + 7,680-token rollout fits
DOC_BATCH = 2048  # EK-FAC document passes (per-document gradients)

# Dolmino sources by role. The pilot's causal check (Ai2's midtraining arms from the same
# ~2T-token start) contrasts a math-code-reasoning mix with a general-QA mix.
SOURCES = {
    "math": ["dolmino-math", "cranemath", "megamatt", "tinymath-mind", "tinymath-pot"],
    "reasoning": ["qwq-reasoning-traces", "openthoughts2-reasoning-traces",
                  "llama_nemotron-reasoning-traces", "gemini-reasoning-traces",
                  "omr-rewrite-fullthoughts", "math-meta-reasoning"],
    "code": ["cranecode", "stack_edu-fim_vigintile_19_Python", "program_verifiable",
             "code-meta-reasoning"],
    "qa": ["nemotron-synth-qa", "tulu-3-sft", "dolmino_1-flan", "wiki_to_rcqa-part1",
           "reddit_to_flashcards"],
    "web": ["common_crawl-high-quality_20_science_math_and_technology",
            "common_crawl-high-quality_20_entertainment", "stem-heavy-crawl",
            "olmocr_science_pdfs-high_quality-science_math_and_technology-2e13"],
}
GROUP = {s: g for g, ss in SOURCES.items() for s in ss}
SHORT = {"common_crawl-high-quality_20_science_math_and_technology": "CC-HQ science/math/tech",
         "common_crawl-high-quality_20_entertainment": "CC-HQ entertainment",
         "olmocr_science_pdfs-high_quality-science_math_and_technology-2e13":
             "olmOCR science PDFs",
         "stack_edu-fim_vigintile_19_Python": "Stack-Edu Python"}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(RL)


# ------------------------------------------------------------------------------- problems


def problems() -> None:
    from reachtrace.data.evals import load_eval_set

    pg = pd.read_csv(ROOT / "results" / "pilot" / "problems_groups.csv")
    mixed = pg[(pg.c_rl >= 2) & (pg.c_rl <= pg.n_rl - 2)]
    probs = pd.concat([load_eval_set(s) for s in sorted(mixed.eval_set.unique())],
                      ignore_index=True)
    probs = probs[probs.problem_id.isin(mixed.problem_id)].merge(
        mixed[["problem_id", "group", "c_base", "n_base", "c_rl", "n_rl"]], on="problem_id")
    ART.mkdir(parents=True, exist_ok=True)
    probs.to_parquet(ART / "problems.parquet")
    print(f"{len(probs)} problems with mixed RL-Zero success in the pilot")
    print(probs.groupby("eval_set").size().to_string())


# ------------------------------------------------------------------------------- rollouts


def rollouts() -> None:
    """vLLM in a separate environment when available (much faster), else HF generate."""
    prompts()
    vllm_py = Path.home() / "vllm_env" / "bin" / "python"
    if vllm_py.exists() and os.environ.get("MID_ENGINE", "vllm") == "vllm":
        cmd = [str(vllm_py), __file__, "rollouts_vllm"]
        print("+", " ".join(cmd), flush=True)
        if subprocess.run(cmd, env={**os.environ, "PYTHONPATH": ""}).returncode == 0:
            return
        print("vLLM failed; falling back to HF generate", flush=True)
    rollouts_hf()


def _prompts() -> pd.DataFrame:
    from reachtrace.prompts import get_template

    probs = pd.read_parquet(ART / "problems.parquet")
    t = get_template("rlzero31_math")
    return probs.assign(prompt=[t.render(q) for q in probs.question])


def rollouts_vllm() -> None:
    # Runs in ~/vllm_env, which has vLLM but not this package: no reachtrace imports here.
    from vllm import LLM, SamplingParams

    probs = pd.read_parquet(ART / "prompts.parquet")
    llm = LLM(model=RL, dtype="bfloat16", max_model_len=8192, seed=SEED,
              gpu_memory_utilization=0.9)
    sp = SamplingParams(n=N_SAMPLES, temperature=1.0, top_p=1.0, max_tokens=MAX_NEW,
                        seed=SEED, stop=["\nSolve the following"])
    t0 = time.time()
    outs = llm.generate(probs.prompt.tolist(), sp)
    recs = []
    for pid, o in zip(probs.problem_id, outs):
        for j, c in enumerate(o.outputs):
            recs.append({"problem_id": pid, "sample_idx": j,
                         "prompt_ids": list(o.prompt_token_ids),
                         "completion_ids": list(c.token_ids), "text": c.text,
                         "finish_reason": c.finish_reason})
    pd.DataFrame(recs).to_parquet(ART / "rollouts.parquet")
    n_tok = sum(len(r["completion_ids"]) for r in recs)
    print(f"vLLM: {len(recs)} rollouts, {n_tok} tokens in {time.time() - t0:.0f}s "
          f"({n_tok / (time.time() - t0):.0f} tok/s)", flush=True)


def rollouts_hf() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    probs = pd.read_parquet(ART / "prompts.parquet")
    tok = _tok()
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(RL, dtype=torch.bfloat16).cuda().eval()
    bs = int(os.environ.get("MID_GEN_BATCH", 8))  # 32 KV heads: ~0.5 MB/token
    jobs = [(pid, p, j) for pid, p in zip(probs.problem_id, probs.prompt)
            for j in range(N_SAMPLES)]
    recs, t0 = [], time.time()
    torch.manual_seed(SEED)
    for b in range(0, len(jobs), bs):
        chunk = jobs[b: b + bs]
        enc = tok([p for _, p, _ in chunk], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                                 max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
        L = enc.input_ids.shape[1]
        for (pid, _, j), row, mask in zip(chunk, out, enc.attention_mask):
            comp = row[L:].tolist()
            hit_eos = tok.eos_token_id in comp
            if hit_eos:
                comp = comp[: comp.index(tok.eos_token_id) + 1]
            recs.append({"problem_id": pid, "sample_idx": j,
                         "prompt_ids": row[:L][mask.bool()].tolist(),
                         "completion_ids": comp, "text": tok.decode(comp,
                                                                    skip_special_tokens=True),
                         "finish_reason": "stop" if hit_eos else "length"})
        print(f"  {len(recs)}/{len(jobs)} rollouts ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(recs).to_parquet(ART / "rollouts.parquet")


def queries() -> None:
    """Grade rollouts; problems with 1-7 of 8 correct become GRPO query rows."""
    from datasets import Dataset

    from reachtrace.attrib.grpo_query import dr_grpo_advantage
    from reachtrace.sample.grade import grade

    probs = pd.read_parquet(ART / "problems.parquet").set_index("problem_id")
    r = pd.read_parquet(ART / "rollouts.parquet")
    r["correct"] = [grade(t, probs.answer[p]).correct for t, p in zip(r.text, r.problem_id)]
    r["n_tokens"] = r.completion_ids.map(len)
    k = r.groupby("problem_id").correct.agg(["sum", "size"])
    keep = k[(k["sum"] > 0) & (k["sum"] < k["size"])].index
    q = r[r.problem_id.isin(keep)].sort_values(["problem_id", "sample_idx"]).copy()
    q["reward"] = q.correct.astype(float)
    q["advantage"] = dr_grpo_advantage(q.reward, q.problem_id)
    Dataset.from_dict({
        "input_ids": [list(a) + list(b) for a, b in zip(q.prompt_ids, q.completion_ids)],
        "labels": [[-100] * len(a) + list(b) for a, b in zip(q.prompt_ids, q.completion_ids)],
        "advantage": q.advantage.tolist(),
    }).save_to_disk(str(ART / "queries"))
    q.drop(columns=["prompt_ids", "completion_ids"]).to_parquet(ART / "queries_meta.parquet")
    summary = r.groupby("problem_id").agg(correct=("correct", "sum"), n=("correct", "size"),
                                          mean_tokens=("n_tokens", "mean"),
                                          capped=("finish_reason",
                                                  lambda s: (s == "length").mean()))
    summary["used"] = summary.index.isin(keep)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.join(probs[["eval_set", "group"]]).to_csv(OUT / "queries.csv")
    print(summary.to_string())
    print(f"{len(keep)} problems with mixed rewards -> {len(q)} query rows, "
          f"{q.n_tokens.sum()} completion tokens")


# ------------------------------------------------------------------------------- docs


def docs() -> None:
    import io

    import zstandard
    from datasets import Dataset
    from huggingface_hub import HfApi, hf_hub_download

    # A source's files are often split by subset or topic (dolmino-math by origin dataset,
    # stem-heavy-crawl by WebOrganizer topic), so documents come from N_SHARDS files drawn
    # with probability proportional to size, i.e. roughly in proportion to the source's
    # bytes, not from its alphabetically first file.
    tree = [f for f in HfApi().list_repo_tree(DOLMINO, repo_type="dataset", path_in_repo="data",
                                              recursive=True)
            if getattr(f, "size", None) and f.path.endswith(".jsonl.zst")]
    tok = _tok()
    rng = np.random.default_rng(SEED)
    rows = []
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
        pick = rng.choice(len(texts), size=min(N_DOCS, len(texts)), replace=False)
        for i in pick:
            shard, text = texts[i]
            ids = tok(text, add_special_tokens=False).input_ids[:MAX_DOC]
            rows.append({"source": src, "group": grp, "shard": shard, "text": text[:4000],
                         "input_ids": ids})
        print(f"{src}: {len(pick)} of {len(texts)} read from {k} of {len(files)} files: "
              f"{[s.split('/')[-1] for s in shards]}", flush=True)
    df = pd.DataFrame(rows)
    # Stratified subsets for EK-FAC: factors on one, scores on another (disjoint).
    df["ekfac"] = ""
    for _, g in df.groupby("source"):
        order = rng.permutation(g.index.to_numpy())
        df.loc[order[:EKFAC_FIT], "ekfac"] = "fit"
        df.loc[order[EKFAC_FIT: EKFAC_FIT + EKFAC_SCORE], "ekfac"] = "score"
    ART.mkdir(parents=True, exist_ok=True)
    for name, d in [("docs", df), ("docs_fit", df[df.ekfac == "fit"]),
                    ("docs_score", df[df.ekfac == "score"]),
                    ("docs_nofit", df[df.ekfac != "fit"])]:
        Dataset.from_dict({"input_ids": d.input_ids.tolist(),
                           "labels": d.input_ids.tolist()}).save_to_disk(str(ART / name))
    df.drop(columns=["input_ids"]).assign(n_tokens=df.input_ids.map(len)).to_parquet(
        ART / "docs_meta.parquet")
    print(f"{len(df)} documents, {df.input_ids.map(len).sum()} tokens")


# ------------------------------------------------------------------------------- attribute


def _bergson(name: str, steps: list[dict]) -> float:
    from reachtrace.attrib import bergson_yaml as by

    y = by.write_pipeline(ART / f"{name}.yaml", str(ART / name), steps)
    cmd = [sys.executable, "-m", "bergson", str(y)]
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def attribute(only: str = "") -> None:
    from reachtrace.attrib import bergson_yaml as by

    want = set(only.split(",")) if only else {"trackstar", "ekfac"}
    common = {"precision": "bf16", "token_batch_size": TOKEN_BATCH, "max_batch_size": None,
              "revision": MIDTRAIN_END}
    query = by.data_cfg(str(ART / "queries"), advantage=True)
    timing = {}
    tpath = ART / "timing.json"
    if tpath.exists():
        timing = json.loads(tpath.read_text())
    if "trackstar" in want and not (ART / "trackstar_p32" / "scores").exists():
        rp = str(ART / "trackstar_p32")
        timing["trackstar_p32"] = _bergson("trackstar_p32", [by.trackstar_step(
            rp, BASE, by.data_cfg(str(ART / "docs")), query, projection_dim=32, **common)])
        _write_json(tpath, timing)
    if "ekfac" in want and not (ART / "ekfac" / "scores").exists():
        keep = list(range(N_LAYERS - EKFAC_LAYERS, N_LAYERS))
        filt = ",".join(["lm_head", *(f"layers.{i}.*" for i in range(N_LAYERS)
                                      if i not in keep)])
        shared = str(ART / "ekfac")
        # EK-FAC's eigenvalue correction holds per-document gradients, so documents go in
        # small token batches; only the 8k-token query rows need TOKEN_BATCH. Each problem's
        # query sum_i A_i g_i is built on its own (aggregation "sum") and the rows merged,
        # as in gate v3, so one influence pass scores every problem.
        small = {**common, "token_batch_size": DOC_BATCH}
        if not (ART / "ekfac" / "hessian" / "kfac").exists():
            timing["ekfac_factors"] = _bergson("ekfac_factors", [by.kfac_hessian_step(
                shared, BASE, by.data_cfg(str(ART / "docs_fit")), filter_modules=filt,
                **small)])
        qm = pd.read_parquet(ART / "queries_meta.parquet")
        qdir = ART / "ekfac_query"
        steps, dirs, start = [], [], 0
        for j, (_, n) in enumerate(qm.groupby("problem_id", sort=True).size().items()):
            d = qdir / f"p{j:02d}"
            steps.append({"build": {
                "index_cfg": by.index_cfg(str(d), BASE, by.data_cfg(
                    str(ART / "queries"), split=f"train[{start}:{start + n}]", advantage=True),
                    filter_modules=filt, **common),
                "preprocess_cfg": {"aggregation": "sum", "unit_normalize": False}}})
            dirs.append(d)
            start += n
        if not (qdir / "merged" / "info.json").exists():
            y = by.write_pipeline(ART / "ekfac_query.yaml", str(qdir), steps)
            t0 = time.time()
            subprocess.run([sys.executable, "-m", "bergson", str(y)], check=True)
            timing["ekfac_query"] = time.time() - t0
            by.merge_query_indices(dirs, qdir / "merged")
        timing["ekfac_scores"] = _bergson("ekfac", [by.influence_step(
            shared, BASE, by.data_cfg(str(ART / f"docs_{EKFAC_SET}")),
            {"path": str(qdir / "merged")}, aggregation="none", filter_modules=filt,
            # 19 full-gradient query rows of 4 layers are ~3 GB each in fp32: score them
            # five at a time in bf16 (one pass over the documents per batch).
            score_kw={"query_batch_size": 5, "precision": "bf16"}, **small)])
        _write_json(tpath, timing)


# ------------------------------------------------------------------------------- evaluate


def _scores(path: Path) -> np.ndarray:
    from bergson.data import load_scores

    s = load_scores(path)
    return np.asarray(s[:], dtype=np.float64).reshape(len(s), -1)


def _zcols(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(0)) / s.std(0)


def evaluate() -> None:
    from scipy.stats import spearmanr

    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(ART / "docs_meta.parquet")
    qm = pd.read_parquet(ART / "queries_meta.parquet")
    A = qm.advantage.to_numpy()
    pids = qm.problem_id.to_numpy()
    probs = sorted(set(pids))
    P = np.stack([pids == p for p in probs], 1).astype(float)  # rows x problems
    rng = np.random.default_rng(0)

    # TrackStar columns are sign(A_i) * cos(q_i, d) (Bergson normalizes after weighting);
    # cos is recovered exactly, so any advantage vector can be applied afterwards.
    raw = _scores(ART / "trackstar_p32" / "scores")
    cos = raw * np.sign(A)[None, :]

    def problem_scores(adv: np.ndarray) -> np.ndarray:
        return cos @ (adv[:, None] * P)  # docs x problems: sum_i A_i cos_i

    def doc_score(adv: np.ndarray) -> np.ndarray:
        return _zcols(problem_scores(adv)).mean(1)  # each problem weighs the same

    def perm_adv() -> np.ndarray:
        out = A.copy()
        for p in probs:
            i = np.flatnonzero(pids == p)
            out[i] = rng.permutation(A[i])
        return out

    src = meta.source.to_numpy()
    sources = list(GROUP)
    groups = list(SOURCES)

    def by_source(x: np.ndarray) -> np.ndarray:
        return np.array([x[src == s].mean() for s in sources])

    def by_group(x: np.ndarray) -> np.ndarray:
        g = meta.group.to_numpy()
        return np.array([x[g == k].mean() for k in groups])

    obs = doc_score(A)
    generic = _zcols(cos @ (P / P.sum(0))).mean(1)  # mean rollout, no contrast
    # Null: advantages permuted within each problem (a contrast between random rollouts).
    # Engagement (exploratory): mean |z| over problems, i.e. how strongly a document is
    # pushed either way by the contrast, against the same null.
    null, null_eng = [], []
    for _ in range(int(os.environ.get("MID_PERMS", 1000))):
        Z = _zcols(problem_scores(perm_adv()))
        null.append(Z.mean(1))
        null_eng.append(by_source(np.abs(Z).mean(1)))
    null_src = np.stack([by_source(x) for x in null])
    null_grp = np.stack([by_group(x) for x in null])
    null_eng = np.stack(null_eng)
    eng = np.abs(_zcols(problem_scores(A))).mean(1)

    def boot_src(x: np.ndarray, n: int = 2000) -> np.ndarray:
        out = np.zeros((n, len(sources)))
        for b in range(n):
            out[b] = [x[rng.choice(np.flatnonzero(src == s), (src == s).sum())].mean()
                      for s in sources]
        return out

    bs, bg = boot_src(obs), boot_src(generic)
    ts_src = by_source(obs)
    tab = pd.DataFrame({
        "source": sources, "group": [GROUP[s] for s in sources],
        "n_docs": [(src == s).sum() for s in sources],
        "trackstar_z": ts_src,
        "ci_lo": np.percentile(bs, 2.5, 0), "ci_hi": np.percentile(bs, 97.5, 0),
        "perm_p_high": (null_src >= ts_src).mean(0), "perm_p_low": (null_src <= ts_src).mean(0),
        "generic_z": by_source(generic),
        "generic_ci_lo": np.percentile(bg, 2.5, 0), "generic_ci_hi": np.percentile(bg, 97.5, 0),
        "engagement": by_source(eng), "engagement_null": null_eng.mean(0),
        "engagement_perm_p": (null_eng >= by_source(eng)).mean(0),
    })

    def split_half(M: np.ndarray, n: int = 200) -> float:
        """Mean Spearman between source rankings from two random halves of the problems
        (plan v3's reliability gate asks for >= 0.5)."""
        out = []
        for _ in range(n):
            perm = rng.permutation(M.shape[1])
            a, b = perm[: M.shape[1] // 2], perm[M.shape[1] // 2:]
            out.append(spearmanr(by_source(_zcols(M[:, a]).mean(1)),
                                 by_source(_zcols(M[:, b]).mean(1)))[0])
        return float(np.mean(out))

    r_half, r_gen = split_half(problem_scores(A)), split_half(cos @ (P / P.sum(0)))
    eng_half = []
    for _ in range(200):
        perm = rng.permutation(len(probs))
        a, b = perm[: len(probs) // 2], perm[len(probs) // 2:]
        Z = _zcols(problem_scores(A))
        eng_half.append(spearmanr(by_source(np.abs(Z[:, a]).mean(1)),
                                  by_source(np.abs(Z[:, b]).mean(1)))[0])
    res = {"n_problems": len(probs), "n_query_rows": len(qm), "n_docs": len(meta),
           "split_half_spearman": r_half,
           "split_half_spearman_brown": 2 * r_half / (1 + r_half),
           "generic_split_half_spearman": r_gen,
           "generic_split_half_spearman_brown": 2 * r_gen / (1 + r_gen),
           "contrast_vs_generic_spearman": spearmanr(ts_src, by_source(generic))[0],
           "engagement_split_half_spearman": float(np.mean(eng_half)),
           "engagement_excess_vs_generic_spearman": spearmanr(by_source(eng) - null_eng.mean(0),
                                                    by_source(generic))[0]}

    # Groups: the arms' contrast (math + code + reasoning vs general QA).
    g_obs = by_group(obs)
    grp = pd.DataFrame({"group": groups, "trackstar_z": g_obs,
                        "perm_p_high": (null_grp >= g_obs).mean(0),
                        "perm_p_low": (null_grp <= g_obs).mean(0)})
    is_mcr = np.isin(meta.group, ["math", "code", "reasoning"])
    is_qa = meta.group.to_numpy() == "qa"

    def arms(x):
        return x[is_mcr].mean() - x[is_qa].mean()

    res["mcr_minus_qa"] = arms(obs)
    res["mcr_minus_qa_perm_p"] = float(np.mean([arms(x) >= arms(obs) for x in null]))

    # EK-FAC on its stratified subset: per-problem scores, agreement with TrackStar on the
    # same documents.
    if (ART / "ekfac" / "scores").exists():
        ek = _scores(ART / "ekfac" / "scores")  # one column per problem (sum_i A_i g_i)
        sub = meta.index[(meta.ekfac == "score") if EKFAC_SET == "score"
                         else (meta.ekfac != "fit")].to_numpy()
        ek_doc = _zcols(ek).mean(1)
        ts_doc = _zcols(problem_scores(A)[sub]).mean(1)
        s_sub = src[sub]
        ek_src = np.array([ek_doc[s_sub == s].mean() for s in sources])
        res["ekfac_split_half_spearman"] = float(np.mean([
            spearmanr(*[[_zcols(ek[:, h]).mean(1)[s_sub == s].mean() for s in sources]
                        for h in np.array_split(rng.permutation(ek.shape[1]), 2)])[0]
            for _ in range(200)]))
        ts_sub_src = np.array([ts_doc[s_sub == s].mean() for s in sources])
        tab["ekfac_z"] = ek_src
        ek_bs = np.array([[ek_doc[rng.choice(np.flatnonzero(s_sub == s), (s_sub == s).sum())]
                           .mean() for s in sources] for _ in range(2000)])
        tab["ekfac_ci_lo"] = np.percentile(ek_bs, 2.5, 0)
        tab["ekfac_ci_hi"] = np.percentile(ek_bs, 97.5, 0)
        res["ekfac_vs_trackstar_source_spearman"] = spearmanr(ek_src, ts_sub_src)[0]
        res["ekfac_vs_trackstar_doc_spearman"] = spearmanr(ek_doc, ts_doc)[0]
        res["ekfac_mcr_minus_qa"] = (ek_doc[np.isin(meta.group.to_numpy()[sub],
                                                    ["math", "code", "reasoning"])].mean()
                                     - ek_doc[meta.group.to_numpy()[sub] == "qa"].mean())
        k = max(1, len(sub) // 10)
        res["top10pct_overlap"] = len(set(np.argsort(-ek_doc)[:k])
                                      & set(np.argsort(-ts_doc)[:k])) / k
    tab.sort_values("trackstar_z", ascending=False).to_csv(OUT / "sources.csv", index=False)
    grp.to_csv(OUT / "groups.csv", index=False)
    timing = json.loads((ART / "timing.json").read_text()) if (ART / "timing.json").exists() \
        else {}
    _write_json(OUT / "summary.json", {**res, "timing_s": timing})
    top = meta.assign(score=obs).sort_values("score", ascending=False)
    with open(OUT / "top_docs.md", "w", encoding="utf-8") as f:
        f.write("# Highest- and lowest-scored documents (TrackStar p32, GRPO query)\n\n")
        for title, d in [("Highest", top.head(8)), ("Lowest", top.tail(5))]:
            f.write(f"## {title}\n\n")
            for _, r in d.iterrows():
                text = " ".join(r.text[:300].split())
                f.write(f"- **{r.source}** (z = {r.score:+.2f}): {text}...\n")
            f.write("\n")
    print(tab.sort_values("trackstar_z", ascending=False).round(3).to_string(index=False))
    print(grp.round(3).to_string(index=False))
    print(json.dumps(res, indent=1, default=float))
    figure(tab)


def figure(tab: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"math": "#2a78d6", "reasoning": "#1f9e89", "code": "#8e5bd0", "qa": "#eb6834",
              "web": "#9a9893"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})
    t = tab.sort_values("generic_z")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 0.26 * len(t) + 1.4), sharey=True)
    y = np.arange(len(t))
    panels = [(axes[0], "generic_z", "generic_ci_lo", "generic_ci_hi",
               "plain rollout query\n(what RL-Zero's solutions resemble)"),
              (axes[1], "trackstar_z", "ci_lo", "ci_hi",
               "GRPO query, correct minus incorrect\n(what separates correct solutions)")]
    for ax, col, lo, hi, title in panels:
        for g, c in colors.items():
            m = (t.group == g).to_numpy()
            ax.hlines(y[m], t[lo][m], t[hi][m], color=c, lw=2)
            ax.plot(t[col][m], y[m], "o", color=c, ms=6, mec="white", mew=1.2, label=g)
        ax.axvline(0, color=muted, lw=0.8)
        ax.set_title(title, fontsize=8.5, color=ink, loc="left")
        ax.grid(axis="x", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("mean z-score of the source's documents\n(95% CI over documents)")
    if "ekfac_z" in t:
        axes[1].plot(t.ekfac_z, y, "D", color=ink, ms=4, mfc="none",
                     label=f"EK-FAC\n({EKFAC_SCORE if EKFAC_SET == 'score' else 96} "
                     "docs/source)")
    axes[0].set_yticks(y, [SHORT.get(x, x) for x in t.source], fontsize=7)
    axes[1].legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(1.01, 0))
    fig.tight_layout()
    fig.savefig(OUT / "sources.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def nofit() -> None:
    """`docs_nofit`: every document not used for the EK-FAC factors (for artifacts made
    before `docs` wrote it)."""
    from datasets import load_from_disk

    meta = pd.read_parquet(ART / "docs_meta.parquet")
    load_from_disk(str(ART / "docs")).select(
        np.flatnonzero(meta.ekfac.to_numpy() != "fit").tolist()).save_to_disk(
        str(ART / "docs_nofit"))


def prompts() -> None:
    _prompts()[["problem_id", "prompt"]].to_parquet(ART / "prompts.parquet")


STAGES = {"problems": problems, "prompts": prompts, "rollouts": rollouts, "nofit": nofit,
          "rollouts_vllm": rollouts_vllm, "rollouts_hf": rollouts_hf, "queries": queries,
          "docs": docs, "attribute": attribute, "evaluate": evaluate}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: olmo7b_midtrain.py {{{','.join(STAGES)}}}")
    STAGES[sys.argv[1]](*sys.argv[2:])
