# Implementation plan

Work packages are ordered by dependency, not by date. Each lists inputs, outputs (artifact paths), the code that implements it, the procedure, and the gate that decides what happens next. GPU-hour figures are planning numbers in A100-80GB hours. They assume 5k output tokens/s per GPU for vLLM decoding and 4k tokens/s for 7B forward+backward with projection, and are replaced by the WP2 benchmark.

```
WP0 pin ─► WP1 evals+decon ─► WP2 signal check ─► WP3 reachability ─► WP4 provenance
                                     │                    │
                                     ▼                    ▼
                              WP5 synthetic gate ──► WP6 7B attribution ─► WP7 analysis ─► WP8 release
```

WP5 needs nothing from WP1–WP4 and runs in parallel on small GPUs.

---

## WP0 Environment and pinning: done

- **Environments.** `uv venv --python 3.11`. The CPU core is `pip install -e ".[dev]"`. The GPU stages use `.[sample,attrib,data]`: vLLM 0.32 needs transformers ≥5.16.1,<5.20 and Bergson ≥2.3 needs transformers ≥5.0, so one environment works. Synthetic training uses `.[synth]` on any GPU with ≥24 GB.
- **Pins.** `reachtrace pin` writes [configs/pins.lock.yaml](../configs/pins.lock.yaml): base `main` plus stage-1/2/3 finals plus three midtraining arms, and RL-Zero 3.1 `main` plus steps 100/200/400/800/1600/2800. `main` and `stage3-step11921` have different commit SHAs but identical safetensors hashes, so the RL-Zero starting point is the stage-3 final.
- **Rule.** Every artifact key includes the revision SHA (`cache.py`). Re-pinning invalidates downstream caches by construction.

## WP1 Evaluation set and decontamination

**Inputs:** the eval sets in `data/evals.py` (verified IDs and columns), Dolci-RL-Zero-Math-7B, and its upstream sources.
**Outputs:** `artifacts/evals/core.parquet`, `artifacts/decon/ngram_report.parquet`, `artifacts/decon/decon_report/`, `artifacts/evals/clean.parquet` (strict and RL-clean flags).

1. `reachtrace evals` builds core: MATH-500, MATH L4–5, AIME 2024 and 2025, AMC 2023, and OlympiadBench (EN, text-only, single-answer). Cross-set duplicates are removed by normalised-question hash.
2. `reachtrace decon-ngram` runs an exact 13-gram scan against Dolci, DAPO-Math-17k and KlearReasoner-MathSub-30K. **Open item:** add the Open-Reasoner-Zero and OMEGA sources (Hub IDs and text columns not yet verified; `data/dolci.py:UPSTREAM_REFS`).
3. `decon detect --training-dir <dolmino math sources> --evals-dir <refs>` scans the Dolmino 7B mix math sources in full (about 20B tokens), plus the Dolma 3 attribution working set.
4. Full-corpus exact-match check: query the HCAI-Lab Dolma 3 6T bloom/index artifacts or an infini-gram index, if one covering Dolma 3 is reachable (open item).
5. Policy: **drop** anything flagged against RL-stage references. **Tag** pretraining-only flags and report every result on both the strict and RL-clean subsets. The headline table gives pre/post counts per set.

**Gate:** at least 1,000 clean problems. If fewer, add MATH Level 5 from train-disjoint splits and further competition sets after decon (candidate: Omni-MATH; ID to verify).

## WP2 Phase-0 signal check (one GPU)

**Purpose:** confirm the pipeline reproduces Ai2's anchor, measure throughput, and see whether certified U exists at all.

1. **Anchor.** RL-Zero 3.1 `main`, AIME 2024 and 2025, temperature 1.0, 32 samples, 32K max tokens. Bootstrapped pass@1 must be within ±5 points of the ~50% plateau in Olmo 3 report Fig. 39. Base model, same prompt string: record pass@1 and pass@32.
2. **Pilot partition.** 200 problems (100 MATH L4–5, 30 AIME24, 30 AIME25, 40 AMC23), both models, adaptive sampling to `n_max = 288`. Report certified labels at K ∈ {1, 4, 16, 64} with hierarchical-bootstrap CIs (`reach/partition.py:bootstrap_counts`).
3. **Base truncation rate.** Fraction of base samples with `finish_reason == "length"`. If the base model rambles to 16K, cap its `max_tokens` at a level justified by the RL model's 99th-percentile correct-solution length, and document the cap.
4. **Template sensitivity.** 50 problems with the 3.1 template, the 3.0 template, and a 4-shot Minerva prompt, on the base model.
5. **Cue pilot.** Base model on the pilot U set with each configured cue (`--cue ".\n\nOkay"`). Before fixing the cue list, also measure the RL model's empirical opening-token distribution on its correct solutions.
6. **Dolci pilot index.** `bergson build` over Dolci prompts and 4 RL rollouts at `main` (identity tokenizer, `--advantage_column advantage`). Gradient-cosine scoring of 20 U and 20 hard-R traces, shown next to the norm-only ranking.

**Gate:**
- **Go** if the anchor reproduces and certified U@16 ≥ 10.
- If U is near zero after cue-forcing, go anyway, reframed: the paper becomes an elicitation-dominant decomposition, and layers 1–2 carry it.
- If the anchor misses, stop and debug the template, max tokens and grader before anything else.

**Budget:** about 15 GPU-h.

## WP3 Reachability instrument at scale

**Outputs:** `artifacts/gen/<pair>/<side>/<revision>/*.parquet`, `artifacts/partitions/partition.parquet`, `artifacts/partitions/cue_<hash>.parquet`.

1. Run `reachtrace sample` for base and RL over the clean set. The adaptive rule is identical for both models: minimum 32, batch 32, stop at c ≥ 8 or when every K in the sweep is determined, ceiling 288. pass@k *curves* use each problem's first 32 samples only (unbiased); labels use all samples.
2. Run `reachtrace partition` for certified labels at K ≤ 64. Observed-mode labels at 256 and 1024 come from a ceiling extension on base-unresolved problems only, and are labelled "observed support" in every figure.
3. **Cue-forcing decomposition.** For certified U@K and hard-R, rerun the base model with the top RL opening cues and define:
   - **U_cue:** base becomes reachable under some cue;
   - **U_late:** remains, but is reachable only at later RL steps (from WP4);
   - **U_deep:** the remainder.
4. **Hard-R** is R with base solve rate in (0, 0.05]. It is the matched comparison group throughout.
5. **Matching** (`analysis/matching.py`): U_deep vs hard-R on RL pass@1 quintile × source set × MATH level (where defined) × RL correct-solution length tercile. Report the matched counts and the MDE (`analysis/stats.mde`).

**Budget:** about 45 GPU-h (1,000 problems, mean n ≈ 140 base and 100 RL, 2.5K and 4K tokens) plus about 25 GPU-h for two cue arms on about 300 problems.

## WP4 Checkpoint provenance (no attribution)

**Revisions** (`reach/stage_ckpt.py:provenance_revisions`): `stage1_final`, `stage2_final` (`stage3_final` = `main`), three midtraining arms, and RL steps 100, 200, 400, 800, 1600 and 2800.

1. Problems: U@16 ∪ U@64 ∪ L ∪ a random hard-R sample of equal size (≈ 300). Use the same sampler and prompt. n_max is 288 for base revisions and 128 for RL steps, since RL steps only need "reachable yes/no" at K ≤ 16.
2. `provenance_table` gives the first revision at which each problem becomes reachable.
3. Analyses:
   - (a) the fraction of U already reachable at `stage2_final` under cue-forcing, which tests whether midtraining supplies it;
   - (b) reachability across the three midtraining arms: is U reachability higher under `math-code-reasoning-web` than under `gen-mc-only`? This is a paired test per problem, with McNemar on certified labels and a paired bootstrap on pass@16;
   - (c) the RL step of first reach versus Dolci proximity (WP7, H2′ behavioural).

**Budget:** about 60 GPU-h. This is the most flexible package: halve it by dropping steps 200 and 1600.

## WP5 Synthetic planted-fragment gate (small GPUs, parallel)

**Code:** `synthetic/tasks.py`, `synthetic/train.py`, `synthetic/groundtruth.py`. **Config:** `configs/synthetic/fragments_v1.yaml`.

1. **Corpus.** `reachtrace synth-corpus` writes 4 × 2,000 planted documents: A = `zorp` substitution, B = `blick` positional, C = `zarp` and D = `blink` look-alikes, in four styles. No document contains both `zorp` and `blick` (tested). These are mixed into 50M generic tokens from FineWeb-Edu.
2. **CPT.** `continued_pretrain` on SmolLM2-135M (fall back to Pythia-160M). Check that A-only and B-only tasks are solvable, and that `blick(zorp(x))` at train lengths 3–4 has **non-zero but rare** base success (pass@64 between 1% and 30%). Adjust lengths or doc counts until true; this is the edge-of-competence condition.
3. **GRPO.** `grpo` with OLMoRL-like settings (dapo loss, no std scaling, β = 0, clip 0.2/0.272, G = 8, T = 1.0) on `rl_train`, over 3 seeds. Every rollout is logged with its text. Evaluate the U-type sets (`eval_long`, `eval_deep`) and the controls (`ctrl_A/B/C`).
4. **Ground truth.**
   - (i) Leave-family-out: retrain CPT+GRPO without A, B, C and D in turn.
   - (ii) LDS: 20 random 50% subsets of the planted documents with generic text fixed; retrain CPT+GRPO with shortened GRPO; record `eval_long` accuracy.
5. **Attribution, exactly the 7B protocol.** Queries are fork-token spans of correct post-GRPO traces on `eval_long`. Corpus documents are scored under the CPT model and under the GRPO model, and RL items are scored with `--advantage_column`. The bin is the family.
6. **Sweep.** Projection dim ∈ {16, 32, 64, 128}; target ∈ {full response, answer line, fork tokens}; unit-norm on or off; TrackStar vs gradient cosine vs EK-FAC. The sweep output is the tuning guide.

**Gate criteria (pre-registered):**

| Gate | Criterion |
|---|---|
| G1 recovery | min(A, B) family mass > max(C, D) on 3/3 seeds; LDS ≥ 0.3 |
| G2 controls | targeted minus norm-only recall@k, and targeted minus fluency-residualised recall@k: 95% CI excludes 0 on ≥ 2/3 seeds |
| G3 co-influence | A–B co-influence rises after GRPO beyond the drift on ctrl_A/ctrl_B queries; CI excludes 0 |
| G4 stability | family ranking Kendall τ ≥ 0.6 across two generation draws |

**Decisions:**
- All pass → WP6 with the winning settings.
- G1 and G2 pass, G3 fails → WP6 for H1/H2′ only; H3 reported as not measurable.
- G1 or G2 fails → **no 7B attribution.** Write the methods paper (gate plus tuning guide) and the behavioural paper (WP3 and WP4).

**Budget:** about 80 small-GPU hours, dominated by the 20 LDS retrains.

## WP6 7B attribution

**Inputs:** WP3 partitions (U_deep, hard-R, random N, and non-math control queries), WP5 settings, and the HCAI-Lab Dolma 3 working set.

1. **Corpus sample (Tier A, ≈ 550k documents at 1,024 tokens).**
   - Pretraining: 300k from the HCAI-Lab working set, with 200k from math, science and code topics and 100k uniform control, carrying their WebOrganizer bins.
   - Midtraining: 200k from `dolma3_dolmino_mix-100B-1025`, stratified by source: 120k math, 40k code, 40k QA/thinking.
   - `data/corpus_sample.py:stratified_sample` and `doc_meta` produce the row-aligned sidecar.
   - Strategy clusters: k-means with k = 256 on embeddings of the math documents, as a second binning.
2. **RL items.** Rollouts on all 13,314 Dolci prompts: 4 per prompt at `step_0400` and at `main` (Tier A), or 8 per prompt at 4 steps (Tier B). Rewards come from the same grader; `data/dolci.py:build_rl_items` computes Dr. GRPO advantages. **Zero-advantage groups are skipped**, since they carry no training gradient.
3. **Queries.** Up to 300 U_deep, 300 matched hard-R and 100 N/random problems, with 4 independent correct RL traces each.
   - Compute teacher-forced log-probs under both models (`attrib/targets.py:teacher_forced_logprobs`). Fork spans have log-ratio > ln 4, merged with gap 1; there is also an answer-line span.
   - One Bergson row per span (`span_queries`). After building the query index, sum span gradients per trace before scoring. This is linear, so it is exact for dot-product scoring and approximate under unit norm; both are reported.
4. **Tokenizer.** Use `write_identity_tokenizer` for every prompt/completion index, and check that one rendered row round-trips exactly before any large build.
5. **Scoring** (`attrib/bergson_cmd.py`):
   - query indices under both models: `q_rl__rlzero31`, `q_rl__base`;
   - `bergson score` streams the corpus under each model against the matching query index;
   - `bergson build` stores the RL-item index under the RL model;
   - `bergson build` stores a 50k-document corpus subsample with norms, for the norm-only control;
   - optional TrackStar preconditioning with the HCAI-Lab preconditioner for the base model, and a preconditioner fitted on a 100k subsample for the RL model.
6. **Aggregation** (`attrib/aggregate.py`): bin mass (mean per bin minus the global mean), stage share, and `sum_spans`. Controls (`attrib/controls.py`): remove the norm-only profile, residualise fluency within strata, and run the random-query baseline.
7. **Robustness.** Bergson `contrast` queries (U minus hard-R mean gradient), and probe-based attribution (activation differences) on a subsample.

**Gate before analysis:** split-half Kendall τ of bin rankings ≥ 0.5, and targeted beats norm-only on U ∪ R. If either fails, report the limits result and stop at layers 1–3.

**Budget, Tier A:**
- query indices: 2 models × about 90M tokens ≈ 13 h;
- streamed corpus: 2 × 0.56B tokens ≈ 80 h;
- RL rollouts: about 320M tokens, ≈ 18 h;
- RL-item gradients: ≈ 15 h after skipping zero-advantage groups;
- norm subsample: ≈ 4 h.

Total about 130 GPU-h. Storage: about 6 GB of scores, RL-item store ≤ 50 GB, and about 0.5 TB of working disk.

**Open verification:** whether Bergson accepts a pre-tokenized dataset with a `labels` column. If it does, fork-token masking becomes one row per trace, which removes the span expansion.

## WP7 Analysis (pre-registered)

Freeze [prereg/analysis_plan.md](../prereg/analysis_plan.md) and record its SHA-256 in the README before WP6 scoring begins.

**Primary metrics (BH at q = 0.05 over partitions × K × these three):**
1. ΔRL-item share = share(U_deep) − share(matched hard-R): paired bootstrap CI, stratified permutation p.
2. H3 statistic Δ_U − max(Δ_R, Δ_rand) from `coinfluence.drift_test`, plus `likelihood_shift_control`.
3. Proximity difference (n-gram and embedding) between U_deep and matched hard-R, with its distribution plotted against the decon threshold.

**Secondary:**
- TOST equivalence of pretraining and midtraining shares (H1, margin ±5 pp);
- top-bin Jaccard against split-half R;
- the WP4 midtraining-arm McNemar;
- RL first-reach step versus proximity (Spearman);
- per-K curves.

Anything not in the frozen plan is labelled exploratory.

## WP8 Release

- **Data:** decontaminated partition atlas with certified labels, cue and step trajectories (parquet plus HF dataset card); Tier A query sets and scores.
- **Code:** this repo; the synthetic benchmark; the Bergson GRPO-attribution tuning guide as a docs PR.
- **lm-evaluation-harness:** an RL-Zero-format math task group with unbiased pass@k over a K sweep. Check the current `repeats` and pass@k support first.
- **Outreach:** `#data-attribution` on the EleutherAI Discord and lucia@eleuther.ai. Lead with the WP2 partition figure, the Dolci pilot next to its norm-only control, and the gate design; ask only the narrow Bergson question.

---

## Remaining open verifications

1. Hub IDs and text columns for Open-Reasoner-Zero and OMEGA as used by Dolci (WP1).
2. Whether any public exact-match index covers Dolma 3 and Dolmino (HCAI-Lab bloom index, infini-gram) (WP1).
3. Licences for MATH-500, aime_2024, amc23 and minervamath (Hub metadata shows none) before release.
4. Bergson: the stored dimension for a given `projection_dim` under `projection_target=per_module`; projection-seed consistency between `build` and `score`; pre-tokenized `labels` support (WP6).
5. HCAI-Lab artifacts: licence, document-ID mapping back to Dolma 3, and whether WebOrganizer labels are included per document (WP6).
6. Whether OLMES' base-model math template differs from the RL-Zero string (WP2 template check).
7. SmolLM2-135M license and tokenizer digit handling. Single-digit tokenization matters for the cipher tasks (WP5).

## Compute summary

| Package | A100-h (planning) |
|---|---|
| WP2 signal check | 15 |
| WP3 reachability + cues | 70 |
| WP4 provenance | 60 |
| WP5 synthetic gate | 80 (small GPUs) |
| WP6 attribution, Tier A | 130 |
| **Total** | **≈ 355** |

The minimum viable path is WP2 + WP3 + a reduced WP4 (≈ 110 h). It produces the behavioural paper with no attribution at all.
