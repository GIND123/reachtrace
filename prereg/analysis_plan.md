# Pre-registered analysis plan (v3 template: freeze after Phase A, before confirmatory runs)

> Status: **DRAFT**. Freeze after Phase A sampling and all pilot-pool work, and *before* any ledger, handoff or attribution run on confirmatory problems. On freezing, fill in every `<...>`, commit, and record `sha256sum prereg/analysis_plan.md` in the README with the date. Later edits go in a dated amendment section and do not change the frozen text. The design is in [docs/05_plan_v3.md](../docs/05_plan_v3.md).

## 1. Models and data (pinned)

- Base: `allenai/Olmo-3-1025-7B@<sha>`. RL: `allenai/Olmo-3.1-7B-RL-Zero-Math@<sha>` (`main`; identical weights to `step_2800`). Lock file: `configs/pins.lock.yaml@<git sha>`.
- Prompt: template `rlzero31_math`, rendered identically for both models and passed as token IDs. Stop guard `"\nSolve the following"`; responses are also cut at the first new-problem marker before grading (`sample/grade.py:truncate_new_problem`).
- Decoding: T = 1.0, top-p = 1.0, max_tokens = 32,768 for labels (Phase B), 16,384 for the Phase A screen only.
- Eval set: `artifacts/evals/clean.parquet@<hash>`, with strict, RL-clean and date-clean (2026 competitions) subsets.
- Pilot pool (never confirmatory): `<list of problem ids>`.

## 2. Problem sets

- **G (gained):** base 0 correct out of n ≥ 256 samples at the 32k cap, under both strict and lenient grading; RL pass@1 ≥ 0.10; not a near-duplicate of any Dolci item.
- **B (boundary):** base rate in (0, 1/64] estimated from each problem's first 64 samples; matched 1:1 to G on RL pass@1 bin × source set, seed `<seed>`.
- **D (decile sample):** `<n>` problems stratified over Phase A base-rate deciles, for trend models.
- Exclusions: Dolci near-duplicates (13-gram or embedding cosine ≥ 0.92, confirmed by hand); problems whose reference answer the grader cannot parse (audited list `<path>`).

## 3. Instruments (settings fixed here)

- Ledger: checkpoints `<list with SHAs>`; 4 correct and 4 incorrect finished RL traces per problem, length-matched within ±20%; masks: opening = first 16 tokens, answer line, critical segments, content = remainder. Traces must fit the 8,192-token context of stage-1/2 checkpoints, or the ledger is restricted to checkpoints that support their length.
- Margin: M_θ(q) = mean per-token log-likelihood of correct minus incorrect traces on content tokens.
- Handoff: prefix fractions {0, cue, 0.1, 0.25, 0.5, 0.75} of the content region plus 2 refinement points; 16 base samples each; h = smallest fraction with base success ≥ 0.25; critical segment = interval with a success jump ≥ 0.25.
- Cue: the RL model's modal opening `<k>` tokens on its correct traces, passed as token IDs, budget-matched (n = 64).

## 4. Primary tests

Benjamini–Hochberg at q = 0.05 over P1–P3.

| ID | Hypothesis | Statistic | Test | Direction (H1 / H2′) |
|---|---|---|---|---|
| P1 | Pre-RL preference for correct content | M_stage3(G); residual of G from the margin trend fitted on B ∪ D | problem-level bootstrap (10k); stratified permutation (10k) | H1: M > 0 and on trend; H2′: M ≈ 0 / below trend |
| P2 | Base completes RL solutions from short prefixes | fraction of G with h ≤ 0.1, G vs B | paired bootstrap | H1: high; H2′: low |
| P3 | Where the margin is gained | RL-stage share of the total margin gain, G vs B | paired bootstrap; permutation | H1: ≤ B; H2′: > B |

Equivalence margins (SESOI) are 0.3 SD of the B distribution of each statistic, calibrated on the pilot pool: `<values>`.

## 5. Secondary analyses (not BH-corrected with the primaries)

- H3 composition signature: share of G with ≥ 2 separated critical segments, each base-completable given the correct prefix, with end-to-end base success ≈ 0.
- Midtraining arms: M under each arm minus M at `stage1-step476838` (2T), `math-code-reasoning-web` vs `gen-mc-only`.
- RL first-reach step vs Dolci proximity (Spearman).
- Within-stage attribution (only if synthetic gates G1–G4 passed): contrastive content-token scores, G vs B, per source bin; Dolci minus placebo for the RL stage, with advantage applied after normalization.
- RQ1 replication: observed support at k ≤ 256; certified labels at K = 64 as a descriptive table.

## 6. Controls (reported for every attribution metric)

Norm-only profile; fluency residualised within strata; random-query baseline; between-draw variance; placebo RL items; BM25 baseline (synthetic gate).

## 7. Power

Gate: if |G| < 60 after decontamination, add Omni-MATH problems before freezing. Report the MDE for every comparison; cells with fewer than 40 matched pairs are labelled "underpowered" and are descriptive only.

## 8. Gates that stop the analysis

- Synthetic gate G1–G4 fails: no 7B attribution claims. G0 fails: ledger claims are qualified.
- 7B split-half Kendall τ of bin rankings < 0.5, or contrastive targeted ≤ norm-only on G ∪ B: report attribution as a limits result only.

## 9. Exploratory

Anything not listed above, including the co-influence analysis (`attrib/coinfluence.py`), per-bin narratives, document-level examples and OLMoTrace span inspection.
