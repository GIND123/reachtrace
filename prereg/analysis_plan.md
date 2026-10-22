# Pre-registered analysis plan (template: freeze before WP6 scoring)

> Status: **DRAFT**. On freezing, fill in every `<...>`, commit, and record `sha256sum prereg/analysis_plan.md` in the README with the date. Later edits go in a dated amendment section and do not change the frozen text.

## 1. Models and data (pinned)

- Base: `allenai/Olmo-3-1025-7B@<sha>`. RL: `allenai/Olmo-3.1-7B-RL-Zero-Math@<sha>`. Lock file: `configs/pins.lock.yaml@<git sha>`.
- Prompt: template `rlzero31_math`, rendered identically for both models. Stop guard `"\nSolve the following"`.
- Decoding: T = 1.0, top-p = 1.0, max_tokens = `<16384 | Phase-0 cap for base>`.
- Eval set: `artifacts/evals/clean.parquet@<hash>`, strict and RL-clean subsets.

## 2. Partitions

- Certified mode at K ∈ {1, 4, 16, 64}: γ = 0.5, one-sided α = 0.05 Clopper–Pearson, n_max = 288. Observed mode only at K ∈ {256, 1024}.
- Hard-R: R@K with base solve rate in (0, 0.05].
- Cue decomposition: U_cue if the base is reachable at K under any pre-specified cue `<list>`. U_late if reachable only at RL step ≥ `<step>`. U_deep is the remainder.
- Exclusions: problems with RL-stage decon flags; problems whose reference answer the grader cannot parse (audited list `<path>`).

## 3. Matching

- Coarsened exact matching of U_deep to hard-R on RL pass@1 quintile × source set × MATH level (or "none") × RL correct-solution length tercile.
- Pairs are drawn 1:1 within strata, seed `<seed>`. Report unmatched counts.

## 4. Attribution settings (from the synthetic gate)

- Method `<trackstar | gradcos | ekfac>`, projection_dim `<d>`, unit-norm `<yes/no>`, preconditioner `<path or none>`.
- Target: fork tokens with log-ratio > ln 4 and merge gap 1 (primary); full response and answer line (secondary).
- Draws: 4 correct traces per problem; metrics averaged over draws, with between-draw variance reported.

## 5. Hypotheses and primary tests

BH at q = 0.05 over {partition-K cells} × {P1, P2, P3}.

| ID | Hypothesis | Statistic | Test | Direction |
|---|---|---|---|---|
| P1 | H2′: U_deep relies more on the RL-stage signal | RL-item stage share, U_deep − matched hard-R | paired bootstrap 95% CI (10k) and stratified permutation (10k) | > 0 |
| P2 | H3: new joint structure beyond drift | Δ_U − max(Δ_R, Δ_rand), equal-n, top-50 bins | bootstrap CI (500) and permutation (500); likelihood-shift coefficient | > 0 and coefficient > 0 |
| P3 | H2′ (gradient-free): U_deep is closer to Dolci prompts | max 8-gram overlap and max embedding cosine (`<model>`), U_deep − hard-R | paired bootstrap and permutation | > 0 |

## 6. Secondary tests (reported, not BH-corrected with the primaries)

- H1 equivalence: pretraining and midtraining shares, U_deep vs hard-R, TOST margin ±5 percentage points.
- Top-50 bin Jaccard(U, R) vs split-half Jaccard(R, R).
- Midtraining arms: McNemar on certified reachability, `math-code-reasoning-web` vs `gen-mc-only`, over U ∪ hard-R.
- Spearman correlation of RL first-reach step with P3 proximity.

## 7. Controls (reported for every metric)

Norm-only profile removed; fluency (mean token log-prob) residualised within difficulty strata; random-query baseline; multi-draw variance; model-swap drift on R, N and random queries.

## 8. Power

Report the MDE in SD units for each comparison (`analysis.stats.mde`). Any cell with fewer than 80 matched pairs is labelled "underpowered", and its result is descriptive only.

## 9. Gates that stop the analysis

- Synthetic gate G1 or G2 fails: no 7B attribution claims.
- 7B split-half Kendall τ of bin rankings < 0.5, or targeted ≤ norm-only on U ∪ R: report as a limits result only.

## 10. Exploratory

Anything not listed above, including per-bin narratives, document-level examples and OLMoTrace span inspection.
