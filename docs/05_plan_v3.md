# reachtrace v3: tracing Olmo 3 RL-Zero gains through the model flow

*Revised working plan, 2026-10-25. Supersedes the v2 plan and the repositioning in `docs/01_idea_analysis.md`. It responds point by point to the audit of v2 and the repo scaffold (see §10).*

## 0. What changes, in one paragraph

v2 asked gradient attribution to do three things it can't: compare stages, separate elicitation from acquisition, and detect composition. It also used a reachability label (certified U) that captures sharpening, not expansion. v3 makes two changes:

- **Behavioural measures that are exact and cheap now answer the stage question.** These are a likelihood ledger, a "discrimination margin" between correct and incorrect solutions, and an RL-to-base handoff map.
- **Gradient attribution is limited to the question it can answer:** which data *within* a stage supports the solutions RL gained, compared with matched problems.

Attribution runs only after a redesigned synthetic gate passes. It uses format-cancelling contrastive queries, and RL items are scored with the advantage applied after normalization.

## 1. Research questions

| RQ | Question | Primary instrument | Needs attribution? |
|---|---|---|---|
| RQ1 | How much real expansion is there on decontaminated, post-cutoff problems, with lenient grading and the same 32k budget for both models? | Two-phase sampling; continuous base-rate estimates | No |
| RQ2 | At which stage (pretrain, midtrain, long-context, RL step) does the model start to prefer correct over incorrect solutions for the problems RL gained? | Likelihood ledger and discrimination margin across about 20 checkpoints | No |
| RQ3 | How much of a correct RL solution must the base model be given before it can finish it? Where are the critical segments? Is there more than one? | Handoff map (prefix-conditioned base sampling) | No |
| RQ4 | Within the stage that RQ2 points to, which sources or bins support the critical content of gained problems more than matched problems? For the RL stage: Dolci items versus placebo items. | Gated, contrastive Bergson attribution, scored at that stage's checkpoint | Yes |
| RQ5 (optional) | Does removing the Dolci prompt clusters that RQ4 credits delay when gained problems first become reachable? | Short RL ablation runs | No (it's causal) |

**Positioning.** Wang et al. (arXiv 2610.06851) showed that *opening cues* elicit reasoning in Olmo-3-7B. Kou et al. (arXiv 2610.30882) showed *when the "wait" marker arrives* during Olmo 3 training. This project goes per problem: for the specific problems RL gains, where does the *preference for correct content* come from, and where along the solution does the base model get stuck? Both papers are cited as the starting point, not competitors.

## 2. Hypotheses as measurable signatures

**G (gained set):**
- Base solves 0 of n ≥ 256 samples at the 32k cap, under both strict and lenient grading.
- The RL model's pass@1 is at least 0.10.
- The problem is decontamination-clean against Dolci.

**B (boundary comparison set):**
- Base rate in (0, 1/64].
- Matched to G on RL pass@1 bin and source set.

Problems across all base-rate deciles are also sampled, for trend models.

| Hypothesis | Ledger / margin (RQ2) | Handoff (RQ3) | Within-stage attribution (RQ4) |
|---|---|---|---|
| **H1 elicitation** | Stage-3 base already prefers correct over incorrect on content tokens (margin M > 0), on trend with B | Base completes the solution from a short prefix: handoff h at or below cue length for most of G | No excess of Dolci over placebo for G compared with B |
| **H2′ RL-stage acquisition** | M ≈ 0 before RL; most of the margin gain happens during RL, beyond the B trend | h large: the base needs much of the RL trace | Dolci excess over placebo is larger for G than for B; proximity predicts the RL step at which a problem is first reached |
| **H3 composition** | Ingredients are preferred before RL; the joint solution is not | At least 2 separated critical segments, each one base-completable given the correct prefix, while end-to-end success ≈ 0 | Not tested at 7B |

These signatures can co-exist. The paper reports what fraction of G fits each one.

## 3. Instruments

### 3.1 Evaluation set (fixes small U and contamination)

- **Date-clean stratum (post-cutoff for every Olmo 3 stage):** AIME 2026, HMMT Feb 2026 (`MathArena/aime_2026`, `MathArena/hmmt_feb_2026`).
- **2025 stratum:** AIME 2025, HMMT Feb 2025, BRUMO 2025, CMIMC 2025 (`MathArena/*`).
- Together about 190 problems. Licence is CC-BY-NC-SA, so the release terms for these rows follow it.
- **Core:**
  - MATH-500.
  - OlympiadBench (English, text-only, single answer, about 600).
  - An Omni-MATH subset with numeric answers (`KbsdJames/Omni-MATH`, about 800 of 4,428, Apache-2.0). This is the main source of G.
  - AMC 2023 and AIME 2024.
- **Pilot pool, never confirmatory:** 200 MATH-train L5 problems plus AIME 2022–23. All tuning, cue selection and pipeline checks happen here.

**Decontamination policy:**
- **Drop** any problem that matches a Dolci item, i.e. the 13.3k prompts actually trained on. A match is a shared 13-gram or embedding cosine ≥ 0.92, confirmed by hand.
- **Tag but keep** matches to upstream pools that were never trained on, to Dolmino or Longmino, or to the Dolma sample.
- Report results on strict, RL-clean and date-clean subsets separately.

### 3.2 Sampling and grading (fixes spurious U)

Both models get the same prompt string, temperature 1.0 and token cap. All sampling uses a **cap of 32k**, Ai2's evaluation budget, with one exception: Phase A is a screen at 16k for both models, and no labels are taken from it.

**Grader:**
1. Cut the response at the first marker of a new problem (`\nProblem`, `\nSolve the following`, `\nQuestion:`, or the prompt's reminder line repeated).
2. Take the last `Answer:` before that cut, falling back to the last `\boxed{}`.
3. As a robustness check only, a lenient "any final answer" grade. A problem counts as G only if it is G under both strict and lenient grading.

A 300-sample manual audit (half base, half RL) must show extraction error below 2%. Format-failure rates are reported for each model.

**Two-phase sampling:**
- **Phase A:** all problems, base n=64 and RL n=32, at 16k.
- **Phase B:**
  - G candidates: base escalated to n=256 at 32k.
  - About 250 comparison problems spread across base-rate deciles: n=64 at 32k.
  - RL: n=64 at 32k for all of them.

Base rates use Jeffreys posteriors. Reachability is a continuous covariate, logit of the base rate.
- "Observed support at k ≤ 256" is used in the Invisible Leash sense, and only for the RQ1 replication.
- Certified labels survive only as a descriptive table at K=64.
- Rates come from the fixed first-64 samples, which removes the stopping-rule bias.

**Cue arm:** the cue is passed as token IDs, not appended as a string. We use Wang et al.'s `.\n\n Okay`, re-checked on the pilot pool. The cue arm gets the same sample budget as the no-cue arm (n=64 each), so U_cue is not inflated by extra draws.

`main` and `step_2800` have identical weights; only `main` is sampled.

### 3.3 Likelihood ledger and discrimination margin (RQ2; the core new instrument)

**Sequences** for each problem in G ∪ B:
- 4 correct and 4 incorrect finished RL traces, matched in length within ±20%.
- The human reference solution, where one exists (MATH, OlympiadBench, AoPS).

**Checkpoints (about 20, all pinned):**
- stage1 at roughly 25%, 50% and 75% of steps, plus `stage1-step1413814`.
- `stage1-step476838`, about 2T tokens, which is where the midtraining arms start (to verify).
- stage2 at 25%, 50% and 75%, plus `stage2-step47684`.
- The 3 midtraining arms (`stage2-step47684-mix-{round5, gen-mc-only, math-code-reasoning-web}-from-2T-ckpt`).
- stage3 final (= `main`).
- RL steps 100, 200, 400, 800, 1600 and 2800.

**Token masks:** opening (the first 16 tokens), answer line, critical segments (from §3.4), and content (everything else).

**Measures** (teacher-forced, forward pass only):
- L_θ is the per-token log-likelihood on each mask.
- **Margin** M_θ(q) = mean L_θ(correct) − mean L_θ(incorrect), on content tokens.
- **Ledger:** the stage increments ΔM_s add up exactly to the total change from early stage 1 to the final RL model. This gives a **stage decomposition that is exact by construction**, with no attribution approximations.

Why this fixes the earlier problems:
- Correct-minus-incorrect pairs share the same generator, format and style, so form cancels. This is the behavioural version of the "Form over Content" control (arXiv 2609.19589).
- Teacher forcing removes the format floor at stage 1, where accuracy is 0.1% (arXiv 2610.30882).
- The opening mask separates the cue effect (Wang et al.) from the content.

**Midtraining arms:** M under each arm minus M at the shared 2T start checkpoint. This is a real data intervention, measured on a continuous scale, so it avoids floor effects. The caveat is that none of the arms received stage 3.

### 3.4 Handoff map (RQ3; fixes fork tokens and H3)

For 2 correct RL traces per problem in G ∪ B:
- Sample base continuations from RL prefixes cut at fractions {0, cue only, 0.1, 0.25, 0.5, 0.75}, 16 samples at each point.
- Refine with 2 more points inside the interval with the largest jump.

From this:
- **Handoff h:** the smallest prefix fraction at which base success is at least 0.25.
- **Critical segments:** intervals where base success jumps by at least 0.25.

These segments are identified causally, from behaviour. They replace the log-ratio "fork tokens", and they are the RQ4 query targets.

**Composition signature (H3):**
- At least 2 separated critical segments.
- The base finishes each segment given the correct preceding prefix.
- The base succeeds about 0% end to end.

## 4. Synthetic gate v2 (validates both the ledger and the attribution)

Use SmolLM2-135M (and 360M as a replica) and a pipeline that mirrors Olmo's:

1. **Pretrain (CPT):** generic text plus fragment families:
   - A and B are the causal operations.
   - C and D are look-alike distractors.
   - **F is format-only:** documents in the exact answer format with irrelevant content.
2. **Midtrain:** a planted "worked-trace" family with a known role.
3. **GRPO** at the edge of competence (training lengths 3–4), evaluated on longer and deeper compositions. Every rollout's text is logged.

Ground truth comes from leave-family-out runs and an LDS **dose design**: 40 subsets, each family kept at a fraction in {0, .25, .5, .75, 1}, 2 GRPO seeds per subset, averaged.

**Name control:** operations are named with per-document random aliases, and the task prompt uses a held-out alias plus one in-context example. A **named** condition is reported too, but gate decisions use the alias condition.

| Gate | Pass criterion |
|---|---|
| G0 ledger | Ledger and margin put A/B onset in CPT and composition onset in GRPO, on 3/3 seeds |
| G1 recovery | min(A,B) > max(C,D,F) on 3/3 seeds; recall@k beats **BM25** and embedding-similarity baselines |
| G2 controls | Beats norm-only and fluency-residualized rankings (95% CI excludes 0 on ≥2/3 seeds) |
| G3 format | With contrastive queries, F ranks below A/B; non-contrastive results are shown to document the confound |
| G4 RL-item scoring | Advantage-after-normalization scoring gives task items more mass than placebo items, with the right sign; Bergson's built-in sign-only scoring is shown for comparison |
| G5 LDS | Lower 95% CI bound > 0 and point estimate ≥ 0.4 (n=40, SE ≈ 0.16) |
| G6 stability | Kendall τ ≥ 0.6 across generation draws |

**Decisions:**
- G0 fails: the ledger claims are qualified.
- G1–G4 fail: no 7B attribution. Write the methods paper instead.
- G5 alone fails: 7B attribution is reported as rankings only.

## 5. Within-stage attribution at 7B (RQ4, gated)

**Principle:** each stage's documents are scored at **that stage's own end checkpoint**, with query gradients taken at the same checkpoint. Comparisons are made only within a stage: G versus B, and Dolci versus placebo. Stage-against-stage questions are answered by the ledger, never by gradient mass.

| Stage | Docs (Tier A) | Bins | Scored at |
|---|---|---|---|
| Pretrain (Dolma 3) | 120k × 1,024 tokens: FineMath 20k, arXiv 15k, Stack-Edu 15k, Wikipedia 5k, plus 65k from the HCAI WebOrganizer-binned set (math/science/education topics plus a random control) | source × WebOrganizer bin | stage1 final |
| Midtrain (Dolmino 7B mix) | 80k × 1,024, balanced over its 24 sources (Olmo 3 report Table 5), math-heavy | source | stage2 final |
| Long-context (Longmino) | 20k × 2,048 | source | stage3 final |
| RL (Dolci) | 2,000 prompts stratified by upstream source and semantic cluster, G=4 rollouts at steps 200, 800 and 2800 | upstream source / cluster | the same RL step |
| RL placebo | 700 held-out math prompts: not in Dolci, embedding-distant from it, difficulty-matched; G=4 at the same steps | — | the same RL step |

Bins are sampled balanced (Matlin-style), with population-token-weighted results reported as secondary. The RL stage uses a constant learning rate of 1e-6, so summing scores across its checkpoints is valid TracIn-CP.

**Queries:** pre-tokenized `input_ids` plus `labels` masks, one row per trace. This removes the span explosion and the token-by-token decoding corruption. (Bergson skips tokenization when `input_ids` is present, `worker_utils.py:426`; that label masks are honoured still needs confirming.)
- **Primary target:** content tokens, contrastive. Score = cos(q_correct, d) − cos(q_incorrect, d), averaged over 4 correct and 4 incorrect traces.
- **Secondary target:** critical segments from §3.4.

**RL-item scoring fix:** Bergson multiplies each loss by its advantage and then `--unit_normalize` divides each item by its norm (`process_grads.py:159`), so only the advantage's sign survives. Instead:
- Build the RL-item index **without** advantages, using plain completion gradients, unit-normalized.
- Store advantages in a sidecar file.
- Score each group as s(q, g) = Σᵢ Aᵢ · cos(q, gᵢ), which keeps the zero-sum structure.
- Positive and negative parts are reported separately.

**Bergson settings:** projection p=16 (Matlin-comparable, about 256 values per module); the gate sweeps p ∈ {16, 32}. Controls on every metric: norm-only (from a stored 50k subsample), fluency residualization, a random-query baseline, and between-draw variance.

**Gate before analysis:** split-half τ ≥ 0.5 on bin rankings, and contrastive targeted scores beat norm-only on G ∪ B. If this fails, report a limits result.

## 6. Optional causal arm (RQ5, partner compute only)

- Use the RL-step trajectories to select G problems that first become reachable by step 400.
- Run 400-step OlmoRL from `main`, under three conditions:
  - remove the Dolci clusters RQ4 ranks highest for those problems;
  - remove random clusters of the same size;
  - remove nothing.
- 2 seeds each.
- Outcome: the RL step at which each problem is first reached, and its likelihood margin.
- Cost: about 1,000–2,000 GPU-h. Pitch this to EleutherAI or Ai2 only after the gate and RQ2 results are in hand.

## 7. Statistics and pre-registration

**Freeze point:** after Phase A and all pilot-pool work, and *before* any ledger, handoff or attribution run on confirmatory problems. Post it on OSF with the hash in the README. The v2 Dolci pilot moves to the pilot pool.

**Primary tests**, Benjamini–Hochberg at q=0.05 over these 3:
1. M_stage3(G) > 0, and G's residual from the margin trend fitted on B and the decile sample.
2. The fraction of G with handoff h ≤ 0.1, compared with B.
3. The RL-stage share of the margin gain, G versus B.

**Analysis method:**
- Problem-level paired bootstrap (10k resamples) and stratified permutation tests.
- Trend models: mixed models on logit of the base rate, with source fixed effects and RL pass@1 as a covariate.
- Equivalence margins (smallest effect size of interest) are set to 0.3 SD of the comparison group, calibrated on the pilot pool, not chosen arbitrarily.

**Power:**
- **Gate:** if G has fewer than 60 problems after decontamination, add more Omni-MATH problems before the freeze.
- Expected G is about 100–250, given Wang et al.'s pass@16 numbers and Omni-MATH's difficulty.
- The ledger and handoff are paired within problem, so power is far higher than for between-group gradient mass.

**Secondary:** within-stage attribution contrasts, proximity versus first-reach step (Spearman), midtraining-arm margins, the composition-signature distribution, and per-K partition tables.

**Dropped:** the co-influence matrix and Δ statistic, gradient "stage shares", and the U_late/U_deep split.

## 8. Phases and gates (about 20 weeks)

| Weeks | Phase | Gate |
|---|---|---|
| 1–2 | **P0:** pin SHAs; benchmark vLLM at 16k/32k (tokens/s, KV savings from sliding-window layers) and Bergson at p=16 (tokens/s, stored dimension, label masks respected, projection seed); grader audit; anchor check (AIME pass@1 ≈ 50% at 32k); pilot runs | Anchor within ±5 points; grader error < 2%; budget re-derived from measured throughput |
| 2–4 | **P1:** eval set, decontamination, Phase A | At least 1,000 clean problems |
| 3–8 | **P2:** synthetic gate v2 (parallel, small GPUs) | G0–G6 |
| 4–6 | **P3:** Phase B; **pre-registration freeze** | G has at least 60 problems |
| 6–10 | **P4:** ledger, margins, handoff, cue arm, midtraining arms, RL-step trajectories | — (this is the core paper) |
| 10–15 | **P5:** RQ4 attribution, if G1–G4 passed and compute is secured | Split-half τ ≥ 0.5; beats norm-only |
| 12–16 | **P6:** RQ5 ablation (optional, partner compute) | — |
| 15–20 | **P7:** write-up and release (atlas, ledger data, gate benchmark, Bergson RL-scoring note) | — |

## 9. Budget (planning numbers; P0 replaces them)

These assume about 1.5k decode tokens/s per A100 (Olmo 3 7B has 32 KV heads, so no GQA) and 500 attribution tokens/s per A100. If attribution runs at Matlin et al.'s measured ~150 tokens/s (20,000 GPU-h for 10.5B tokens), it costs about 3×.

| Item | A100-h |
|---|---|
| Phase A screen (about 1,500 problems) | ~110 |
| Phase B (32k cap, escalation) | ~125 |
| Handoff map | ~55 |
| Cue arm (budget-matched) | ~25 |
| Ledger (about 20 checkpoints, forward only) | ~15 |
| Checkpoint pass@k on G (stage2, arms, 3 RL steps) | ~50 |
| **Behavioural paper subtotal** | **~380** |
| Synthetic gate v2 | ~150 |
| RQ4 attribution (about 530M gradient tokens, plus rollouts and norms) | ~350 (range 180–1,100) |
| **Full Tier A** | **~880** |
| RQ5 ablation (optional) | 1,000–2,000 |

On H100s these numbers roughly halve. The behavioural paper alone fits the original ≤500 GPU-h envelope; the full plan does not.

If P0 measures attribution below 400 tokens/s, shrink the corpus proportionally, recompute the MDE, or seek NAIRR or EleutherAI compute.

**Storage:**
- At p=16 and ~224 modules, about 115 KB per item. The RL-item store and norm subsample come to about 10 GB.
- Scores: about 5 GB.
- Checkpoints: about 300 GB.

## 10. Audit problem → fix

| Audit finding | Fix in v3 |
|---|---|
| Budget about 10× too small | Throughput benchmark in P0; ~880 A100-h stated plainly; behavioural-only option |
| Certified U is sharpening, not expansion; hard-R empty at K ≤ 16 | Continuous base rate; G defined at n ≥ 256 and 32k; certified labels only descriptive |
| H2′ also predicted by elicitation | Margin and handoff measured against the B trend; within-RL comparison uses Dolci versus placebo |
| Behavioural layer already published | Repositioned onto per-problem margin, ledger and handoff, built on top of Wang et al. and Kou et al. |
| Stage shares not identifiable; Longmino and FineMath/arXiv/Stack-Edu missing | Exact ledger for stages; attribution only within a stage, at that stage's checkpoint; corpus includes all of them |
| Form over content | Correct-minus-incorrect contrast in both ledger and attribution; format family F in the gate |
| Unit-norm erases the advantage | Advantage applied after normalization, in post-hoc scoring; checked by gate G4 |
| No placebo for RL items | Held-out-prompt placebo rollouts |
| H3 metric doesn't measure composition; N has no RL traces | Co-influence dropped; behavioural composition signature from the handoff map |
| Gate passes on lexical cues; weak LDS; seed noise | Random aliases, BM25 baseline, dose-design LDS with 40 subsets × 2 seeds, ledger validation (G0) |
| Fork tokens are opening tokens with noisy gradients | Critical segments found behaviourally; contrastive content tokens; pre-tokenized masks |
| Truncation, stop guard, last-`Answer:`, cue tokenization, cue sample budget | Same 32k cap; cut at new-problem markers; lenient robustness grade; token-ID cue; budget-matched cue arm |
| Stage-1 floor; arms lack stage 3 | Teacher-forced margins instead of pass@k; arms compared with their own 2T start |
| Pilot peeks at the primary test; TOST margin; stopping bias; upstream over-dropping | Pilot pool kept separate; freeze before confirmatory runs; calibrated margins; first-64 rates; tag upstream matches instead of dropping |
| `main` = `step_2800` | Sample `main` only |

## 11. Code changes for later

- **`reach/`:** continuous rates, two-phase sampling, observed-support mode.
- **`sample/grade.py`:** cut at new-problem markers, lenient grade, audit tooling.
- **`prompts.py`:** token-ID cue prefill.
- **New `ledger/`:** teacher-forced log-likelihoods, masks, margins, stage decomposition.
- **New `handoff/`:** prefix sampling, critical segments.
- **`attrib/`:**
  - pre-tokenized label masks;
  - contrastive scoring;
  - RL-item post-hoc advantage scoring;
  - placebo items;
  - remove `coinfluence.py` from the confirmatory path.
- **`data/corpus_sample.py`:** source bins, including FineMath, arXiv, Stack-Edu and Longmino.
- **`synthetic/`:** aliases, F and midtrain families, dose-design LDS, BM25 baseline, ledger check.
- **`prereg/`:** rewrite around the 3 new primary tests.

## 12. Week-1 checks still open

1. vLLM KV memory with Olmo 3's sliding-window layers, and real decode tokens/s at 32k.
2. Bergson: label masks honoured for pre-tokenized rows (code suggests yes), stored dimension at p=16, projection seed reuse across checkpoints, actual tokens/s.
3. Which stage-1 checkpoint the midtraining arms started from (`stage1-step476838` is an estimate for 2T; confirm on the model card).
4. Longmino source composition.
5. The numeric-answer subset of Omni-MATH and its grading.
6. Hub IDs for the Open-Reasoner-Zero and OMEGA sources (for decontamination and choosing placebo prompts).
7. Whether OLMES's base template differs from the RL-Zero template.

## 13. Open decisions

1. **Compute:** behavioural paper plus gate (~530 A100-h), or the full Tier A (~880 A100-h)?
2. **RQ5:** pursue the RL ablation with EleutherAI or Ai2 compute?
3. **Visibility:** keep the GitHub repo public while the idea has no results yet?
4. **Repo:** add this plan as `docs/05_plan_v3.md` and rewrite `prereg/analysis_plan.md` to match.

## Key sources

- Wang et al., *Base Models Can Reason By Taking a Cue From Training Data*, [arXiv 2610.06851](https://arxiv.org/abs/2610.06851)
- Kou et al., *Inherited, Not Invented: Tracing the Provenance of Wait*, [arXiv 2610.30882](https://arxiv.org/abs/2610.30882)
- Matlin et al., *Capability Provenance in Language Models*, [arXiv 2606.19625](https://arxiv.org/abs/2606.19625)
- Nautiyal, *BehaviorTrace*, [arXiv 2610.10422](https://arxiv.org/abs/2610.10422)
- Kim et al., *Form Over Content in Gradient-Based Data Attribution*, [arXiv 2609.19589](https://arxiv.org/abs/2609.19589)
- Liu et al., *RL-IF*, [arXiv 2610.28883](https://arxiv.org/abs/2610.28883)
- Wu et al., *The Invisible Leash*, [arXiv 2507.14843](https://arxiv.org/abs/2507.14843)
- Team Olmo, *Olmo 3*, [arXiv 2512.13961](https://arxiv.org/abs/2512.13961)
- [EleutherAI/bergson](https://github.com/EleutherAI/bergson) (source checked at commit d8478a1)
- [allenai/Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B), [allenai/Olmo-3.1-7B-RL-Zero-Math](https://huggingface.co/allenai/Olmo-3.1-7B-RL-Zero-Math), [allenai/Dolci-RL-Zero-Math-7B](https://huggingface.co/datasets/allenai/Dolci-RL-Zero-Math-7B)
