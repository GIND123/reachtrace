# Idea analysis: what survives, what breaks, and the repositioned project

*Review of "Where Do RL Reasoning Gains Come From? Provenance Attribution of RLVR on Olmo 3 (v2)" against the literature and infrastructure as of 2026-10-22. Every factual claim here was checked; see [04_verification_log.md](04_verification_log.md).*

## 1. Verdict

The core question is still open and worth doing: *which stage of an open model flow supplies the capability behind the problems RLVR newly solves?* No paper found in a search on 2026-10-22 attributes RL-Zero expansion problems to pretraining, midtraining or RL-stage data with gradient-based attribution on a fully open flow.

Three parts of v2, though, are overtaken or broken:

1. **The synthetic gate is no longer a novel test of composition.** At least four papers now show in controlled settings that RL composes existing skills. The gate is still valuable, but only as a **validation of attribution under GRPO**.
2. **H1 (elicitation) is partly answered on this exact model by behavioural means.** Forcing an opening cue raises Olmo-3-7B's MATH-500 pass@1 from 42% to 78% (Wang et al., arXiv 2610.06851). Any "U" problem must therefore be re-tested under cue-forcing before attribution means anything.
3. **Several design choices would produce confounded or uninterpretable numbers.** They are listed in §3; each now has a fix in the code.

Three verified facts make the project stronger than v2 assumed:

- **28 intermediate RL-Zero checkpoints exist** (`step_0100` … `step_2800`).
- **Three midtraining data-mix ablation checkpoints are public** (Olmo 3 report, Table 7).
- **Matlin et al. released their Dolma 3 attribution working set and preconditioner** for the same base model.

## 2. Novelty after the July–October 2026 literature

| Claim in v2 | Status now | Closest work |
|---|---|---|
| R/U/L/N partition | Instrument only (v2 already conceded this) | Invisible Leash 2507.14843; Yue et al. 2504.13837; statistical crossovers 2609.22547 |
| Synthetic f∘g gate tests composition (H3) | **Not novel as a composition result** | Yuan et al. 2509.25123 (string transforms f(g(x)), exactly this design); Abdulsalam, Patel & Saxe 2607.07646 (rewrite grammar, RL builds composed procedures, gated by how pretraining organises primitives); He et al. 2609.19465; Zhang, Neubig & Yue 2512.07783 |
| Gate as ground-truth test of *attribution* under GRPO | **Still novel.** BehaviorTrace (2610.10422) validates rollout-level attribution of a planted *behaviour*; nobody validates corpus-fragment attribution of a *composed skill* | BehaviorTrace 2610.10422 |
| U is elicitation vs acquisition | **Partly answered behaviourally on Olmo 3** | Wang et al. 2610.06851 (cue tokens recover much of the RL gain and trace back to training data); Zhang et al. 2610.24663 (~4% of token positions carry ~90% of the pass@1 gain) |
| Linking RL outcomes to pretraining text | Adjacent work exists, but it uses retrieval rather than attribution | Wang & Chen 2610.25836 (failure n-grams → retrieve pretraining text → targeted CPT; includes OLMo 2) |
| Stage-resolved gradient attribution of RL-Zero gains (pretrain vs midtrain vs RL items), U vs matched R | **Open** | Matlin et al. 2606.19625 is the template, but covers base-model capabilities only |
| Checkpoint provenance along the released flow | **Open, and better than v2 knew** (RL step branches and midtraining arms) | Olmo 3 report Fig. 25 (midtraining mix → RL-Zero dynamics, early models only) |

**Defensible contribution, one sentence.** On a fully open model flow, we decompose the problems RL-Zero newly solves into those unlocked by an opening cue, those unlocked along RL training, and the rest. We then trace the rest to pretraining, midtraining or RL-stage data with attribution that has passed a ground-truth GRPO gate and the BehaviorTrace confound controls.

## 3. Design problems found, and the fix for each

**P1. "Unreachable at K=1024" cannot be certified at the planned budget.** With zero successes in n samples, the one-sided 95% Clopper–Pearson bound certifies pass@K < 0.5 only when n > K·ln 20/ln 2 ≈ 4.32 K. That means 277 samples for K=64 and ≈4,400 per problem for K=1024. v2's adaptive sampler capped at 1024, so its large-K "U" labels would have been unfalsifiable.
*Fix:* `reach/passk.py` has three modes. **certified** is three-valued (reach / unreach / undetermined) and is used for K ≤ 64 with `n_max = 288`. **observed** is Invisible-Leash-style support and is the only mode used at K = 256 and 1024. **estimate** is the plug-in pass@K. `reachtrace certify K` prints the n required.

**P2. Difficulty matching on base pass@1 is degenerate.** Every U problem has base pass@K = 0, so coarsened exact matching on base pass@1 bins cannot pair U with R.
*Fix:* match on RL pass@1, source set and level, and RL solution length. Use **hard-R** (base solve rate in (0, 5%]) as the primary comparison group: these are the R problems nearest the boundary (`analysis/matching.py`).

**P3. Stage *levels* are confounded by format.** RL items are on-policy rollouts in exactly the query traces' format. Full-response gradients of any RL-model trace will align with RL items whatever the provenance, and stage sample sizes are set by design.
*Fix:* (a) Only U-minus-matched-R *differences* are tested. (b) The primary query target is the **fork tokens**, where log p_RL − log p_base is large under teacher forcing (`attrib/targets.py`). This follows Zhang et al. (2610.24663) and Wang et al. (2506.01939): the shared-format tokens are exactly the ones RL did not change. (c) The full-response target is kept as a secondary, and the answer-line target as BehaviorTrace's in-context check.

**P4. Bergson's chat-template wrapping corrupts RL-Zero contexts.** For prompt/completion data Bergson renders `[user, assistant]` through `tokenizer.apply_chat_template`. With the RL-Zero template that output drops the "Remember to put your answer…" line and inserts stray newlines, so every RL-item and query gradient would be taken in a context the model never trained on.
*Fix:* a local tokenizer copy with an identity template, plus pre-rendered prompts (`attrib/bergson_cmd.write_identity_tokenizer`).

**P5. H3's statistic has a mechanical explanation.** Δ_U > Δ_R could hold simply because RL moved the model most on U-like text. Correlation-matrix noise also shrinks with query count, so unequal partition sizes bias Δ.
*Fix:* equal-n subsampling, the median over subsamples as the point estimate, and a permutation null. A **likelihood-shift control** regresses each query's attribution shift on how much RL changed that trace's log-probability (`attrib/coinfluence.py`).

**P6. The zero-advantage trap in the synthetic gate.** If f∘g has pass@256 ≈ 0 for the base model, every GRPO group gets reward 0, the advantage is 0 and nothing is learned. v2's design could not train.
*Fix:* GRPO trains on short inputs at the edge of competence (rare but non-zero base success; Zhang, Neubig & Yue 2512.07783). The U-type evaluation uses longer inputs and deeper compositions (`synthetic/tasks.py`).

**P7. Decontamination references were incomplete.** The Olmo 3 report (§6.1) builds Dolci-RL-Zero-Math from DAPO-Math, Klear-Reasoner Math, **Open-Reasoner-Zero and OMEGA**; v2 scanned only the first two. Ai2 already decontaminated Dolci against pretraining and their evals, and ran a spurious-reward negative control. Our eval set is different, though (MATH L4–5, AMC23, OlympiadBench), so our own scan is still required.

**P8. The prompt wording was wrong.** The 3.1 template says "Solve the following **problem**". The report's Fig. 37 shows the 3.0 wording ("math problem"). v2's stop guard (`"\nSolve the following math problem"`) would never fire for 3.1.
*Fix:* both templates are encoded and tested byte-for-byte against the HF Jinja, and the guard is `"\nSolve the following"`.

**P9. "4 rollouts from the final model" under-represents the RL signal.** RL-Zero 3.1 ran 2,800 steps × 32 prompts × group size 8, about 717k rollouts and roughly 6.7 epochs over 13,314 prompts (Table 49; branch count).
*Fix:* now that checkpoints exist, generate rollouts at several RL steps (TracIn-CP style). Skip zero-advantage groups, which contribute no gradient in training either.

**P10. Storage was over-budgeted.** `bergson score` streams the corpus against a small query index and writes scores only: about Q × D × 4 bytes, roughly 5 GB for Tier A, instead of a 36–600 GB gradient store. Gradient norms are lost that way, so the norm-only control is built from a ~50k-document stored subsample.

**P11. The pretraining sample duplicates prior work.** The HCAI-Lab release (Matlin et al.) has Dolma 3 working samples, WebOrganizer 24×24 bins and a 100k-document preconditioner for `Olmo-3-1025-7B`. Reusing it makes the pretraining bins directly comparable with a COLM 2026 paper and saves the sampling work. Dolmino (midtraining) is not in that release and is sampled by source.

**P12. U may be tiny where it matters.** U@K shrinks as K grows. U@1 or U@4 is large but mostly boundary problems that the base reaches at K = 64, which is sharpening, not expansion. The scientifically interesting set is certified U@64 after decon and cue-forcing, and it may hold tens of problems.
*Fix:* report H1, H2′ and H3 per K with minimum detectable effects. Pre-register "underpowered" labels (MDE ≈ 0.44 σ at 80 per group). Enlarge with additional decontaminated competition sets if needed.

## 4. New opportunities from verified infrastructure

1. **RL-step reachability trajectories.** For each U problem, find the first RL step at which it becomes reachable, using `step_0100` … `step_2800`. This is a behavioural datum about when the capability arrives, and it can be correlated with proximity to Dolci prompts (H2′ without gradients).
2. **Midtraining data-mix counterfactual.** `stage2-step47684-mix-{gen-mc-only, math-code-reasoning-web, round5}-from-2T-ckpt` are three 100B-token midtraining runs from the same 2T-token stage-1 checkpoint. Base reachability of U problems under each arm, with and without cue-forcing, is an Ai2-run *data intervention*, so it is closer to causal evidence than attribution. Caveat: the arms start from 2T, not from the final 5.9T stage-1 checkpoint, and none received stage 3 or RL.
3. **Cue-forcing decomposition of U.** U splits into U_cue (reachable by the base model once given the RL model's opening tokens), U_late (reachable only at later RL steps) and U_deep (the rest). Attribution targets U_deep against matched hard-R.
4. **Bergson `contrast` queries.** `QuerySetConfig.contrast` subtracts a control set's mean gradient, so "what supports U more than matched R" can be scored in one pass. Use it as a robustness check on the per-query analysis.

## 5. Repositioned project

**Working title:** *Tracing RL-Zero gains through the Olmo 3 model flow.*

| Layer | Question | Method | Depends on attribution? |
|---|---|---|---|
| 1. Behavioural decomposition | How much of U is cue-elicitable, arrives late in RL, or is deep? | Certified partitions; cue-forcing; RL-step trajectories | No |
| 2. Data-mix counterfactual | Does midtraining reasoning data gate U reachability? | Base reachability across the three Table 7 arms and stage finals | No |
| 3. Attribution gate | Does Bergson recover planted fragment provenance for a composed skill under GRPO? | Planted fragments, leave-family-out, LDS, BehaviorTrace controls | Validates it |
| 4. Stage attribution | Does U_deep draw on different stages or bins than matched hard-R? Is joint structure new beyond drift? | Fork-token queries; streamed TrackStar/gradcos; stage share; co-influence with drift and likelihood-shift controls | Yes, gated by layer 3 |

**Revised hypotheses** (frozen in [prereg/analysis_plan.md](../prereg/analysis_plan.md)):

- **H1 elicitation:** the stage share and top-bin profile of U_deep are equivalent to matched hard-R (TOST ±5 pp), and top-bin Jaccard(U, R) ≥ the split-half Jaccard of R. *Behavioural corollary:* a majority of certified U@64 is U_cue or reachable under the `math-code-reasoning-web` arm.
- **H2′ RL-stage signal:** the RL-item share is higher for U_deep than for matched hard-R after the norm-only and fluency controls. *Gradient-free corollary:* U_deep problems sit closer to Dolci prompts (n-gram and embedding) than matched hard-R, and closer problems become reachable at earlier RL steps.
- **H3 composition:** node Jaccard for U is at least that of R, while Δ_U − max(Δ_R, Δ_rand) > 0 (bootstrap CI excludes 0, permutation p < 0.05 after BH), and the partition coefficient survives the likelihood-shift control.

**Publishable outcomes, ordered by cost:**
1. Layers 1–2 alone: an Olmo 3 RL-Zero decomposition paper (behavioural, decontaminated, certified). Workshop-ready.
2. Layer 3 failing: a negative methods result plus a Bergson GRPO-attribution tuning guide, a natural EleutherAI collaboration.
3. All layers passing: the main-track provenance paper.

## 6. Risks, ranked

| Risk | Likelihood | Mitigation |
|---|---|---|
| Format/fluency confounds dominate gradient scores (BehaviorTrace) | High | Fork-token targets, U−R differences only, norm-only and fluency controls, gate first |
| Certified U@64 is tiny after decon and cue-forcing | High | Report per K; MDE labels; enlarge the problem pool; layers 1–2 still publish |
| Document-level noise (Ruis, Matlin) | High | Bins and stages only; split-half τ ≥ 0.5 gate |
| Regenerated rollouts ≠ training rollouts | Certain | Multi-checkpoint rollouts; called an on-policy proxy throughout |
| Scooped on behavioural layer by cue/fork papers | Medium | They used Qwen or a single model pair; our layer adds certified bounds, RL trajectories and data-mix arms on a fully open flow |
| Base model rambles to max length, inflating sampling cost | Medium | Stop guard; measure in Phase 0; cap max_tokens for base at the Phase 0 truncation rate |
| Bergson API drift | Medium | Pin ≥2.3.0; all calls go through one module; smoke test |

## 7. Pitch to EleutherAI (three lines)

1. A certified, decontaminated decomposition of Olmo 3.1 RL-Zero gains into cue-elicited, late-RL and deep problems, using Ai2's step and midtraining-arm checkpoints.
2. A planted-fragment GRPO benchmark that checks whether Bergson recovers a composed skill's provenance, with BehaviorTrace controls. This yields a GRPO-attribution tuning guide (projection dim, targets, normalization) as a docs contribution.
3. Stage-level attribution of the deep set, using `--advantage_column` (v2.3.0) on Dolci rollouts and the HCAI-Lab Dolma 3 working set. Ask: a sanity check on projection and normalization defaults for 7B GRPO-gradient stores.
