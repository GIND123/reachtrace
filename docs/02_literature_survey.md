# Literature survey

*Verified 2026-10-22. Every arXiv ID below was resolved through the arXiv API (title, authors, date). Claims marked ‡ come from the paper body (Olmo 3 report via full text, Bergson via its source and README); all other summaries are limited to what each paper's abstract states. Venues are given only where verified.*

## 1. Does RLVR expand what a model can do?

**The pass@k debate.** Yue et al. ([2504.13837](https://arxiv.org/abs/2504.13837)) report that RLVR models win at small k while base models achieve higher pass@k at large k, and that distillation, unlike RLVR, can add new reasoning patterns. ProRL (Liu et al., [2505.24864](https://arxiv.org/abs/2505.24864)) reports the opposite under prolonged RL with KL control and reference resets: RL-trained models outperform base models across pass@k, including on tasks the base model fails at any sampling budget.

**Empirical support.** Wu et al., *The Invisible Leash* ([2507.14843](https://arxiv.org/abs/2507.14843); Fang Wu, Weihao Xuan, Danny Xing, Ximing Lu, Mingjie Liu, Yi Dong, Zaid Harchaoui, Yejin Choi) introduce *empirical support dynamics*. They match verified successes question by question into preservation, expansion and shrinkage, and find that "the shrinkage of observed support outweighs its expansion" at the stated sampling budgets. This is the R/U/L/N instrument used here. (The ProRL-1.5B-v2 counts quoted in our v2 plan come from an earlier version; recheck them against v5 before citing.)

**Statistics of crossovers.** Yang & Chen ([2609.22547](https://arxiv.org/abs/2609.22547)) argue that a visible pass@k crossover need not be statistically established. They build confidence bands across k, find no crossing established in initial evaluations of five public RLVR pairs, and show that base success rate alone does not determine post-RL success: the mapping is a Markov kernel. Their power analysis says more prompts help more than more samples per prompt. Zhou ([2607.20543](https://arxiv.org/abs/2607.20543)) locates pass@k inversion on boundary prompts whose rare correct trajectories are too sparse to appear in finite GRPO groups. Rath et al. ([2610.07405](https://arxiv.org/abs/2610.07405)) show pass@k misses diversity changes that GRPO and RFT move in opposite directions.

**Mechanistic readings of the gain.**
- Cheng et al. ([2610.04158](https://arxiv.org/abs/2610.04158)) model RL as reshaping *strategy selection*. This can broaden or narrow coverage at a budget without expanding strategy support.
- Krishnamurthy, Huang & Rajaraman ([2606.13125](https://arxiv.org/abs/2606.13125)) separate strategy selection from strategy improvement on Qwen-2.5-1.5B.
- Sun et al. ([2609.01274](https://arxiv.org/abs/2609.01274)) find that much of the RL gain on SimpleRL-Zoo pairs matches base-model operating points reachable under search.
- Noukhovitch et al. ([2609.13443](https://arxiv.org/abs/2609.13443), Mila/Ai2) document a "Matthew effect": RL improves easy problems far more than hard ones.

**Theory.** The sharpening view (Huang et al., [2412.01951](https://arxiv.org/abs/2412.01951)) and the coverage principle (Chen et al., [2510.15020](https://arxiv.org/abs/2510.15020)) both formalise post-training as limited by the mass the pretrained model already places on good responses.

**Implication for this project.** "Unreachable" must be a certified statistical statement, and partitions must be reported as a function of K. Problem count, not samples per problem, sets the power of U-versus-R comparisons.

## 2. What RL actually changes

**Few tokens.**
- Zhang et al. ([2610.24663](https://arxiv.org/abs/2610.24663)), on Qwen3-4B-Base: borrowing the RL model's token only where the two models' greedy choices differ, about 4% of positions, recovers 90.5% of the pass@1 gain on AIME. Forcing the opening "Alright" alone recovers 35%.
- Wang et al. ([2506.01939](https://arxiv.org/abs/2506.01939)): high-entropy "forking" tokens drive effective RLVR.
- Zhu et al. ([2506.01347](https://arxiv.org/abs/2506.01347)): training on negative samples alone preserves pass@k.

**Cues learned in pretraining (on Olmo 3).** Wang et al. ([2610.06851](https://arxiv.org/abs/2610.06851), MIT/Berkeley) show that fixing starting-token cues makes base models competitive with their RL counterparts. ".\n\nOkay" raises **Olmo-3-7B** MATH-500 pass@1 from 42% to 78%. RL makes these cues more likely, and causal data interventions show the cue effects come from training data. **This is the most direct prior evidence for H1 on our model family.** Our cue-forcing arm exists because of it.

**Few parameters, preserved representations.**
- RL updates a small subnetwork (Mukherjee et al., [2505.11711](https://arxiv.org/abs/2505.11711)).
- RLVR updates avoid principal weight directions (Zhu et al., [2511.08567](https://arxiv.org/abs/2511.08567)).
- Online RL forgets less than SFT because it is biased toward KL-minimal solutions (Shenfeld, Pari & Agrawal, [2509.04259](https://arxiv.org/abs/2509.04259)).
- RL preserves base representations while SFT introduces many specialised features, with a compact feature set mediating cases where RL succeeds and the base fails (Shi et al., [2604.25011](https://arxiv.org/abs/2604.25011)).
- Math RL transfers to other domains where SFT does not (Huan et al., [2507.00432](https://arxiv.org/abs/2507.00432)).
- Base-model steering recovers much of the gap to thinking models, so base models "know how" and RL teaches "when" (Venhoff et al., [2510.07364](https://arxiv.org/abs/2510.07364)).

**Learning dynamics.** Ren & Sutherland ([2407.10490](https://arxiv.org/abs/2407.10490)) give a step-wise account of how finetuning on one example changes predictions on others. This is the conceptual basis for reading co-influence changes.

**Implication.** Representation drift under RL is small and structured, which makes the drift baseline for H3 measurable. Attribution targets should focus on the few positions RL changed, not the whole response.

## 3. Pretraining and midtraining as determinants of RL

**Pretraining-to-RL interplay.**
- *Echo Chamber* (Zhao et al., [2504.07912](https://arxiv.org/abs/2504.07912)) trains models from scratch on known open-data mixtures and finds RL converges on one pretraining output distribution, amplifying it.
- Gandhi et al. ([2503.01307](https://arxiv.org/abs/2503.01307)) show four cognitive behaviours present in the base model (verification, backtracking, subgoal setting, backward chaining) predict whether RL self-improves, and priming with them unlocks gains.
- Reflection already emerges during pretraining (Essential AI, [2504.04022](https://arxiv.org/abs/2504.04022)).

**Controlled pipelines.**
- Zhang, Neubig & Yue ([2512.07783](https://arxiv.org/abs/2512.07783)) isolate pretraining, midtraining and RL on synthetic reasoning tasks. RL yields true pass@128 gains only when pretraining leaves headroom *and* RL data sit at the model's edge of competence; midtraining helps substantially under fixed compute.
- Shen et al. ([2607.16097](https://arxiv.org/abs/2607.16097)) use chess (5M–1B models, with a math check at 1B): post-RL performance is predicted by pretraining loss, and on hard puzzles RL surfaces moves that were nearly absent under SFT.

**Midtraining for RL.**
- OctoThinker (Wang et al., [2506.20512](https://arxiv.org/abs/2506.20512)) shows midtraining strategy determines RL scalability.
- ExpRL (Xiang et al., [2606.17024](https://arxiv.org/abs/2606.17024)) uses RL-based midtraining with reference solutions as reward scaffolds.
- Mid-training on self-generated data improves later RL ([2605.08472](https://arxiv.org/abs/2605.08472)).
- The Olmo 3 report‡ (Fig. 25) shows an early base model midtrained with insufficient reasoning data fails to grow response length under RL-Zero. It proposes RL-Zero as a testbed for midtraining mixes.

**Pretraining data selected from RL behaviour.** Wang & Chen ([2610.25836](https://arxiv.org/abs/2610.25836), Princeton) find RL failures are predicted by a small set of n-grams. They retrieve pretraining text with those n-grams and continue pretraining on 35M tokens before RL, improving OLMo 2 and Qwen3 in 10 of 11 settings. This is the closest *data-linkage* work. It uses retrieval rather than attribution, starts from failures rather than gains, and does not resolve stages.

**Implication.** The stage question (pretraining vs midtraining vs RL) is well motivated by controlled studies but unmeasured on a real open flow. The released Olmo 3 midtraining arms allow a behavioural version of the test.

## 4. Composition under RL

- Yuan et al. ([2509.25123](https://arxiv.org/abs/2509.25123)) define skills as string transformations f(x). When f and g are learned before RL, RL teaches unseen compositions g(f(x)), which generalise to deeper compositions and transfer across tasks; next-token training on the same data does not.
- Abdulsalam, Patel & Saxe ([2607.07646](https://arxiv.org/abs/2607.07646)) pretrain on primitive rewrite chains with a fully known distribution. RL solves held-out problems the pretrained model rarely solves even with large budgets by building and consolidating composed procedures. Pretraining ablations show this is gated by whether pretraining organises primitives into reduction procedures.
- He et al. ([2609.19465](https://arxiv.org/abs/2609.19465)) find a decomposed-to-composed asymmetry.
- Qin et al. ([2505.22756](https://arxiv.org/abs/2505.22756)) decompose problem solving into plan, execute and verify, and find a coverage wall for fundamentally new problems.
- Chu et al. ([2501.17161](https://arxiv.org/abs/2501.17161)): "SFT memorizes, RL generalizes."

**Implication.** Composition under RL is established in synthetic settings. What is untested is whether *training-data attribution* can detect it, and whether real 7B RL-Zero gains show it. Our gate is framed as an attribution benchmark that also replicates the composition result.

## 5. Contamination, spurious rewards and templates

- **Spurious rewards.** Random rewards improve Qwen2.5-Math-7B MATH-500 by 21.4 points, versus 29.1 for true rewards, but often fail on other families including OLMo 2 (Shao et al., [2506.10947](https://arxiv.org/abs/2506.10947)).
- **Contamination.** Wu et al. ([2507.10532](https://arxiv.org/abs/2507.10532)) attribute such effects to contamination.
- **Ai2's negative control.** The Olmo 3 report‡ (§6.2, Fig. 27) trains Olmo 3 Base on Dolci RL-Zero with random rewards and sees no gains, which supports their decontamination.
- **Templates.** Liu et al. ([2503.20783](https://arxiv.org/abs/2503.20783), the Dr. GRPO paper) show templates change measured base-model reasoning. The Olmo 3 report‡ confirms "simple" templates beat post-trained templates for a purely midtrained base.
- **Tools.** Decontamination uses Ai2's `decon` (verified CLI) with exact n-grams over infini-gram-style indices (Liu et al., [2401.17377](https://arxiv.org/abs/2401.17377)). OLMoTrace (Liu et al., [2504.07096](https://arxiv.org/abs/2504.07096)) gives verbatim span tracing for qualitative inspection only.

## 6. Training-data attribution methods

- **Gradient-tracing and influence.** TracIn (Pruthi et al., [2002.08484](https://arxiv.org/abs/2002.08484)); EK-FAC influence functions at LLM scale (Grosse et al., [2308.03296](https://arxiv.org/abs/2308.03296)); TRAK (Park et al., [2303.14186](https://arxiv.org/abs/2303.14186)), which also defines the linear datamodeling score (LDS) used in our gate.
- **TrackStar** (Chang et al., [2410.17413](https://arxiv.org/abs/2410.17413)): optimizer-aware, unit-normalised gradient attribution at pretraining scale.
- **SOURCE** (Bae et al., [2405.12186](https://arxiv.org/abs/2405.12186)): approximate unrolled differentiation.
- **MAGIC** (Ilyas & Engstrom, [2504.16430](https://arxiv.org/abs/2504.16430)): near-optimal attribution by metadifferentiation.
- **ASTRA** (Wang et al., [2507.14740](https://arxiv.org/abs/2507.14740)): better inverse-Hessian-vector products.
- **LESS** (Xia et al., [2402.04333](https://arxiv.org/abs/2402.04333)): gradient-based data selection.
- **Bergson** (Quirke, Jaburi, Johnston, Li, Paulo, Martres, Gupta, Belrose, Biderman; [2606.11660](https://arxiv.org/abs/2606.11660)) implements these with on-disk gradient stores and streaming scoring. Its current README leaderboard‡ (GPT-2, WikiText fine-tune) lists LDS **0.931 MAGIC, 0.685 EK-FAC+ASTRA(CG), 0.454 EK-FAC, 0.276 TrackStar (no optimizer), 0.156 gradient cosine**. These numbers supersede the ones quoted in our v2 plan. Bergson v2.3.0‡ added `--advantage_column` for GRPO rollouts; `--reward_column` computes group-mean advantages keyed on the prompt column.
- **Alternatives and variants.** Probe-based attribution (Xiao et al., [2602.11079](https://arxiv.org/abs/2602.11079)) ranks datapoints by activation-difference similarity. It beats gradient attribution on OLMo 2's DPO data at a tenth of the cost and is a candidate robustness method for our diff-based question. Nishida et al. ([2608.13515](https://arxiv.org/abs/2608.13515)) measure task-agnostic influence across Pythia pretraining. Kou et al. ([2505.19949](https://arxiv.org/abs/2505.19949)) apply influence functions to attribute math and code reasoning to SFT data attributes.

## 7. Attribution of reasoning and capability provenance

**Ruis et al.** ([2411.12580](https://arxiv.org/abs/2411.12580), ICLR 2025) use EK-FAC influence on 7B and 35B models over 2.5B pretraining tokens. For reasoning queries, influence is spread across documents that encode *procedural* knowledge, with similar influence across questions of the same task. This is the reason to aggregate.

**Matlin et al.** ([2606.19625](https://arxiv.org/abs/2606.19625), COLM 2026; Matlin, Chakraborty, Eom, Okamoto, Castilla, Jaburi, Deng, Min, Quirke, Biderman, Riedl) run TrackStar via Bergson on **OLMo3-7B** over a Dolma 3 working set aggregated into WebOrganizer's 24×24 bins (Wettig et al., [2502.10341](https://arxiv.org/abs/2502.10341)). They state that document-level scores are too noisy to identify which regions support which capability, and validate partially with targeted unlearning. Their artifacts are public under `HCAI-Lab-GT` on Hugging Face‡: working samples, a 100k-document preconditioner, TrackStar scores and an index dataset. **This is our methodological template and data source for the pretraining bins.**

## 8. Attribution in online RL

Hu et al. ([2505.19281](https://arxiv.org/abs/2505.19281)) attribute within a local rollout buffer.

**BehaviorTrace** (Nautiyal, [2610.10422](https://arxiv.org/abs/2610.10422)) plants a behaviour during GRPO on Qwen2.5-1.5B over three seeds and finds that much of the apparent signal is confounded:
- a gradient-norm-only control reaches 4.2–4.5× chance and matches or beats the best targeted estimator on 2 of 3 seeds;
- fluency predicts the label as well as every gradient method at saturated checkpoints;
- per-rollout verdicts flip across seeds and generation draws.

The one robust signal is the gradient of trigger tokens aligned with a target built where the behaviour occurs. **Every control in our plan comes from this checklist, and our fork-token target extends its one positive finding.**

## 9. The open model flow

The Olmo 3 report ([2512.13961](https://arxiv.org/abs/2512.13961))‡ releases every stage. RL-Zero details from §6 and Table 49:
- Dolci-RL-Zero-Math has 13,314 prompts from DAPO-Math, Klear-Reasoner Math, Open-Reasoner-Zero and OMEGA. It is decontaminated against pretraining and eval data, and offline-filtered to remove prompts the final base solves 8/8.
- RL-Zero training: group size 8, temperature 1.0, 32 prompts per step, 16K response length, 2,000 steps for 3.0, learning rate 1e-6 constant, clip 0.2/0.272, truncated importance sampling cap 2.0.
- Evaluation: 32K tokens at temperature 1.0.
- 3.1 changed the completion length from 12K to 16K and stopped masking truncated sequences, plateauing near 50% AIME pass@1 (Fig. 39).

OLMES ([2406.08446](https://arxiv.org/abs/2406.08446)) is the evaluation standard. Biderman et al.'s position paper ([2606.06533](https://arxiv.org/abs/2606.06533), "…A Science of AI Must Study Training Dynamics") argues for studying training dynamics rather than patching behaviour post hoc. Cite the arXiv title; v2 notes the ICML listing uses "Learning Dynamics".

## 10. Positioning

| Work | Open pretraining data | RL from base | Stage-resolved | Gradient TDA | Ground-truth validation | Real 7B |
|---|---|---|---|---|---|---|
| Invisible Leash; Yue et al. | — | ✓ | — | — | — | ✓ |
| Wang et al. 2610.06851 (cues) | ✓ (Olmo 3 among others) | ✓ | — | — (causal data edits) | ✓ (interventions) | ✓ |
| Wang & Chen 2610.25836 | ✓ (OLMo 2) | ✓ | partly (before/after midtraining insertion) | — (n-gram retrieval) | ✓ (CPT intervention) | OLMo2-1B, Qwen3-4B named in abstract |
| Zhang, Neubig & Yue; Yuan et al.; Abdulsalam et al. | synthetic | ✓ | ✓ (synthetic) | — | ✓ | — |
| Matlin et al. | ✓ (Dolma 3) | — | — | ✓ (TrackStar/Bergson) | partial (unlearning) | ✓ |
| BehaviorTrace | — | GRPO fine-tune | — | ✓ | ✓ (planted) | 1.5B |
| **This project** | ✓ (Dolma 3, Dolmino, Dolci) | ✓ | ✓ (stage finals, midtraining arms, RL steps, stage attribution) | ✓ | ✓ (planted-fragment gate) | ✓ |

**Gap.** No published work attributes RL-Zero expansion problems to pretraining, midtraining and RL-stage data on a real open flow, with attribution validated against ground truth under GRPO and with controls for format, norm and fluency confounds.

## 11. Corrections to the v2 plan's citations

| v2 statement | Correction |
|---|---|
| Invisible Leash authors "Fang Wu, Weihao Xuan, Ximing Lu, Mingjie Liu, Yi Dong, Zaid Harchaoui, Yejin Choi" | v5 adds Danny Xing as third author; the abstract wording is now "observed support" |
| Bergson author order ends "… Biderman, Belrose" | arXiv v4 order ends "… Girish Gupta, Nora Belrose, Stella Biderman" |
| Bergson LDS table (MAGIC 0.983, EK-FAC 0.257/0.474, SOURCE 0.387, TrackStar 0.184) | The current README table differs (see §6); cite the version you pin |
| RL-Zero template "Solve the following math problem…" | That is 3.0. **3.1 says "Solve the following problem…"** |
| Dolci sources DAPO + Klear | Plus Open-Reasoner-Zero and OMEGA |
| Intermediate RL-Zero checkpoints "unverified" | `step_0100`…`step_2800` exist on 3.1; `step_100`…`step_1900` on 3.0 |
| Bergson `--reward_column advantage` | Use `--advantage_column advantage` (≥2.3.0) or `--reward_column reward` with group-defining `--prompt_column` |

## References (arXiv IDs, all resolved 2026-10-22)

2002.08484 TracIn · 2303.14186 TRAK · 2308.03296 Grosse et al. · 2401.17377 infini-gram · 2402.04333 LESS · 2405.12186 SOURCE · 2406.08446 OLMES · 2407.10490 Ren & Sutherland · 2410.17413 TrackStar · 2411.12580 Ruis et al. · 2412.01951 Sharpening · 2501.17161 SFT memorizes, RL generalizes · 2502.10341 WebOrganizer · 2503.01307 Gandhi et al. · 2503.20783 Dr. GRPO · 2504.04022 Rethinking reflection · 2504.07096 OLMoTrace · 2504.07912 Echo Chamber · 2504.13837 Yue et al. · 2504.16430 MAGIC · 2505.11711 RL subnetworks · 2505.19281 Snapshot of influence · 2505.19949 Kou et al. · 2505.22756 Qin et al. · 2505.24864 ProRL · 2506.01347 Negative reinforcement · 2506.01939 High-entropy minority tokens · 2506.10947 Spurious rewards · 2506.20512 OctoThinker · 2507.00432 Huan et al. · 2507.10532 Reasoning or memorization · 2507.14740 ASTRA · 2507.14843 Invisible Leash · 2509.04259 RL's Razor · 2509.25123 Yuan et al. · 2510.07364 Venhoff et al. · 2510.15020 Coverage principle · 2511.08567 Path not taken · 2512.07783 Zhang, Neubig & Yue · 2512.13961 Olmo 3 · 2602.11079 Probe-based attribution · 2604.25011 Shi et al. · 2605.08472 Self-generated midtraining · 2606.06533 Biderman et al. · 2606.11660 Bergson · 2606.13125 Select and improve · 2606.17024 ExpRL · 2606.19625 Matlin et al. · 2607.07646 Abdulsalam et al. · 2607.16097 Shen et al. · 2607.20543 Zhou · 2608.13515 Nishida et al. · 2609.01274 Sun et al. · 2609.13443 Noukhovitch et al. · 2609.19465 He et al. · 2609.22547 Yang & Chen · 2610.04158 Cheng et al. · 2610.06851 Wang et al. (cues) · 2610.07405 Rath et al. · 2610.10422 BehaviorTrace · 2610.24663 Zhang et al. (few decisions) · 2610.25836 Wang & Chen
