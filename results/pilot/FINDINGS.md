# Pilot findings (exploratory, 2026-10-26)

**What this is.** A small, cheap test of the plan-v3 instruments on Olmo 3 (base `Olmo-3-1025-7B` vs `Olmo-3.1-7B-RL-Zero-Math`), run on Modal for about $20. It is not the pre-registered study: n is small, base and RL got 16 samples each per problem, and generations were capped at 7,680 tokens so every trace fits the 8,192-token context of stage-1/2 checkpoints. Numbers are in [README.md](README.md); code is `scripts/pilot.py`.

**Problems.** 124 after removing Dolci near-duplicates: AIME 2026 (29), HMMT Feb 2026 (33), Omni-MATH difficulty 4–6.5 (47), MATH-500 level 5 (15). Groups at n = 16: G = base 0/16 under strict and lenient grading with RL ≥ 2/16 (7 problems); B = base 1–2/16 with RL ≥ 2/16 (11); E = easy reference (16).

## Findings

1. **Decontamination by date is not enough.** "AIME 2026 Problem 1" as listed by MathArena is verbatim in Dolci-RL-Zero-Math-7B (row 574, same answer, 277), and Dolci was uploaded in November 2025, before the contest. 13 of 60 sampled Omni-MATH problems (22%) are Dolci near-duplicates (> 40% of the question's 13-grams shared). A rule that flags any single shared 13-gram is unusable: competition boilerplate flagged 26 of 101 problems, including 2026 contests.

2. **At this budget RL looks like sharpening.** RL pass@1 is 3–4× the base, but pass@16 is close: AIME 2026 0.31 vs 0.28, HMMT 2026 0.21 vs 0.15, Omni-MATH 0.45 vs 0.43 (MATH-500 L5 0.80 vs 0.60). Caveat: 72–86% of RL samples on the hard sets hit the 7,680-token cap, so this cap understates the RL model.

3. **RL's likelihood change sits in the first few tokens.** Teacher-forcing the RL model's own correct solutions, RL (base → final) raises log-likelihood by +3.6 nats/token at positions 0–4 and +0.53 at 4–16, but only about +0.05 nats/token across the body of the solution. Opening log-likelihood goes −1.48 → −0.71 (RL step 200) → −0.17 (final). This is the cue effect of Wang et al. (arXiv 2610.06851), seen here as likelihood along the RL trajectory.

4. **Midtraining is where RL-style correct solutions become likely.** Content log-likelihood of correct RL traces: −1.08 at the end of pretraining, −0.52 after midtraining (+0.56), −0.50 after long-context training, −0.45 after RL (+0.05). The rise is spread over the whole solution, not just the opening.

5. **Ai2's midtraining data-mix arms make this causal.** From the same ~2T-token stage-1 checkpoint, 100B tokens of the math-code-reasoning mix raise correct-trace content log-likelihood by +0.59 nats/token; the general-QA mix by +0.10 (difference 0.49, 95% CI [0.47, 0.51], 35 problems). The arms are pre-release `olmo2-retrofit` checkpoints, loaded as Olmo 3 with sliding window 4096 (the config says 4097); see the caveats below.

6. **A preference for correct over incorrect content appears in midtraining, not RL.** The margin M (per-token log-likelihood of a correct minus an incorrect solution, same window, answer line excluded) needs finished incorrect solutions:
   - On RL traces it looked positive, but most incorrect RL traces are truncated; restricted to finished ones (10 problems) M ≈ 0 everywhere. That version measured fluency.
   - On the base model's *own finished* solutions (32 problems, 39 pairs, lengths within 2x), M is ≈ 0 after pretraining (−0.03 [−0.13, 0.08]), rises during midtraining (increment +0.146 [0.094, 0.198]) and is unchanged by long-context training and RL (RL increments ≈ 0, −0.019 late in RL).
   - The midtraining arms reproduce this from the same ~2T start: math-code-reasoning mix M = +0.13 [0.02, 0.24], general-QA mix −0.01 [−0.11, 0.10].
   So the model's preference for correct reasoning over flawed reasoning is learned from reasoning-heavy midtraining data; RL-Zero does not add to it.

7. **The base needs a substantial part of an RL solution, not just the cue.** Given only the RL trace's first 16 tokens, base success on G problems goes 0% → 10% (B: 9% → 14%). It reaches 35% with the first quarter of the solution, 48% with half, 75% with three quarters (B: 30%, 69%, 80%). About 20% of G and B problems are handed off within the first 10% of the solution. 2 of 5 G problems show the composition signature (two separate jumps in base success).

## How to read 3, 4 and 7 together

Per-token changes from RL in the body are small (~0.05 nats/token), but over a ~5,000-token solution that is ~250 nats, i.e. a large change in the probability of the whole solution. The pilot therefore does not say "RL only changes the opening". It says the opening is where RL's change is *concentrated*, while the content that RL solutions use was made likely by reasoning-heavy midtraining, and the small, spread-out RL changes in the body still matter for whether the base can finish (finding 7).

## Caveats

- Small n: G = 7 (5 in the ledger and handoff), B = 11; one pilot; exploratory, not pre-registered.
- G is "0 of 16 base samples", far weaker than the plan's 0 of 256; escalation was skipped to stay within budget.
- The 7,680-token cap truncates most RL samples on hard problems; RL accuracy is understated and most incorrect RL traces are truncated.
- Decontamination here is a 13-gram check against Dolci only (not against Dolmino or Dolma).
- The ~2T start of the midtraining arms is `stage1-step476838`, the branch nearest 2T tokens; the model card says only "a Stage 1 checkpoint trained to 2T tokens".
- The arm checkpoints were loaded with a patched config (sliding window 4096 instead of the pre-release 4097); their log-likelihoods are in the expected range, but the patch is not verified against Ai2's conversion.
- The ledger uses the RL model's own traces, so likelihood under the final RL model is not an independent measurement for the RL stage.

## What this means for the full study

- Keep the opening/content split and the position profile: they separate cue effects from content cleanly.
- Compute the margin on finished solutions only (section 2b); for RL traces, sample until each problem has finished incorrect traces, or raise the cap (32k for the long-context and RL checkpoints, 8k windows for stage-1/2).
- The midtraining arms are the strongest causal lever available; plan for loading them properly (`olmo2-retrofit` conversion).
- Budget: RL-Zero writes ~7,000 tokens per sample at a 7,680 cap and vLLM runs at ~1,400 generated tokens/s per H100. Sampling dominates cost: about $11.5 per 2,000 RL samples. The ledger is cheap (~$0.1 per checkpoint for ~0.6M tokens).
