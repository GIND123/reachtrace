# Attribution methods for RQ4, following the Bergson authors' advice

*2026-11-01. Supersedes the attribution settings in `docs/05_plan_v3.md` §5 (projection 16,
hand-built contrastive queries, sign-only RL items). Source checked: EleutherAI/bergson at
commit 15caf72 (release 2.7.0).*

## 1. What we asked and what we were told

We asked Lucia Quirke (EleutherAI, Bergson maintainer) for sensible defaults for
advantage-weighted RL item stores at 7B and whether our synthetic test is a convincing
check of attribution under GRPO. Her advice (2026-10-28), and what we do with each point:

| Advice | Implementation |
|---|---|
| Bergson's *contrastive queries* (`query.contrast`) are experimental; the *GRPO queries* feature (rollout advantages as a dataset column) is well tested | Queries are GRPO queries: one pre-tokenized row per rollout with `advantage_column`. `query.contrast` is not used. §2 shows the GRPO query already is a correct-minus-incorrect contrast |
| EKFAC is the moderately expensive, moderately accurate default; its Hessian can be fit on a subset (e.g. 30K items) | `kfac_hessian_step(subset=N)` fits factors on `train[:N]`; `influence` steps reuse them with `resume: true`. The gate compares full-fit with 25%-subset-fit factors |
| TrackStar: projection 32 or 64, always autocorrelation preconditioning + unit normalization | `trackstar_step`: `projection_dim` 32/64, `unit_normalize: true`, autocorrelation Hessian. Plan v3's p = 16 is kept only as a comparison row |
| Compressed methods correlate weakly with ground truth; EKFAC or EKFAC + ASTRA (tune its learning rate) are more accurate | The gate scores EKFAC, EKFAC + ASTRA (momentum SGD, learning rate swept) and ASTRA's conjugate-gradient solver against retraining |
| For many queries over a large dataset: TrackStar index for full passes, then EKFAC reranks the top ~10K (`examples/pretraining_scale_attribution`) | `twostage`: TrackStar p64 over the corpus, EKFAC on each query's top 10% (`score_cfg.candidates`). The gate measures how much of full-EKFAC's top 1% it keeps |
| Start from the leaderboard and the reproducible YAML of each method | Every builder in `src/reachtrace/attrib/bergson_yaml.py` names the YAML it starts from; every run writes a `bergson <yaml>` file next to its outputs |

## 2. GRPO queries and how each method treats the advantage

For a problem with n rollouts, k of them correct, the Dr. GRPO advantage is A = 1 − k/n for a
correct and −k/n for an incorrect rollout. The advantage-weighted query gradient is

    Σᵢ Aᵢ gᵢ = k(n − k)/n · (mean of correct gᵢ − mean of incorrect gᵢ)

(tested in `tests/test_gate_v3.py`). It is the contrastive query of plan v3, scaled by the
group's reward variance, and it is the direction one GRPO step on that problem moves the
model. Format and style shared by correct and incorrect rollouts cancel, so it addresses
the form-over-content confound (arXiv 2609.19589) with a supported feature. The objective
whose change it predicts is J_p(θ) = Σᵢ Aᵢ log p_θ(yᵢ | x_p) (log-probs summed over
tokens), which equals k(n − k)/n times the correct-minus-incorrect margin that the
likelihood ledger tracks across Olmo 3 checkpoints.

How Bergson 2.7.0 handles the advantage, read from source:

* The loss of each row is multiplied by its advantage before the backward pass
  (`collector/collector.py`, `fwd_bwd`), for queries and training rows alike.
* **EKFAC** (`influence`) refuses unit normalization, and scores are linear in the query
  gradient. With `aggregation: sum` over one problem's rows the query is exactly Σᵢ Aᵢ gᵢ.
  The advantage is fully respected; nothing to fix.
* **TrackStar** normalizes each row *before* aggregation
  (`process_grads.normalize_and_aggregate_grads`), so Aᵢgᵢ/‖Aᵢgᵢ‖ keeps only sign(Aᵢ).
  We build the query with `aggregation: none` and then multiply each column by |Aᵢ| and
  sum within the problem (`attrib/grpo_query.py`). The result is Σᵢ Aᵢ cos(H^-½qᵢ, H^-½d).
  The same applies to RL items indexed with `advantage_column` under unit normalization.
* Rows from groups with a single reward value have A = 0, a zero gradient and a 0/0
  cosine; they are dropped (they also contribute nothing to a GRPO update).
* `bergson validate` evaluates the query with plain cross-entropy and ignores the
  advantage, so it cannot provide LDS ground truth for GRPO queries; the gate measures J_p
  itself (`synthetic/cpt.py`).

Other facts checked in source: pre-tokenized rows keep their `labels` mask through to the
loss (`utils/worker_utils.py`, `keep = {"length", "input_ids", "labels"}`); TrackStar's
query/value Hessian mixing (`mix_hessians`) can be switched off when the query set is too
small for its own Hessian estimate, which is our case (tens to hundreds of rollouts); the
two-stage candidates file can be a `query --record` CSV with one `Top` row per candidate.

## 3. Synthetic gate v3: a check of attribution under GRPO with measured ground truth

`scripts/gate_v3.py`, `src/reachtrace/synthetic/{v3,cpt,rollout}.py`. SmolLM2-135M.

**Flow.** Continued pretraining (CPT) on 3,000 FineWeb-Edu documents plus 600 planted
documents per family, 3 epochs, then GRPO (TRL; Dr. GRPO advantages, no KL, clip 0.2/0.272,
8 rollouts per prompt, constant learning rate) on A-then-B and C-then-D composition prompts.

| Family | Content | Role |
|---|---|---|
| A, B | definitions and examples of `zorp` (digit substitution) and `blick` (positional rewrite) | causal for A-then-B |
| C, D | the same for look-alike operations `zarp`, `blink` | distractors |
| M | sentences that mention `zorp` and `blick` but teach nothing | lexical decoy: BM25 should fall for it |
| F | the RL prompt/answer format around trivial arithmetic | format only |
| W | solved C-then-D compositions in the RL solution format | procedure (analogue of reasoning-trace midtraining) |
| G | generic web text | background |

**Queries.** The RL model's rollouts on held-out A-then-B problems (long inputs first, the
analogue of problems the base cannot reach), 8 per problem, kept when rewards are mixed.
Gradients are taken at the CPT checkpoint: each stage's documents are scored at that
stage's own end checkpoint (plan v3 §5).

**Ground truth.** Retraining with per-document loss weights in an otherwise identical,
deterministic run (same order, batch shapes, schedule and denominators), so a retrain differs
from the full run only through the removed documents. Two full runs give identical J.

* *Dose-design LDS*: 40 retrains, each keeping every family (and G) at a fraction in
  {0, .25, .5, .75, 1}, nested within family. Per query, the Spearman correlation between
  the summed scores of kept documents and the retrained J_p; mean over queries, bootstrap
  over subsets.
* *Random-half LDS*: 24 retrains on random 50% document subsets (Park et al. 2023).
* *QLD*: retrain without each method's top 1% (summed over queries) and compare the drop in
  ΣJ_p with three random 1% removals (the leaderboard's proponent QLD).
* *Leave-family-out CPT → GRPO*: the A-then-B accuracy gained by RL when each family is
  removed from CPT, i.e. which pretraining data the RL gain depends on.
* *Ledger (G0)*: margin M and J along base → CPT → GRPO checkpoints.

**Prompt tokenization.** A prompt that ends in a blank line tokenizes differently alone
than inside a training document (SmolLM2 splits "\n\n" before a word but not at the end of
a string). The first gate run hit this; prompts now end in `Solution:` and solutions start
with a space, and a test checks that prompt tokens are a prefix of prompt + solution tokens.

## 4. Running on Lambda

`scripts/lambda_gpu.py` launches one instance through the Lambda Cloud API, refuses to
launch if spend so far plus the requested hours would pass the budget, keeps a local spend
ledger, and installs a watchdog on the instance that terminates it through the API at the
deadline (shutting the OS down does not stop billing). Instance and SSH-key names and the
API key come from environment variables only.

## 5. Results

[results/gate_v3/](../results/gate_v3/README.md) has the gate results and [results/olmo7b_bench/](../results/olmo7b_bench/README.md) the 7B costs. In short:

- EK-FAC ranked the planted families the RL gain depends on correctly (ρ = 0.89). TrackStar p32 ranked the families by their effect on the query margin correctly (ρ = 0.68). Neither did both.
- Document-level LDS stayed at or below 0.085 for every method, because single-document effects in this setting are not locally linear.
- Subset-fit EK-FAC factors and two-stage retrieval both reproduced 90% or more of full EK-FAC's top 1%.

- EK-FAC refined by ASTRA's conjugate-gradient solver is the best method against family-level effects on the objective (ρ = 0.75, p = 0.033, 6 of 7 signs). ASTRA's momentum-SGD solver, with its step size tuned by the objective as on the leaderboard, did not improve on EK-FAC.
- Exact metagradients through the bit-exact training run do no better at the document level. A weight scan shows why: changing one document's weight by 0.1% moves the objective by 3–5%, so document-level retraining is chaotic here.
- At 7B ([results/olmo7b_midtrain/](../results/olmo7b_midtrain/README.md)), the plain rollout query ranks midtraining sources reliably (split-half 0.83; QwQ reasoning traces first). The GRPO contrast does not pass the reliability gate (TrackStar 0.05, EK-FAC 0.20).
- That gate pooled problems. With 64 rollouts per problem ([results/olmo7b_deep/](../results/olmo7b_deep/README.md)), each problem's contrast replicates across halves of its own rollouts: document-level split-half 0.07–0.14, 2–6 SD above a shuffled-label null. Different problems' contrasts do not agree (0.00–0.02), so the contrast is problem-specific. On the one problem whose incorrect solutions all finish at similar length (AIME 2026 #16), it survives dropping cut-off rollouts and controlling for length (0.12, z 3.6). Reliability is therefore gated within each problem against a shuffled-label null. Long documents are scored as the mean of 1,024-token chunks.

The 7B study therefore reports source-level rankings from both stages of the recommended pipeline, not document-level effect sizes.
