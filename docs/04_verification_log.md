# Verification log

Facts the code and plan depend on, how each was checked, and the result. All checks were run on 2026-10-22.

## Models and checkpoints (Hugging Face API)

| Claim | Check | Result |
|---|---|---|
| RL-Zero 3.1 Math has intermediate checkpoints | `GET /api/models/allenai/Olmo-3.1-7B-RL-Zero-Math/refs` | **Yes:** `main` plus `step_0100` … `step_2800` every 100 (29 branches) |
| RL-Zero 3.0 Math intermediate checkpoints | same endpoint for `allenai/Olmo-3-7B-RL-Zero-Math` | `main` plus `step_100` … `step_1900` (20 branches) |
| RL-Zero 3.1 Code checkpoints | same endpoint | 34 branches (`step_050` …) |
| Base stage revisions | refs for `allenai/Olmo-3-1025-7B` | stage1: 1,421 branches (final `stage1-step1413814`); stage2: 49 (final `stage2-step47684`); stage3: 13 (final `stage3-step11921`) |
| Midtraining ablation arms | refs plus the model card | `stage2-step47684-mix-{round5, gen-mc-only, math-code-reasoning-web}-from-2T-ckpt`. The card says these are the Table 7 domain-tradeoff runs: 100B midtraining tokens from a stage-1 checkpoint at 2T tokens |
| `main` = end of stage 3 | `GET /api/models/.../tree/{main, stage3-step11921}`, safetensors LFS oids | Different commit SHAs, **identical weight files** |
| SHAs | `reachtrace pin` → [configs/pins.lock.yaml](../configs/pins.lock.yaml) | 14 revisions pinned |
| RL-Zero 3.1 chat template | raw `chat_template.jinja` | Header "Solve the following **problem** step by step…" (3.0 says "math problem"). Rendered string = header + `\n\n` + question + `\n\nRemember to put your answer on its own line after "Answer:"`. Tested in `tests/test_prompts.py` |
| Tokenizer | `tokenizer_config.json` | eos `<|endoftext|>`, no BOS, pad `<|pad|>`, model_max_length 65,536 |
| transformers requirement | model card | ≥ 4.57.0 (architecture `Olmo3ForCausalLM`) |

## Datasets

| Dataset | Check | Result |
|---|---|---|
| `allenai/Dolci-RL-Zero-Math-7B` | datasets-server `/info`, `/first-rows` | 13,314 train rows; columns `prompt`, `ground_truth`, `messages`, `dataset`; ODC-BY; `messages` holds the raw problem without the template |
| `allenai/dolma3_dolmino_mix-100B-1025`, `allenai/dolma3_longmino_mix-50B-1025` | `/api/datasets` | exist; ODC-BY |
| `allenai/dolma3` | `/api/datasets` | **does not resolve as a dataset.** The base card points to `allenai/dolma3_mix-6T-1025` for stage 1 |
| Eval sets | `/first-rows` | MATH-500 (`problem`, `answer`, `level`, `unique_id`); aime_2024 (`problem`, `answer`, `id`; split `train`); AIME2025 (configs `AIME2025-I`/`-II`, `question`, `answer`); amc23 (`question`, `answer`); minervamath (`question`, `answer`); gsm8k `main` (`answer` after `####`); hendrycks_math (per-subject configs, `Level N`, boxed in `solution`); OlympiadBench `OE_TO_maths_en_COMP` (`final_answer` as a list string, `is_multiple_answer`, `modality`) |
| Licences | Hub `cardData.license` | AIME2025 MIT; OlympiadBench, DAPO-Math-17k, KlearReasoner-MathSub-30K Apache-2.0; hendrycks_math MIT; **none listed** for MATH-500, aime_2024, amc23, minervamath |
| HCAI-Lab artifacts (Matlin et al.) | `/api/datasets?author=HCAI-Lab-GT` | 50 datasets, including `dolma3-data-attribution-index`, `dolma3-6t-preconditioner-100k` (100k docs, 251M tokens), `dolma3-trackstar-influence-scores`, `dolma3-6t-sample-*`, `dolma3-6t-bloom-index`; model repo `unlearning-checkpoints` |

## Olmo 3 report (arXiv 2512.13961v2, read via full text)

| Item | Result |
|---|---|
| Dolci-RL-Zero-Math sources | DAPO Math, Klear-Reasoner Math, Open-Reasoner-Zero, OMEGA. Decontaminated against pretraining and eval data; offline-filtered to drop prompts solved 8/8 by the final base |
| RL-Zero hyperparameters (Table 49) | LR 1e-6 constant; 2,000 steps (3.0); max prompt 2,048; response 16,384; 32 prompts per batch; group size 8; TIS cap 2.0; T = 1.0; clip 0.2 / 0.272; 8 learner + 64 actor GPUs |
| Eval protocol | 32K tokens, T = 1.0; AIME pass@1 bootstrapped over 32 samples, plus pass@32 |
| 3.1 changes (A.6.4) | completion length 12K → 16K; truncated sequences no longer masked; Fig. 39 plateau ≈ 50% AIME pass@1 |
| Spurious-reward control (Fig. 27) | random rewards on Dolci RL-Zero give no gains on their evals |
| Midtraining → RL-Zero (Fig. 25) | an "insufficient" reasoning midtraining mix does not grow response length under RL |

## Bergson (source at commit d8478a1, 2026-10-21; PyPI)

| Item | Result |
|---|---|
| Latest release | 2.4.0 (2026-10-21); requires Python ≥ 3.10, transformers ≥ 5.0, huggingface-hub ≥ 1.13 |
| GRPO support | `DataConfig.reward_column` (advantage = reward − group mean grouped by `prompt_column`) **or** `advantage_column` (added v2.3.0, 2026-10-20). They are mutually exclusive. Loss is the Dr. GRPO policy gradient |
| Prompt/completion tokenization | `apply_chat_template([user, assistant])`, loss on the assistant span located by `rfind`. With the RL-Zero template this drops the reminder line, hence the identity-template tokenizer |
| Streaming scoring | `bergson score <out> --query_path <query index> --score individual --unit_normalize --projection_dim N` writes `scores.bin` without storing corpus gradients |
| Other relevant config | `QuerySetConfig.contrast` (subtract a control set's mean gradient); `projection_target` per_module or global; `attribute_tokens`; `loss_fn` ce, kl or log_odds; `bergson trackstar` pipeline with `target_downweight_components` |
| Leaderboard (README) | LDS: MAGIC 0.931; EK-FAC+ASTRA(CG) 0.685; Eigenvalue-corrected Shampoo 0.517; SOURCE 0.473; EK-FAC 0.454; BM25 0.220; TrackStar (no optimizer) 0.276; TRAK-8 0.138; gradient cosine 0.156 |
| Contact | lucia@eleuther.ai; `#data-attribution` on the EleutherAI Discord |

## Other tooling

| Tool | Result |
|---|---|
| vLLM (PyPI) | 0.32.0; Python ≥ 3.11, < 3.15; transformers ≥ 5.16.1, < 5.20; torch == 2.13.0 |
| math-verify | 0.9.0 |
| TRL GRPOConfig (v1.16.0 source) | `loss_type` ∈ {grpo, dr_grpo, dapo (default), bnpo, cispo, sapo, luspo, vespo}; `scale_rewards` ∈ {group, batch, none}; `beta` default 0.0; `epsilon` 0.2; `epsilon_high` |
| allenai/decon (README) | `decon evals --download --config`, `decon detect --training-dir --evals-dir [--purify]`, `decon review [--stats] <dir>`, `decon compare`, `decon server --port`; sensitivity via `contamination_score_threshold` |

## Literature

All 62 arXiv IDs in [02_literature_survey.md](02_literature_survey.md) were resolved through `export.arxiv.org/api/query?id_list=…`, which returns title, authors and date. Recent related work was found with alphaXiv search (recency-prioritised) on three queries: RLVR × attribution, RL reasoning boundary × composition, and Olmo × provenance.

## Added 2026-10-26 (plan v3 build)

| Claim | Check | Result |
|---|---|---|
| RL-Zero 3.1 `main` vs `step_2800` | `/api/models/.../tree/{main,step_2800}` LFS oids | **Identical weights**; sample `main` only |
| Olmo 3 7B attention | `config.json` | `num_key_value_heads: 32` (no GQA); sliding window 4,096 on 3 of 4 layers |
| Context of stage checkpoints | `config.json` at `stage1-step707000`, `stage1-step1413814`, `stage2-step47684` | `max_position_embeddings: 8192`, no RoPE scaling; `stage3-step11921` has YaRN to 65,536. Ledger traces must fit 8,192 tokens |
| Midtraining arms' 2T start | stage-1 branch list; 5.93T / 1,413,814 steps ≈ 4.19M tokens/step | 2T ≈ `stage1-step476838` (estimate; confirm on the model card) |
| Bergson unit normalization vs advantages | `bergson/process_grads.py:159` (commit d8478a1) | Each item gradient is divided by its norm *after* the advantage is applied, so only sign(A) survives. Fixed by post-hoc weighting (`attrib/rl_scoring.py`) |
| Bergson pre-tokenized input | `bergson/utils/worker_utils.py:426` | Tokenization is skipped when `input_ids` is present; label masks still to be confirmed |
| Bergson loss reduction | `config.py` | Index default `loss_reduction="sum"` (no length normalization, consistent with Dr. GRPO) |
| Matlin et al. compute | paper App. L | 20,000 GPU-h for the 5.68M-document index (10.5B tokens): ~150 tokens/s/GPU; d=16 (256 per module); working set excludes FineMath, arXiv, Stack-Edu, Wikipedia |
| Post-cutoff eval sets | Hub API / datasets-server | `MathArena/{aime_2026, hmmt_feb_2026, hmmt_feb_2025, brumo_2025, cmimc_2025}` (CC-BY-NC-SA-4.0; columns `problem_idx, problem, answer[, problem_type]`); `KbsdJames/Omni-MATH` (Apache-2.0, 4,428 rows, `difficulty`, `source`) |
| Modal pricing | modal.com/pricing | H100 $3.95/h, A100-80GB $2.50/h, L4 $0.80/h; volume storage free up to 1 TiB |
| vLLM on Modal | smoke test | vLLM 0.32.0 pip wheel has no FlashInfer cubins and the image has no nvcc; set `VLLM_USE_FLASHINFER_SAMPLER=0` |
| vLLM throughput, Olmo 3 7B, 1×H100 | pilot jobs (124 problems × 16 samples, 7,680-token cap) | base 1,530 and RL 1,338 generated tokens/s (wall-clock incl. ~2 min load). RL mean 6,975 tokens/sample, 73% at the cap |
| "AIME 2026 Problem 1" (MathArena) vs Dolci | 13-gram overlap | **Verbatim in Dolci-RL-Zero-Math-7B row 574 with the same answer (277)**; Dolci was uploaded 2025-11-17, before the contest. A post-cutoff date does not guarantee cleanliness |
| Omni-MATH vs Dolci | 13-gram overlap > 0.4 of the question | 13 of 60 sampled difficulty-4–6.5 problems (22%) are Dolci near-duplicates |
| 13-gram "any shared n-gram" decon rule | pilot selection | Flags boilerplate ("where m and n are relatively prime positive integers. Find m+n"); use an overlap fraction (0.4 in the pilot) plus manual review |

## Added 2026-11-01 (attribution methods after the Bergson authors' advice)

| Claim | Check | Result |
|---|---|---|
| Bergson release | PyPI; repo at commit 15caf72 | 2.7.0; installs with torch 2.11 (cu128) / 2.15 (cu132), transformers 5.19, TRL 1.17.1 |
| GRPO queries | `config/config.py` (`QuerySetConfig` inherits `DataConfig`) | A query set accepts `advantage_column`; the row loss is multiplied by the advantage before backward (`collector/collector.py`, `fwd_bwd`) |
| `query.contrast` | `build.py`, `subtract_control` | Subtracts a control set's mean gradient; needs aggregation mean/sum. Described to us by the maintainer as experimental; not used |
| Unit normalization vs advantage | `process_grads.normalize_and_aggregate_grads` | Per-row normalization runs before aggregation, so a unit-normalized GRPO query keeps only sign(A) per row. Restored by multiplying columns by \|A\| (`attrib/grpo_query.py`, tested) |
| EK-FAC and unit normalization | `hessians/pipeline.py` | `hessian_pipeline` raises if `unit_normalize`; scores are linear in the query, so the advantage is exact |
| Factors on a subset | `examples/pretraining_scale_attribution/build.yaml`; `hessian_pipeline` step 2 skip under `resume` | A `hessian` step on `train[:N]` writes `{run}/hessian/kfac`; `influence` with `resume: true` reads it back |
| Two-stage candidates | `score/candidates.py` | `score_cfg.candidates.scores` may be a `query --record` CSV; rows with `direction == "Top"` give `result_index` |
| TrackStar Hessian mixing | `cli/trackstar.py`, `TrackstarConfig.mix_hessians` | Can be disabled to precondition with the training autocorrelation alone (documented for small query sets) |
| TrackStar p64 memory | gate v3 run | Per-module autocorrelation is (64²)² fp32 = 67 MB; 210 modules of SmolLM2-135M ≈ 14 GB (OOM on an 8 GB GPU, fine on A100-40GB) |
| Pre-tokenized labels | `utils/worker_utils.py` | `labels` is kept with `input_ids` (`keep = {"length", "input_ids", "labels"}`), so label masks reach the loss |
| `bergson validate` and advantages | `validate.py` (`per_doc_query_losses`, `mean_query_loss`) | Query loss is plain cross-entropy; advantages are ignored, so GRPO-query LDS needs our own retraining loop |
| Leaderboard (GPT-2 / WikiText) | `examples/gpt2_wikitext_leaderboard/README.md` | LDS: EK-FAC 0.454, EK-FAC+ASTRA 0.605, EK-FAC+CG 0.685, TrackStar p16/p32/p64 0.120/0.194/0.276, BM25 0.220, MAGIC 0.931 |
| Lambda Cloud API | `GET /api/v1/instance-types` | Python's default urllib User-Agent is rejected by Cloudflare (error 1010); A100-40GB SXM4 $1.99/h, GH200 $2.29/h, H100 SXM $4.29/h on 2026-11-01 |
| Lambda A100 driver | `nvidia-smi` on the instance | CUDA 12.8 driver; the default torch wheel (cu132) fails, the cu128 wheel works |
| SmolLM2 prompt boundary | tokenizer check | `"...)).\n\n"` alone ends in one `"\n\n"` token; inside a document followed by a word it is `"\n"`, `"\n"`. Gate prompts end in `"Solution:"` and solutions start with a space |
| Retrain determinism (A100) | two identical weighted-CPT runs, same and different instances | The flash / memory-efficient SDPA backwards are not deterministic (PyTorch only warns under `warn_only`): identical runs gave J = 176.4 and 167.7. With the math backend (`synthetic/cpt.py`), two runs and a run on a second A100 instance all give J = 162.392 exactly. The same CPT on an H100 gives a different J (165.4), so a retrain family must stay on one GPU type |
| Smoothness of the gate objective | full run vs one-document removal vs half weight (`groundtruth smooth`) | CPT lr 2e-4: one document out of 7,200 moves J by 30–100% (chaotic). lr 5e-5: 0.4–2.6%, but half weight moves J the opposite way from removal (+4.2 vs −1.6), so not locally linear |
| Lambda watchdog | H100 instance at its deadline | Terminated through the API at the deadline as designed (a CG run in progress was lost) |
| MAGIC reverse pass | `tests/test_magic.py`: replay vs `train_cpt`; metagradients vs central finite differences in float64 | The replay is bit-identical to `train_cpt`. On a float64 GPT-2, dJ/dw matches finite differences to ~1e-9 relative (Adam eps 1e-3 and 1e-8). On Llama (SmolLM2's architecture) finite differences stall at ~1e-3 because `LlamaRMSNorm` computes in float32 even for a float64 model, so the tight check uses GPT-2 |
| Adam's eps and smoothness | same test, h from 1e-4 to 1e-10 | With eps = 1e-8 finite differences converge more slowly for some documents (1.7e-5 relative error at h = 1e-6 vs 1e-9 at eps = 1e-3): m / (sqrt(v) + eps) bends sharply where gradient entries are near eps |
| Bergson MAGIC | `bergson/magic/` (2.7.0) | Bergson ships MAGIC with its own replayable trainer and a leave-k-out validator (`skip_validation: false`, `subsets: bank`). Its query ignores advantages, so the gate differentiates through its own `train_cpt` instead (`synthetic/magic.py`) |
| ASTRA step-size rule | leaderboard `ekfac_astra.yaml` header | Step size chosen by the lowest mean final / \|step-0\| objective over 1e-3 to 1e-7 (3e-6 there; 1e-5 and above diverge). ASTRA logs the objective every 50 steps on that step's random batch |
| vLLM on Lambda's driver | `pip install vllm==0.11.0` in a separate venv on an H100 SXM5 (driver 570, CUDA 12.8) | vLLM 0.11.0 pins torch 2.8.0, whose PyPI wheel is cu128, so it runs; it maps `Olmo3ForCausalLM` to its Olmo2 implementation. vLLM ≥ 0.11.1 needs torch ≥ 2.9 |
| Olmo 3 7B attention | `config.json` | 32 query and 32 key/value heads (no GQA), sliding window 4096 on 3 of every 4 layers: ~0.5 MB of KV cache per token in bf16 |
| Dolmino source folders | `list_repo_tree` on `allenai/dolma3_dolmino_mix-100B-1025` | Sources are split into files by origin or topic; the alphabetically first file is not representative (`dolmino-math` starts with a CodeSearchNet Go subset, `stem-heavy-crawl` with its adult-content topic bin). Documents are drawn from 4 files per source chosen with probability proportional to size |
| Hugging Face downloads | `snapshot_download` on Lambda | Weights arrive through the Xet chunk cache, so `du` on the model folder under-reports |
| MAGIC replay at scale | `gate_v3.py magic` on an A100, smooth regime | Replayed run reproduces the original J_total exactly (162.391751); 6 targets in 101 min |
| Retraining chaos (smooth regime) | `groundtruth eps`: one document's weight at 0.9–1.1 | Weight 0.999 / 1.001 moves J_total by +8.9 / +5.2, more than removing the document (+4.2); MAGIC's tangent predicts ±0.001 |
| ASTRA SGD step size | summed query, 300 steps | Diverges at ≥ 3e-6 (as the leaderboard at ≥ 1e-5); objective falls at 1e-6 and 3e-7 |
| ASTRA CG | 16 per-problem queries, 20 steps, damping 10 | Every objective fell by > 99.8%; 11 of 16 ended negative; 72 min on an A100 with the factor fit |
| EK-FAC memory at 7B | H100 80 GB, last 4 layers | Factors OOM at an 8,192-token batch (eigenvalue correction holds per-document gradients); fine at 2,048. 19 full-gradient query rows (~3 GB each, fp32) need `score_cfg.query_batch_size` |
| Stage-2 checkpoint context | Bergson on `stage2-step47684` | Rejects `token_batch_size` above 8,192 (the checkpoint's max length) |
