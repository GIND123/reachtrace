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
