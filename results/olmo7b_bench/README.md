# Cost of the recommended attribution pipeline at 7B

What the two-stage setup of [docs/06_attribution_methods.md](../../docs/06_attribution_methods.md) (TrackStar index over everything, then EK-FAC rerank of the top candidates) costs on `allenai/Olmo-3.1-7B-RL-Zero-Math`, measured on one Lambda H100 PCIe (80 GB) on 2026-11-01. Script: [scripts/olmo7b_bench.py](../../scripts/olmo7b_bench.py); the exact Bergson YAMLs are in [configs/](configs/).

**Rows.** 256 Dolci-RL-Zero-Math prompts rendered with the RL-Zero template, each followed by 2,048 tokens of FineWeb-Edu text standing in for a rollout, with random ±0.5 advantages (`advantage_column`). That is 562,529 tokens in total (about 2.2k per row). Cost depends on lengths and shapes, not on what the text says, so this needs no generation. Precision bf16, `lm_head` excluded.

| Stage | Rows | Tokens | Wall time | Tokens/s | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| TrackStar p32: autocorrelation Hessian, query index, score every row | 256 | 562,529 | 434 s | 1,297 | 30.7 GB |
| TrackStar p64: the same | 256 | 562,529 | 660 s | 852 | 57.3 GB |
| EK-FAC factors (KFAC + eigenvalue correction), last 4 of 32 layers | 64 | 140,632 | 386 s | 365 | 39.7 GB |
| EK-FAC rerank of 32 candidates for one GRPO query, last 4 layers | 32 | 70,316 | 107 s | 655 | 32.3 GB |

Each TrackStar row makes two passes over the rows (Hessian, then scoring), so a single pass runs at about 2,600 tokens/s. Wall times include model loading (about 30 s per stage).

## What this means for the full study

- **TrackStar p32 is the right first stage at 7B.** It fits a 40 GB A100 with headroom. p64 needs an 80 GB GPU: its per-module autocorrelation Hessians are 16× larger, and at 7B they exceed a 40 GB card.
- **First-stage budget.** The plan v3 RL stage (2,000 prompts × 4 rollouts × 3 checkpoints, about 5k tokens per rollout) is about 120M tokens, i.e. about 26 GPU-hours of TrackStar p32 at this rate (about $85 at Lambda's H100 PCIe price). The pretraining tier (120k documents × 1,024 tokens) is about the same size.
- **EK-FAC is the expensive stage, as advised.** It ran on a block of 4 layers because full 7B factors (about 70 GB in fp32) do not fit next to the model on a 40 GB GPU. Scaling the measured time linearly to 32 layers gives about 46 tokens/s for the factor pass and 82 tokens/s for reranking. That is an **upper bound** on cost, since the forward/backward part of each step does not grow with the number of layers tracked. Even so, fitting factors on 30K items needs short items or several GPUs, and reranking 10K candidates of 5k tokens for every query is out of reach on one GPU. The practical setting is to fit the factors on a 30K subset of ~1k-token documents with `module_partitions`, and rerank hundreds rather than thousands of candidates per query.
- **These are measurements on one GPU type with stand-in text.** Real rollouts vary in length, and padding changes the token batching. P0 of the plan should repeat this on the actual item sets before the budget is fixed.
