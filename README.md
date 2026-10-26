# reachtrace

**Where do RL reasoning gains come from?** Provenance attribution of RLVR on the fully open Olmo 3 model flow.

When RL-Zero training lets `allenai/Olmo-3.1-7B-RL-Zero-Math` solve problems that `allenai/Olmo-3-1025-7B` cannot reach, does that capability trace back to:

- **(H1) elicitation**: the same pretraining and midtraining regions that support problems the base model can already reach;
- **(H2′) acquisition**: the RL-stage signal itself (Dolci prompts and on-policy rollouts); or
- **(H3) composition**: new joint structure over regions that already supported the parts?

The current design is **plan v3** ([docs/05_plan_v3.md](docs/05_plan_v3.md)). Stage questions are answered by exact behavioural measures; gradient attribution is limited to comparisons *within* a stage and runs only after a synthetic gate passes.

1. **Likelihood ledger and discrimination margin.** Teacher-forced log-likelihood of correct vs incorrect RL traces at every checkpoint along the flow (pretraining, midtraining, long-context, RL steps). The correct-minus-incorrect margin cancels format, and its stage increments add up exactly.
2. **Handoff map.** How much of a correct RL solution the base model needs before it can finish it; behaviourally identified critical segments.
3. **Continuous reachability.** Base solve rates on decontaminated, partly post-cutoff problems, with lenient grading and matched budgets (replaces certified U/R labels).
4. **Within-stage attribution (gated).** Contrastive, label-masked Bergson queries; RL items scored with the advantage applied after normalization, against placebo rollouts.

First results from a small exploratory pilot are in [results/pilot/](results/pilot/README.md).

## Documents

| File | What it is |
|---|---|
| [docs/01_idea_analysis.md](docs/01_idea_analysis.md) | Critical review of the v2 plan: what is novel after the July–Oct 2026 literature, what breaks, and the repositioned project |
| [docs/02_literature_survey.md](docs/02_literature_survey.md) | Verified literature survey (every arXiv ID resolved on 2026-10-22) |
| [docs/03_implementation_plan.md](docs/03_implementation_plan.md) | Work packages, specifications, gates and decision rules, with no calendar |
| [docs/04_verification_log.md](docs/04_verification_log.md) | Facts checked against the Hub, PyPI, the Bergson source and the Olmo 3 report, with how each was checked |
| [docs/05_plan_v3.md](docs/05_plan_v3.md) | **Current plan (v3)**: ledger, margin, handoff, gated within-stage attribution |
| [prereg/analysis_plan.md](prereg/analysis_plan.md) | Pre-registration draft for v3, frozen after Phase A |
| [results/pilot/](results/pilot/README.md) | Exploratory pilot on Olmo 3 (small n, 7,680-token cap) |

## Layout

```
src/reachtrace/
  prompts.py          RL-Zero templates, byte-identical for base and RL (tested against the HF Jinja)
  registry.py         revision -> SHA pinning; stage-final / midtrain-arm / RL-step branch parsing
  cache.py            content-addressed generation/index/score keys, resumable manifests
  data/               eval sets (unified schema), Dolci RL items (Dr. GRPO advantages), corpus sampling
  decon/              n-gram scan for RL references; wrapper for allenai/decon
  sample/             vLLM runner, adaptive sequential sampling, Answer:-line grading
  reach/              pass@k, continuous Jeffreys rates and G/B groups, partitions, checkpoint provenance
  ledger/             teacher-forced likelihood ledger, masks, discrimination margin, stage increments
  handoff/            prefix handoff map, critical segments, composition signature
  remote/             Modal jobs: weight download, vLLM sampling from token IDs, ledger scoring
  attrib/             Bergson commands, RL-item scoring (advantage after normalization), contrastive
                      scores, label-masked rows, BM25 baseline, aggregation, controls
  synthetic/          planted-fragment corpus (v2: aliases, format and worked-trace families,
                      dose-design LDS), composition tasks, CPT + GRPO, ground truth
  analysis/           bootstrap, stratified permutation, BH, TOST, MDE, CEM matching, proximity
configs/              model pair, attribution tier A, synthetic gate
tests/                runs on CPU in seconds
```

## Quick start

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"     # CPU analysis core + tests
pytest

reachtrace pin --config configs/models/olmo3_rlzero31_math.yaml   # writes configs/pins.lock.yaml
reachtrace render "What is 1+1?"                                   # exact prompt string
reachtrace certify 64                                              # -> 277 zero-success samples needed

# GPU stages (one env works: vllm 0.32 and bergson>=2.3 both accept transformers 5.16-5.19)
uv pip install -e ".[sample,attrib,data]"
reachtrace evals && reachtrace decon-ngram
reachtrace sample --config configs/models/olmo3_rlzero31_math.yaml --side base --evals artifacts/evals/clean.parquet
reachtrace sample --config configs/models/olmo3_rlzero31_math.yaml --side rl   --evals artifacts/evals/clean.parquet
reachtrace partition --config configs/models/olmo3_rlzero31_math.yaml \
  --base-dir artifacts/gen/olmo3_base_vs_rlzero31_math/base/main \
  --rl-dir   artifacts/gen/olmo3_base_vs_rlzero31_math/rl/main
reachtrace synth-corpus
reachtrace bergson-cmds
```

## Running on Modal

GPU stages run on [Modal](https://modal.com) (`pip install -e ".[remote]"`, then `modal token new` once). Jobs read the Hugging Face token from a Modal secret named `hf-token`.

```bash
.venv/bin/python scripts/smoke.py           # tiny-model end-to-end check on an L4 (cents)
.venv/bin/python scripts/pilot.py select    # then: download, sample, escalate, ledger, handoff, report
```

`scripts/pilot.py` logs every job's cost to `artifacts/pilot/costs.jsonl` and refuses to start a GPU step once the running total passes its budget cap.

## Status

CPU-side logic is tested (statistics, grading, ledger, handoff, RL-item scoring, synthetic gate v2). The Modal sampling and ledger jobs have run on Olmo 3 in the exploratory pilot. Bergson attribution and synthetic-gate training have not run yet.

## Author

Govind Arun Kumar, M.S. Applied Machine Learning, University of Maryland, College Park (govind02@umd.edu).
