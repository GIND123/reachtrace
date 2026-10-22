# reachtrace

**Where do RL reasoning gains come from?** Provenance attribution of RLVR on the fully open Olmo 3 model flow.

When RL-Zero training lets `allenai/Olmo-3.1-7B-RL-Zero-Math` solve problems that `allenai/Olmo-3-1025-7B` cannot reach, does that capability trace back to:

- **(H1) elicitation**: the same pretraining and midtraining regions that support problems the base model can already reach;
- **(H2′) acquisition**: the RL-stage signal itself (Dolci prompts and on-policy rollouts); or
- **(H3) composition**: new joint structure over regions that already supported the parts?

The project uses three kinds of evidence, ordered from cheapest to most fragile. Each one can be published on its own.

1. **Behavioural instrument.** Decontaminated R/U/L/N reachability partitions with *certified* reachability bounds, plus cue-forcing and fork-token controls.
2. **Checkpoint provenance (no attribution).** Where a problem becomes reachable along the released flow: end of pretraining (`stage1-step1413814`), end of midtraining (`stage2-step47684`), three midtraining data-mix ablation arms, and 28 RL-Zero step checkpoints (`step_0100` … `step_2800`).
3. **Gradient attribution.** Stage-level and bin-level attribution with Bergson, run only after a synthetic planted-fragment gate shows it recovers known causes under GRPO with the BehaviorTrace confound controls.

## Documents

| File | What it is |
|---|---|
| [docs/01_idea_analysis.md](docs/01_idea_analysis.md) | Critical review of the v2 plan: what is novel after the July–Oct 2026 literature, what breaks, and the repositioned project |
| [docs/02_literature_survey.md](docs/02_literature_survey.md) | Verified literature survey (every arXiv ID resolved on 2026-10-22) |
| [docs/03_implementation_plan.md](docs/03_implementation_plan.md) | Work packages, specifications, gates and decision rules, with no calendar |
| [docs/04_verification_log.md](docs/04_verification_log.md) | Facts checked against the Hub, PyPI, the Bergson source and the Olmo 3 report, with how each was checked |
| [prereg/analysis_plan.md](prereg/analysis_plan.md) | Pre-registration template, to be frozen and hashed before any 7B attribution |

## Layout

```
src/reachtrace/
  prompts.py          RL-Zero templates, byte-identical for base and RL (tested against the HF Jinja)
  registry.py         revision -> SHA pinning; stage-final / midtrain-arm / RL-step branch parsing
  cache.py            content-addressed generation/index/score keys, resumable manifests
  data/               eval sets (unified schema), Dolci RL items (Dr. GRPO advantages), corpus sampling
  decon/              n-gram scan for RL references; wrapper for allenai/decon
  sample/             vLLM runner, adaptive sequential sampling, Answer:-line grading
  reach/              pass@k, Clopper-Pearson certification, partitions, checkpoint provenance
  attrib/             Bergson commands, fork-token targets, bin/stage aggregation, controls, co-influence
  synthetic/          planted-fragment corpus, composition tasks, CPT + GRPO, leave-out / LDS ground truth
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

## Status

The repo is scaffolding with tested CPU-side logic: statistics, partitions, grading, prompt fidelity, aggregation, co-influence, the synthetic task generator and the command builders. GPU paths (vLLM sampling, Bergson builds, CPT/GRPO) are written against verified APIs but have not run yet. Phase 0 in the implementation plan is the first GPU run.

## Author

Govind Arun Kumar, M.S. Applied Machine Learning, University of Maryland, College Park (govind02@umd.edu).
