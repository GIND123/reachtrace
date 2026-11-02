#!/usr/bin/env bash
# Separate venv for sampling with vLLM on Lambda's CUDA 12.8 driver. vLLM 0.11.0 pins
# torch 2.8.0, whose PyPI wheel is built for CUDA 12.8; later vLLM needs torch >= 2.9.
# It maps Olmo3ForCausalLM to its Olmo2 implementation. Used by olmo7b_midtrain.py rollouts.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/vllm_env ] || uv venv --python 3.11 ~/vllm_env
VIRTUAL_ENV=~/vllm_env uv pip install vllm==0.11.0 transformers==4.57.1 pandas pyarrow
~/vllm_env/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
