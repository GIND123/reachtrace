#!/usr/bin/env bash
# One-time environment on a fresh Lambda instance (Ubuntu, NVIDIA driver preinstalled).
# Installs uv and a Python 3.11 venv with torch, bergson 2.7.0, TRL and the repo itself.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
cd ~/reachtrace
[ -d ~/venv ] || uv venv --python 3.11 ~/venv
source ~/venv/bin/activate
uv pip install -e ".[dev]" "bergson==2.7.0" "trl>=1.16" "transformers>=5.16,<5.20" \
  datasets matplotlib
# The default torch wheel targets CUDA 13; drivers older than 580 need the cu128 build
# (installed last, so nothing above replaces it).
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
if [ "$DRIVER_MAJOR" -lt 580 ]; then
  uv pip install --reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import torch, bergson, trl; print(torch.__version__, torch.cuda.is_available(), bergson.__version__, trl.__version__)"
# Non-interactive ssh shells skip ~/.bashrc on Ubuntu; remote commands source this instead.
echo 'source ~/venv/bin/activate; export HF_HOME=~/hf PATH=$HOME/.local/bin:$PATH' > ~/rt_env.sh
