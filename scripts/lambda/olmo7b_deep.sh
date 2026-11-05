#!/usr/bin/env bash
# scripts/olmo7b_deep.py on Lambda, in two phases that can run on different instances:
#   sample     (any GPU that fits vLLM, e.g. A100 40 GB): both environments, rollouts with
#              one vLLM engine per GPU while the documents download, queries
#   attribute  (80 GB GPU: a full backward pass on a 7.9k-token query row needs ~60 GB):
#              TrackStar over all chunks; push artifacts/olmo7b_deep first
# Artifacts in ~/reachtrace/artifacts/olmo7b_deep.
set -euo pipefail
cd ~/reachtrace
mkdir -p ~/logs
export DEEP_ART=~/reachtrace/artifacts/olmo7b_deep
case "${1:-sample}" in
  sample)
    export PATH="$HOME/.local/bin:$PATH"  # both setups need uv; install it once, first
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    bash scripts/lambda/setup_remote.sh > ~/logs/setup_remote.log 2>&1 &
    bash scripts/lambda/setup_vllm.sh > ~/logs/setup_vllm.log 2>&1 &
    wait
    source ~/rt_env.sh
    python scripts/olmo7b_deep.py problems
    python scripts/olmo7b_deep.py docs > ~/logs/docs.log 2>&1 &
    DOCS=$!
    python scripts/olmo7b_deep.py rollouts
    python scripts/olmo7b_deep.py queries
    wait $DOCS
    ;;
  attribute)
    bash scripts/lambda/setup_remote.sh > ~/logs/setup_remote.log 2>&1
    source ~/rt_env.sh
    python scripts/olmo7b_deep.py attribute
    ;;
esac
echo "DEEP_DONE $1"
