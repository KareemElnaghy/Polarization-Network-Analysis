#!/usr/bin/env bash
# One-time setup for the UBU machine (Ubuntu + NVIDIA GPU). Idempotent.
# Usage: bash ops/UBU/setup.sh [REPO_DIR]   (default: ~/Polarization-Network-Analysis)
set -euo pipefail
REPO_URL="https://github.com/KareemElnaghy/Polarization-Network-Analysis.git"
REPO_DIR="${1:-$HOME/Polarization-Network-Analysis}"
BRANCH="llm_implementation"

echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "nvidia-smi failed: install the NVIDIA driver first"; exit 1; }
free -h | head -2; lscpu | grep -E "Model name|^CPU\(s\)"

# Python 3.12 (deadsnakes) if not present
if ! command -v python3.12 >/dev/null; then
  sudo apt-get update && sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3.12-dev git build-essential
fi

[ -d "$REPO_DIR/.git" ] || git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
git fetch origin && git checkout "$BRANCH" && git pull --ff-only origin "$BRANCH"

[ -d .venv ] || python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
# CUDA 12.1 wheels; change the index URL if the driver is older (see nvidia-smi "CUDA Version")
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install transformers datasets accelerate sentence-transformers sentencepiece protobuf

python - <<'PY'
import torch, transformers, sentence_transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "sentence-transformers", sentence_transformers.__version__)
assert torch.cuda.is_available(), "CUDA not available"
PY
echo "git SHA: $(git rev-parse --short HEAD)"
echo "OK. Next: copy data/llm/distill/distill_v1.parquet here (see MACHINES.md) and run the training script."
