#!/usr/bin/env bash
# One-time setup for the MAC machine (Apple Silicon, MPS only). Idempotent.
# Usage: bash ops/MAC/setup.sh [REPO_DIR]   (default: ~/Polarization-Network-Analysis)
set -euo pipefail
REPO_URL="https://github.com/KareemElnaghy/Polarization-Network-Analysis.git"
REPO_DIR="${1:-$HOME/Polarization-Network-Analysis}"
BRANCH="llm_implementation"

sysctl -n machdep.cpu.brand_string; sysctl -n hw.memsize | awk '{printf "RAM %.0f GB\n", $1/1073741824}'

PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
echo "using $PY ($($PY --version))"
[ -d "$REPO_DIR/.git" ] || git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
git fetch origin && git checkout "$BRANCH" && git pull --ff-only origin "$BRANCH"

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e .
# CPU/MPS analysis only; sentence-transformers is for optional small embedding jobs
pip install torch sentence-transformers

python - <<'PY'
import torch
print("torch", torch.__version__, "mps", torch.backends.mps.is_available())
PY
echo "git SHA: $(git rev-parse --short HEAD)"
echo "OK. Next: copy the analysis exports and run folders here (see MACHINES.md)."
