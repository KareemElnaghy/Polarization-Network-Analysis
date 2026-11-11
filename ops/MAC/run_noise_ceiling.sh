#!/usr/bin/env bash
# P1.2 on the MacBook: split seeds 0-9 (WIN runs 10-19). Needs only the analysis bundle (exports/analysis_inputs), no data tree.
# Usage: bash ops/MAC/run_noise_ceiling.sh [SPLITS=0-9] [WORKERS=6]    Stop cleanly: touch runs/noise_ceiling/STOP    Resume: rerun the same command.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
SPLITS="${1:-0-9}"; WORKERS="${2:-6}"
[ -f exports/analysis_inputs/SHA256SUMS.txt ] || { echo "missing exports/analysis_inputs (unzip analysis_inputs.zip into exports/)"; exit 1; }
(cd exports/analysis_inputs && shasum -a 256 -c SHA256SUMS.txt)
mkdir -p logs_llm
nohup caffeinate -i python scripts/final/noise_ceiling.py --inputs exports/analysis_inputs --splits "$SPLITS" --R 20 --workers "$WORKERS" --out runs/noise_ceiling > "logs_llm/noise_ceiling_MAC.log" 2>&1 &
echo "started; tail -f logs_llm/noise_ceiling_MAC.log   (outputs: runs/noise_ceiling/split_XX_R20.csv; email or Drive these back)"
