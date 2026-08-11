#!/bin/bash
#SBATCH --job-name=gcls-summary
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-summary-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-summary-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
: "${DEV_ARRAY_JOB_ID:?submit with DEV_ARRAY_JOB_ID exported}"
RUN_ROOT="$PROJECT/artifacts/training/gcls_development_${DEV_ARRAY_JOB_ID}"

cd "$PROJECT"
"$PYTHON" scripts/summarize_candidate_selector.py \
  --input "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.json" \
  --seeds 3 \
  --minimum-global-base-delta 0.2 \
  --minimum-global-local-delta 0.15 \
  --maximum-first-token-drop 0.001 \
  --maximum-domain-drop 0.05
