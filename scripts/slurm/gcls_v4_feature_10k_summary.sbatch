#!/bin/bash
#SBATCH --job-name=gcls-v4-f10ksum
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v4-f10ksum-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v4-f10ksum-%j.err

set -euo pipefail
: "${GCLS_V4_FEATURE_10K_JOB_ID:?submit with the 10K array job id}"

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v4_feature_10k_${GCLS_V4_FEATURE_10K_JOB_ID}"

cd "$PROJECT"
"$PYTHON" scripts/summarize_gcls_v4_feature_10k.py --run-root "$RUN_ROOT"
