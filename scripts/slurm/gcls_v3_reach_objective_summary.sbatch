#!/bin/bash
#SBATCH --job-name=gcls-v3-robjsum
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v3-robjsum-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v3-robjsum-%j.err

set -euo pipefail
: "${GCLS_V3_REACH_OBJECTIVE_JOB_ID:?submit with the objective array job id}"

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v3_reach_objective_${GCLS_V3_REACH_OBJECTIVE_JOB_ID}"

cd "$PROJECT"
"$PYTHON" scripts/summarize_gcls_v3_reach_objective.py --run-root "$RUN_ROOT"
