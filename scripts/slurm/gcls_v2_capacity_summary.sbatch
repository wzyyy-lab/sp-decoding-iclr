#!/bin/bash
#SBATCH --job-name=gcls-v2-capsum
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-capsum-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-capsum-%j.err

set -euo pipefail
: "${GCLS_V2_CAPACITY_JOB_ID:?submit with the completed capacity array job id}"

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v2_capacity_${GCLS_V2_CAPACITY_JOB_ID}"

cd "$PROJECT"
"$PYTHON" scripts/summarize_gcls_v2_capacity.py --run-root "$RUN_ROOT"
