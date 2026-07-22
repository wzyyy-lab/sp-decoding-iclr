#!/bin/bash
#SBATCH --job-name=sph-head-audit
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-head-audit-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-head-audit-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/analysis/head_checkpoint_audit_${SLURM_JOB_ID}.json"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/audit_head_checkpoints.py \
  --runs-root "$PROJECT/artifacts/training/probe_10035142" \
  --runs-root "$PROJECT/artifacts/training/probe_survival_aux_10035188" \
  --output "$OUTPUT"
