#!/bin/bash
#SBATCH --job-name=sph-train-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-train-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-train-smoke-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_domino_smoke_10022412"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_survival_head.py \
  --data "$DATA" \
  --target "$ASSETS/models/Qwen3-4B" \
  --candidate-k 16 \
  --head-type bidirectional \
  --normalization local \
  --rank 32 \
  --model-dim 64 \
  --num-heads 4 \
  --batch-size 3 \
  --epochs 30 \
  --learning-rate 0.001 \
  --train-split train \
  --validation-split test \
  --test-split test \
  --evidence-tier plumbing_smoke \
  --output "$PROJECT/artifacts/training/smoke_${SLURM_JOB_ID}"
