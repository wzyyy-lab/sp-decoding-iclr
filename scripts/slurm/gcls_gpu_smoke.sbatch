#!/bin/bash
#SBATCH --job-name=gcls-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-smoke-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
RUN_ROOT="$PROJECT/artifacts/training/gcls_smoke_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_candidate_selector.py \
  --data "$DATA" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/global_seed0" \
  --scope global \
  --candidate-k 16 \
  --model-dim 128 \
  --token-dim 64 \
  --transition-dim 32 \
  --num-heads 8 \
  --num-layers 2 \
  --batch-size 16 \
  --epochs 2 \
  --learning-rate 0.001 \
  --loss-weighting dpace \
  --rank-weight-power 0.5 \
  --seed 0 \
  --memorization-blocks 64 \
  --skip-gate \
  --evidence-tier capacity_probe
