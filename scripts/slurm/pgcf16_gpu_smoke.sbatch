#!/bin/bash
#SBATCH --job-name=pgcf16-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pgcf16-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pgcf16-smoke-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
TRAIN="$PROJECT/artifacts/canonical/r047_anchor_t4_train_10164718"
OUT="$PROJECT/artifacts/models/pgcf16_smoke_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

nvidia-smi
"$PYTHON" scripts/train_pgcf16.py \
  --train-rollout "$TRAIN" \
  --eval-rollout "$TRAIN" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$OUT" \
  --head global \
  --train-split train \
  --same-subset \
  --max-train-records 32 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --max-steps 20 \
  --eval-every-steps 5 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --gradient-clip 1 \
  --seed 0 \
  --model-dim 256 \
  --num-heads 8 \
  --num-layers 2 \
  --ff-multiplier 2 \
  --require-cuda \
  --require-default-parameter-count
