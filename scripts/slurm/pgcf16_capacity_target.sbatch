#!/bin/bash
#SBATCH --job-name=pgcf16-cap-t
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pgcf16-cap-target-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pgcf16-cap-target-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
TRAIN="$PROJECT/artifacts/canonical/r047_anchor_t4_train_10164718"
OUT="$PROJECT/artifacts/models/pgcf16_capacity_target_${SLURM_JOB_ID}"

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
  --loss-mode curriculum \
  --train-split train \
  --same-subset \
  --max-train-records 512 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --max-steps 8000 \
  --eval-every-steps 200 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --gradient-clip 1 \
  --seed 0 \
  --model-dim 256 \
  --num-heads 8 \
  --num-layers 2 \
  --ff-multiplier 2 \
  --require-cuda \
  --require-default-parameter-count \
  --capacity-gate-mode target \
  --require-capacity-gate
