#!/bin/bash
#SBATCH --job-name=pgcf16-screen
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pgcf16-screen-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pgcf16-screen-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
TRAIN="$PROJECT/artifacts/canonical/r047_anchor_t4_train_10164718"
EVAL="$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_10164718"
HEADS=(global local)
HEAD="${HEADS[$SLURM_ARRAY_TASK_ID]}"
OUT="$PROJECT/artifacts/models/pgcf16_r047_screen_${SLURM_ARRAY_JOB_ID}/${HEAD}_seed0"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

nvidia-smi
"$PYTHON" scripts/train_pgcf16.py \
  --train-rollout "$TRAIN" \
  --eval-rollout "$EVAL" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$OUT" \
  --head "$HEAD" \
  --loss-mode curriculum \
  --train-split train \
  --eval-split validation_select \
  --train-diagnostic-records 512 \
  --train-diagnostic-stride 31 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --max-steps 20000 \
  --eval-every-steps 500 \
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
