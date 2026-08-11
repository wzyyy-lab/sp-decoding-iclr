#!/bin/bash
#SBATCH --job-name=gcls-v1-screen
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-5%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-screen-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-screen-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v1_screen_${SLURM_ARRAY_JOB_ID}"
SCOPES=(local causal global local causal global)
LOSSES=(uniform uniform uniform dpace dpace dpace)
SCOPE="${SCOPES[$SLURM_ARRAY_TASK_ID]}"
LOSS="${LOSSES[$SLURM_ARRAY_TASK_ID]}"
LABEL="${SCOPE}_${LOSS}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_global_direct_selector.py \
  --data "$DATA" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/${LABEL}_seed0" \
  --scope "$SCOPE" \
  --candidate-k 16 \
  --model-dim 128 \
  --num-heads 8 \
  --num-layers 2 \
  --batch-size 32 \
  --epochs 10 \
  --learning-rate 0.0006 \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --loss-weighting "$LOSS" \
  --dpace-alpha 0.5 \
  --gradient-clip 1.0 \
  --seed 0 \
  --train-split train \
  --validation-split validation_select \
  --skip-gate \
  --evidence-tier development
