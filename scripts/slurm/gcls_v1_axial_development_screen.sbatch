#!/bin/bash
#SBATCH --job-name=gcls-v1-axdev
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-7%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-axdev-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-axdev-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v1_axial_screen_${SLURM_ARRAY_JOB_ID}"

LABELS=(
  local_d32_dpace
  causal_d32_dpace
  global_d32_dpace
  local_d64_dpace
  causal_d64_dpace
  global_d64_dpace
  global_d128_dpace
  global_d32_uniform
)
SCOPES=(local causal global local causal global global global)
DIMS=(32 32 32 64 64 64 128 32)
HEADS=(4 4 4 4 4 4 8 4)
LAYERS=(1 1 1 1 1 1 2 1)
LOSSES=(dpace dpace dpace dpace dpace dpace dpace uniform)

TASK="$SLURM_ARRAY_TASK_ID"
LABEL="${LABELS[$TASK]}"
SCOPE="${SCOPES[$TASK]}"
DIM="${DIMS[$TASK]}"
HEAD="${HEADS[$TASK]}"
LAYER="${LAYERS[$TASK]}"
LOSS="${LOSSES[$TASK]}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_global_direct_selector.py \
  --data "$DATA" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/${LABEL}_seed0" \
  --scope "$SCOPE" \
  --mixer axial \
  --candidate-k 16 \
  --model-dim "$DIM" \
  --num-heads "$HEAD" \
  --num-layers "$LAYER" \
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
