#!/bin/bash
#SBATCH --job-name=gcls-v1-reg
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-7%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-reg-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-reg-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v1_regularization_${SLURM_ARRAY_JOB_ID}"

LABELS=(
  global_d32_plain_dpace
  global_d32_reg_dpace
  global_d64_reg_dpace
  global_d128_reg_dpace
  local_d32_reg_dpace
  causal_d32_reg_dpace
  global_d32_reg_uniform
  global_d16_reg_dpace
)
SCOPES=(global global global global local causal global global)
DIMS=(32 32 64 128 32 32 32 16)
HEADS=(4 4 4 8 4 4 4 4)
LAYERS=(1 1 1 2 1 1 1 1)
DROPOUTS=(0 0.1 0.1 0.1 0.1 0.1 0.1 0.1)
DECAYS=(0 0.01 0.01 0.01 0.01 0.01 0.01 0.01)
LOSSES=(dpace dpace dpace dpace dpace dpace uniform dpace)

TASK="$SLURM_ARRAY_TASK_ID"
LABEL="${LABELS[$TASK]}"
SCOPE="${SCOPES[$TASK]}"
DIM="${DIMS[$TASK]}"
HEAD="${HEADS[$TASK]}"
LAYER="${LAYERS[$TASK]}"
DROPOUT="${DROPOUTS[$TASK]}"
DECAY="${DECAYS[$TASK]}"
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
  --candidate-k 16 \
  --model-dim "$DIM" \
  --num-heads "$HEAD" \
  --num-layers "$LAYER" \
  --dropout "$DROPOUT" \
  --batch-size 32 \
  --epochs 10 \
  --learning-rate 0.0006 \
  --weight-decay "$DECAY" \
  --warmup-ratio 0.04 \
  --loss-weighting "$LOSS" \
  --dpace-alpha 0.5 \
  --gradient-clip 1.0 \
  --seed 0 \
  --train-split train \
  --validation-split validation_select \
  --skip-gate \
  --evidence-tier development
