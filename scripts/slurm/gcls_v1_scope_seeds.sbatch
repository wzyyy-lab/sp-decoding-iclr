#!/bin/bash
#SBATCH --job-name=gcls-v1-seeds
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
# Submit task 8 separately with ``sbatch --array=8`` after one slot clears;
# this cluster's QOS permits at most eight submitted array elements per user.
#SBATCH --array=0-7%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-seeds-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-seeds-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v1_scope_seeds_${SLURM_ARRAY_JOB_ID}"

SCOPES=(local local local causal causal causal global global global)
SEEDS=(0 1 2 0 1 2 0 1 2)

TASK="$SLURM_ARRAY_TASK_ID"
SCOPE="${SCOPES[$TASK]}"
SEED="${SEEDS[$TASK]}"
LABEL="axial_${SCOPE}_d64_dpace_seed${SEED}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_global_direct_selector.py \
  --data "$DATA" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/$LABEL" \
  --scope "$SCOPE" \
  --mixer axial \
  --candidate-k 16 \
  --model-dim 64 \
  --num-heads 4 \
  --num-layers 1 \
  --batch-size 32 \
  --epochs 10 \
  --learning-rate 0.0006 \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --loss-weighting dpace \
  --dpace-alpha 0.5 \
  --gradient-clip 1.0 \
  --seed "$SEED" \
  --train-split train \
  --validation-split validation_select \
  --skip-gate \
  --calibrate-margin \
  --max-calibration-first-token-drop 0.001 \
  --max-calibration-domain-drop 0 \
  --evidence-tier development
