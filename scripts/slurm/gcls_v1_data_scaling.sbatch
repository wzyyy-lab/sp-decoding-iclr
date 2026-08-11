#!/bin/bash
#SBATCH --job-name=gcls-v1-scale
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-3%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-scale-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-scale-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v1_data_scaling_${SLURM_ARRAY_JOB_ID}"

# The subsets are hash-ranked and nested. Epoch counts make all four tasks
# perform approximately 5,000 optimizer updates, isolating data diversity
# from optimization budget.
PROMPTS=(250 500 1000 1987)
EPOCHS=(80 40 20 10)

TASK="$SLURM_ARRAY_TASK_ID"
PROMPT_COUNT="${PROMPTS[$TASK]}"
EPOCH_COUNT="${EPOCHS[$TASK]}"
LABEL="axial_global_d64_dpace_p${PROMPT_COUNT}_seed0"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_global_direct_selector.py \
  --data "$DATA" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/$LABEL" \
  --scope global \
  --mixer axial \
  --candidate-k 16 \
  --model-dim 64 \
  --num-heads 4 \
  --num-layers 1 \
  --batch-size 32 \
  --epochs "$EPOCH_COUNT" \
  --learning-rate 0.0006 \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --loss-weighting dpace \
  --dpace-alpha 0.5 \
  --gradient-clip 1.0 \
  --seed 0 \
  --max-train-prompts "$PROMPT_COUNT" \
  --train-subset-seed 20260730 \
  --train-split train \
  --validation-split validation_select \
  --skip-gate \
  --calibrate-margin \
  --max-calibration-first-token-drop 0.001 \
  --max-calibration-domain-drop 0 \
  --evidence-tier development
