#!/bin/bash
#SBATCH --job-name=gcls-v1-axcap
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-3%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-axcap-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v1-axcap-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v1_axial_capacity_${SLURM_ARRAY_JOB_ID}"

LABELS=(global_d32_uniform global_d32_dpace global_d128_uniform global_d128_dpace)
DIMS=(32 32 128 128)
HEADS=(4 4 8 8)
LAYERS=(1 1 2 2)
LOSSES=(uniform dpace uniform dpace)
EPOCHS=(120 120 80 80)

TASK="$SLURM_ARRAY_TASK_ID"
LABEL="${LABELS[$TASK]}"
DIM="${DIMS[$TASK]}"
HEAD="${HEADS[$TASK]}"
LAYER="${LAYERS[$TASK]}"
LOSS="${LOSSES[$TASK]}"
EPOCH="${EPOCHS[$TASK]}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_global_direct_selector.py \
  --data "$DATA" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/${LABEL}_seed0" \
  --scope global \
  --mixer axial \
  --candidate-k 16 \
  --model-dim "$DIM" \
  --num-heads "$HEAD" \
  --num-layers "$LAYER" \
  --batch-size 32 \
  --epochs "$EPOCH" \
  --learning-rate 0.0006 \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --loss-weighting "$LOSS" \
  --dpace-alpha 0.5 \
  --gradient-clip 1.0 \
  --seed 0 \
  --memorization-blocks 512 \
  --memorization-opportunity-fraction 0.5 \
  --skip-gate \
  --evidence-tier capacity_probe \
  --require-capacity-gate \
  --min-candidate-accuracy 0.99 \
  --min-hard-candidate-accuracy 0.97 \
  --min-first-miss-repair-rate 0.95 \
  --min-oracle-gap-recovered 0.95 \
  --max-harmed-fraction 0.01
