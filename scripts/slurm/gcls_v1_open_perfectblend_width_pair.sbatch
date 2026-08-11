#!/bin/bash
#SBATCH --job-name=gcls-opb-d128
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-opb-d128-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-opb-d128-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
VALIDATION="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
COLLECTION="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v1_open_perfectblend_d128_${SLURM_ARRAY_JOB_ID}"

SCOPES=(local global)
TASK="$SLURM_ARRAY_TASK_ID"
SCOPE="${SCOPES[$TASK]}"
LABEL="axial_${SCOPE}_d128_l2_dpace_pall_seed0"

TRAIN_ARGS=()
for INDEX in {0..7}; do
  PART=$(printf '%03d' "$INDEX")
  TRAIN_ARGS+=(--train-data "$COLLECTION/part-$PART")
done

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=2

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_global_direct_selector.py \
  --data "$VALIDATION" \
  "${TRAIN_ARGS[@]}" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/$LABEL" \
  --scope "$SCOPE" \
  --mixer axial \
  --candidate-k 16 \
  --model-dim 128 \
  --num-heads 8 \
  --num-layers 2 \
  --batch-size 64 \
  --epochs 3 \
  --learning-rate 0.0006 \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --loss-weighting dpace \
  --dpace-alpha 0.5 \
  --gradient-clip 1.0 \
  --seed 0 \
  --train-subset-seed 20260730 \
  --train-split train \
  --validation-split validation_select \
  --skip-gate \
  --calibrate-margin \
  --max-calibration-first-token-drop 0.001 \
  --max-calibration-domain-drop 0 \
  --evidence-tier development
