#!/bin/bash
#SBATCH --job-name=gcls-v2-obj
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-obj-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-obj-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
VALIDATION="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
COLLECTION="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v2_objective_${SLURM_ARRAY_JOB_ID}"

LABELS=(historical_dpace_a0p5 reach_lam0 reach_lam0p1 reach_lam0p25)
LOSSES=(dpace accepted_reach accepted_reach accepted_reach)
SAFETY_WEIGHTS=(0 0 0.1 0.25)

TASK="$SLURM_ARRAY_TASK_ID"
LABEL="${LABELS[$TASK]}"
LOSS="${LOSSES[$TASK]}"
SAFETY_WEIGHT="${SAFETY_WEIGHTS[$TASK]}"

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
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_global_direct_selector.py \
  --data "$VALIDATION" \
  "${TRAIN_ARGS[@]}" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/${LABEL}_seed0" \
  --scope global \
  --mixer axial \
  --node-encoder additive \
  --candidate-k 16 \
  --model-dim 64 \
  --num-heads 4 \
  --num-layers 1 \
  --dropout 0 \
  --batch-size 64 \
  --epochs 9 \
  --learning-rate 0.0006 \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --gradient-clip 1.0 \
  --loss-weighting "$LOSS" \
  --dpace-alpha 0.5 \
  --base-safety-weight "$SAFETY_WEIGHT" \
  --base-safety-margin 0.1 \
  --seed 0 \
  --max-train-prompts 25000 \
  --train-subset-seed 20260730 \
  --train-split train \
  --validation-split validation_select \
  --skip-gate \
  --calibrate-margin \
  --max-calibration-first-token-drop 0.001 \
  --max-calibration-domain-drop 0 \
  --evidence-tier development
