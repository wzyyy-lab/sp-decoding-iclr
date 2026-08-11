#!/bin/bash
#SBATCH --job-name=gcls-v4-f10k
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v4-f10k-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v4-f10k-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
VALIDATION="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
COLLECTION="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_25k_seed20260730"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v4_feature_10k_${SLURM_ARRAY_JOB_ID}"

LABELS=(compact_axial_additive_d64_seed0 probe_flat_compat_d640_seed0)
MIXERS=(axial flat)
NODE_ENCODERS=(additive compatibility)
MODEL_DIMS=(64 640)
HEADS=(4 10)
LAYERS=(1 4)
LEARNING_RATES=(0.0006 0.0003)
TASK="$SLURM_ARRAY_TASK_ID"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_global_direct_selector.py \
  --data "$VALIDATION" \
  --train-data "$COLLECTION" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/${LABELS[$TASK]}" \
  --scope global \
  --mixer "${MIXERS[$TASK]}" \
  --node-encoder "${NODE_ENCODERS[$TASK]}" \
  --candidate-k 16 \
  --model-dim "${MODEL_DIMS[$TASK]}" \
  --num-heads "${HEADS[$TASK]}" \
  --num-layers "${LAYERS[$TASK]}" \
  --dropout 0 \
  --batch-size 64 \
  --epochs 30 \
  --learning-rate "${LEARNING_RATES[$TASK]}" \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --gradient-clip 1.0 \
  --loss-weighting candidate_dpace \
  --dpace-alpha 0.5 \
  --base-safety-weight 0 \
  --seed 0 \
  --max-train-prompts 10000 \
  --train-subset-seed 20260730 \
  --train-split train \
  --validation-split validation_select \
  --skip-gate \
  --calibrate-margin \
  --max-calibration-first-token-drop 0.001 \
  --max-calibration-domain-drop 0 \
  --evidence-tier development
