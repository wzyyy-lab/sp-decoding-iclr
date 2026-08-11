#!/bin/bash
#SBATCH --job-name=gcls-v3-robj
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-2%3
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v3-robj-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v3-robj-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
VALIDATION="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
COLLECTION="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_25k_seed20260730"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v3_reach_objective_${SLURM_ARRAY_JOB_ID}"

LABELS=(post1_control post0_hard post0p1_soft)
POST_BREAK_WEIGHTS=(1 0 0.1)
TASK="$SLURM_ARRAY_TASK_ID"
LABEL="${LABELS[$TASK]}"
POST_BREAK_WEIGHT="${POST_BREAK_WEIGHTS[$TASK]}"

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
  --epochs 12 \
  --learning-rate 0.0006 \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --gradient-clip 1.0 \
  --loss-weighting reachable_dpace \
  --dpace-alpha 0.5 \
  --post-break-weight "$POST_BREAK_WEIGHT" \
  --base-safety-weight 0 \
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
