#!/bin/bash
#SBATCH --job-name=gcls-v2-capres
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-2%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-capres-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-capres-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v2_capacity_rescue_${SLURM_ARRAY_JOB_ID}"

LABELS=(compat_arr_budget compat_cdpace05 additive_cdpace05)
NODE_ENCODERS=(compatibility compatibility additive)
LOSSES=(accepted_reach candidate_dpace candidate_dpace)

TASK="$SLURM_ARRAY_TASK_ID"
LABEL="${LABELS[$TASK]}"
NODE_ENCODER="${NODE_ENCODERS[$TASK]}"
LOSS="${LOSSES[$TASK]}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_global_direct_selector.py \
  --data "$DATA" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/${LABEL}_seed0" \
  --scope global \
  --mixer flat \
  --node-encoder "$NODE_ENCODER" \
  --candidate-k 16 \
  --model-dim 128 \
  --num-heads 8 \
  --num-layers 2 \
  --dropout 0 \
  --batch-size 32 \
  --epochs 320 \
  --learning-rate 0.0006 \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --gradient-clip 1.0 \
  --loss-weighting "$LOSS" \
  --dpace-alpha 0.5 \
  --base-safety-weight 0 \
  --base-safety-margin 0.1 \
  --seed 0 \
  --memorization-blocks 128 \
  --memorization-opportunity-fraction 0.5 \
  --skip-gate \
  --evidence-tier capacity_probe \
  --min-candidate-accuracy 0.99 \
  --min-hard-candidate-accuracy 0.97 \
  --min-first-miss-repair-rate 0.95 \
  --min-oracle-gap-recovered 0.95 \
  --max-harmed-fraction 0.01
