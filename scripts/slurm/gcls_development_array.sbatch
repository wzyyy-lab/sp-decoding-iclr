#!/bin/bash
#SBATCH --job-name=gcls-dev
#SBATCH --partition=i64m1tga40u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-dev-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-dev-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
RUN_ROOT="$PROJECT/artifacts/training/gcls_development_${SLURM_ARRAY_JOB_ID}"
SCOPES=(local global)
SCOPE_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
SCOPE="${SCOPES[$SCOPE_INDEX]}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_candidate_selector.py \
  --data "$DATA" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/${SCOPE}_seed${SEED}" \
  --scope "$SCOPE" \
  --candidate-k 16 \
  --model-dim 128 \
  --token-dim 64 \
  --transition-dim 32 \
  --num-heads 8 \
  --num-layers 2 \
  --batch-size 32 \
  --epochs 20 \
  --learning-rate 0.0003 \
  --loss-weighting dpace \
  --rank-weight-power 0.5 \
  --seed "$SEED" \
  --train-split train \
  --validation-split validation_select \
  --gate-split validation_gate \
  --max-first-token-drop 0.001 \
  --max-domain-drop 0.05 \
  --evidence-tier development
