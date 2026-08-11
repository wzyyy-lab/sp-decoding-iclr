#!/bin/bash
#SBATCH --job-name=fada-domino-opb215k-head
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-opb215k-head-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-opb215k-head-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"

SCOPES=(gru_rank full_head)
LRS=(3e-5 1e-5)
SCOPE=${SCOPES[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
OUTPUT="$PROJECT/artifacts/training/fada_domino_opb215k_${SCOPE}_${SLURM_ARRAY_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/train_domino_cached_head.py \
  --canonical "$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_25k_seed20260730" \
  --additional-train-canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --eval-canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --objective decay_ce \
  --epochs 3 \
  --batch-size 64 \
  --eval-batch-size 1 \
  --learning-rate "$LR" \
  --trainable-scope "$SCOPE" \
  --l2sp-weight 1e-3 \
  --bootstrap-samples 5000 \
  --output "$OUTPUT"
