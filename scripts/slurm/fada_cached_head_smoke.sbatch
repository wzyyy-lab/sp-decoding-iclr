#!/bin/bash
#SBATCH --job-name=fada-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-smoke-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/fada_smoke_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
"$PYTHON" scripts/train_domino_cached_head.py \
  --canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --objective breaker_margin \
  --epochs 1 \
  --batch-size 4 \
  --eval-batch-size 1 \
  --learning-rate 3e-5 \
  --trainable-scope gru_rank \
  --l2sp-weight 1e-3 \
  --max-train-blocks 64 \
  --max-eval-blocks 64 \
  --bootstrap-samples 500 \
  --output "$OUTPUT"
