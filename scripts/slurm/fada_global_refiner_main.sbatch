#!/bin/bash
#SBATCH --job-name=fada-ref-main
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-ref-main-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-ref-main-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/fada_global_refiner_main_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/train_domino_global_refiner.py \
  --canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --objective breaker_margin \
  --epochs 5 \
  --batch-size 16 \
  --eval-batch-size 1 \
  --learning-rate 1e-5 \
  --loss-decay-gamma 7.0 \
  --prefix-weight 1.0 \
  --consistency-weight 0.3 \
  --iterations 1 2 4 6 \
  --bootstrap-samples 5000 \
  --output "$OUTPUT"
