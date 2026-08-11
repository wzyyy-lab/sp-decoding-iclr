#!/bin/bash
#SBATCH --job-name=fada-domino-top16-reach
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-top16-reach-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-top16-reach-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/fada_domino_top16_reachable_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/train_domino_global_refiner.py \
  --canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_opb10k_balanced_v1" \
  --additional-train-canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --eval-canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --architecture direct_selector \
  --selector-dim 256 \
  --selector-heads 8 \
  --selector-layers 3 \
  --selector-mixer axial \
  --selector-node-encoder compatibility \
  --selector-candidates base_topk_plus_released \
  --objective breaker_margin \
  --candidate-topk 16 \
  --epochs 3 \
  --batch-size 32 \
  --eval-batch-size 1 \
  --learning-rate 3e-5 \
  --weight-decay 0 \
  --prefix-weight 0.25 \
  --margin-temperature 1.0 \
  --margin-offset 0.0 \
  --consistency-weight 0 \
  --iterations 1 \
  --bootstrap-samples 5000 \
  --output "$OUTPUT"
