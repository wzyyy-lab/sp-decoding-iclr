#!/bin/bash
#SBATCH --job-name=fada-domino-top16-pre-reach
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-top16-pre-reach-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-top16-pre-reach-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/fada_domino_top16_pretrained_reach_${SLURM_JOB_ID}"
INIT="$PROJECT/artifacts/training/gcls_v4_feature_100k_10132819/compact_axial_additive_d64_full_seed0/best.pt"

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
  --selector-dim 64 \
  --selector-heads 4 \
  --selector-layers 1 \
  --selector-mixer axial \
  --selector-node-encoder additive \
  --selector-candidates base_topk_plus_released \
  --selector-init-checkpoint "$INIT" \
  --selector-reset-output \
  --selector-freeze-encoder \
  --objective breaker_margin \
  --candidate-topk 16 \
  --epochs 3 \
  --batch-size 64 \
  --eval-batch-size 1 \
  --learning-rate 3e-4 \
  --weight-decay 0 \
  --prefix-weight 0.25 \
  --margin-temperature 1.0 \
  --margin-offset 0.0 \
  --consistency-weight 0 \
  --iterations 1 \
  --bootstrap-samples 5000 \
  --output "$OUTPUT"
