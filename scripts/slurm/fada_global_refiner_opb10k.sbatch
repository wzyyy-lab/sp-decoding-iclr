#!/bin/bash
#SBATCH --job-name=fada-ref-opb10k
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-ref-opb10k-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-ref-opb10k-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/fada_global_refiner_opb10k_${SLURM_JOB_ID}"

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
  --objective decay_ce \
  --candidate-topk 16 \
  --epochs 3 \
  --batch-size 64 \
  --eval-batch-size 1 \
  --learning-rate 1e-5 \
  --loss-decay-gamma 7.0 \
  --consistency-weight 0.3 \
  --iterations 1 2 4 6 \
  --bootstrap-samples 5000 \
  --output "$OUTPUT"
