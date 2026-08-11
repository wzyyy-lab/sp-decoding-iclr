#!/bin/bash
#SBATCH --job-name=domino-beam-rerank-v3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-beam-rerank-v3-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-beam-rerank-v3-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/domino_beam_reranker_v3_opb10k_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
"$PYTHON" scripts/train_domino_beam_reranker.py \
  --train-canonical artifacts/canonical/qwen3_4b_domino_opb10k_balanced_v1 \
  --eval-canonical artifacts/canonical/qwen3_4b_domino_phase3_v1 \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$OUTPUT" \
  --train-split train \
  --eval-split validation_select \
  --max-train-prompts 9900 \
  --train-blocks-per-prompt 2 \
  --epochs 1 \
  --gradient-accumulation-blocks 8 \
  --candidate-topk 16 \
  --beam-width 64 \
  --model-dim 128 \
  --num-heads 4 \
  --num-layers 2 \
  --learning-rate 3e-5 \
  --list-temperature 0.5 \
  --list-weight 0 \
  --frontier-weight 1 \
  --eval-every-steps 250 \
  --seed 2
