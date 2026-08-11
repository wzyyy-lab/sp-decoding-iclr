#!/bin/bash
#SBATCH --job-name=domino-set-v4
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-set-v4-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-set-v4-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
"$PYTHON" scripts/train_domino_beam_reranker.py \
  --train-canonical artifacts/canonical/qwen3_4b_domino_phase3_v1 \
  --eval-canonical artifacts/canonical/qwen3_4b_domino_phase3_v1 \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "artifacts/training/domino_beam_set_v4_phase3_${SLURM_JOB_ID}" \
  --train-split train \
  --eval-split validation_select \
  --max-train-prompts 1985 \
  --train-blocks-per-prompt 8 \
  --epochs 1 \
  --gradient-accumulation-blocks 4 \
  --candidate-topk 16 \
  --beam-width 16 \
  --normalization full_vocab \
  --reranker-kind set \
  --model-dim 192 \
  --num-heads 6 \
  --num-layers 2 \
  --set-layers 2 \
  --target-feature-dim 0 \
  --base-gamma 0.75 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.04 \
  --list-temperature 0.5 \
  --ranking-regret-weight 0.25 \
  --eval-every-steps 125 \
  --seed 3
