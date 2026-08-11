#!/bin/bash
#SBATCH --job-name=domino-fullmargin-p3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-fullmargin-p3-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-fullmargin-p3-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/domino_full_frontier_margin_phase3_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
"$PYTHON" scripts/train_domino_backbone_lora.py \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --train-canonical artifacts/canonical/qwen3_4b_phase3_tier1_10035436 \
  --eval-canonical artifacts/canonical/qwen3_4b_phase3_tier1_10035436 \
  --eval-domino-cache artifacts/canonical/qwen3_4b_domino_phase3_v1 \
  --output "$OUTPUT" \
  --max-train-prompts 1986 \
  --epochs 1 \
  --gradient-accumulation-prompts 16 \
  --peak-learning-rate 1e-5 \
  --eval-every-steps 25 \
  --objective frontier_margin \
  --protection-margin 0.05 \
  --protection-weight 10 \
  --train-full-backbone \
  --train-causal-head \
  --seed 0 \
  --attn-implementation sdpa
