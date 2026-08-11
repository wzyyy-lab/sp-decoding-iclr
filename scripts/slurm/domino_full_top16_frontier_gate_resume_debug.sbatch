#!/bin/bash
#SBATCH --job-name=domino-fullfront-gate
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-fullfront-gate-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-fullfront-gate-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
INITIAL="$PROJECT/artifacts/training/domino_full_top16_frontier_phase3_10158305/best_adaptation.pt"
OUTPUT="$PROJECT/artifacts/training/domino_full_top16_frontier_gate_resume_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
"$PYTHON" scripts/train_domino_backbone_lora.py \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --initial-adaptation "$INITIAL" \
  --train-canonical artifacts/canonical/qwen3_4b_phase3_tier1_10035436 \
  --eval-canonical artifacts/canonical/qwen3_4b_phase3_tier1_10035436 \
  --eval-domino-cache artifacts/canonical/qwen3_4b_domino_phase3_v1 \
  --output "$OUTPUT" \
  --train-split validation_gate \
  --eval-split validation_select \
  --max-train-prompts 149 \
  --train-blocks-per-prompt 0 \
  --epochs 4 \
  --gradient-accumulation-prompts 16 \
  --peak-learning-rate 3e-6 \
  --eval-every-steps 10 \
  --objective topk_frontier_curriculum \
  --candidate-topk 16 \
  --protection-margin 0.05 \
  --protection-weight 5 \
  --curriculum-final-base-weight 0.25 \
  --curriculum-transition-ratio 0.25 \
  --train-full-backbone \
  --train-causal-head \
  --seed 1 \
  --attn-implementation sdpa
