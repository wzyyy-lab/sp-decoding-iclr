#!/bin/bash
#SBATCH --job-name=domino-jtop16-p3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-jtop16-p3-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-jtop16-p3-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/domino_joint_top16_phase3_acc16_${SLURM_JOB_ID}"
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
  --epochs 3 \
  --gradient-accumulation-prompts 16 \
  --peak-learning-rate 3e-5 \
  --eval-every-steps 75 \
  --objective topk_curriculum \
  --candidate-topk 16 \
  --curriculum-final-base-weight 0.5 \
  --curriculum-transition-ratio 0.5 \
  --train-causal-head \
  --seed 0 \
  --attn-implementation sdpa
