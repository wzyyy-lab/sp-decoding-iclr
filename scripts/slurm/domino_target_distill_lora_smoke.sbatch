#!/bin/bash
#SBATCH --job-name=domino-tdl-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-tdl-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-tdl-smoke-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/domino_target_distill_lora_smoke_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/train_domino_backbone_lora.py \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --train-canonical artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770/part-000 \
  --eval-canonical artifacts/canonical/qwen3_4b_phase3_tier1_10035436 \
  --eval-domino-cache artifacts/canonical/qwen3_4b_domino_phase3_v1 \
  --output "$OUTPUT" \
  --max-train-prompts 32 \
  --max-eval-prompts 8 \
  --epochs 1 \
  --train-blocks-per-prompt 4 \
  --gradient-accumulation-prompts 4 \
  --peak-learning-rate 5e-5 \
  --eval-every-steps 4 \
  --objective target_distill \
  --candidate-topk 16 \
  --target-temperature 2 \
  --target-protect-weight 1 \
  --target-repair-weight 4 \
  --seed 0 \
  --attn-implementation sdpa
