#!/bin/bash
#SBATCH --job-name=domino-reach-12k
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-reach-12k-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-reach-12k-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/domino_backbone_lora_reachable_opb12k_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/train_domino_backbone_lora.py \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --train-canonical artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770/part-000 \
  --eval-canonical artifacts/canonical/qwen3_4b_phase3_tier1_10035436 \
  --eval-domino-cache artifacts/canonical/qwen3_4b_domino_phase3_v1 \
  --output "$OUTPUT" \
  --max-train-prompts 12000 \
  --epochs 1 \
  --peak-learning-rate 1e-4 \
  --eval-every-steps 250 \
  --objective greedy_reachable \
  --feature-anchor-weight 0 \
  --seed 0 \
  --attn-implementation sdpa
