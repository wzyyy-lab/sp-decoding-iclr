#!/bin/bash
#SBATCH --job-name=gfpr-gateb-train-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-gateb-train-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-gateb-train-smoke-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/train_gfpr_head.py \
  --train-rollout "$PROJECT/artifacts/canonical/gfpr_phase3_train_dynamic_10163980" \
  --eval-rollout "$PROJECT/artifacts/canonical/gfpr_validation_select_fixed_10163965" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$PROJECT/artifacts/models/gfpr_gateb_train_smoke_${SLURM_JOB_ID}" \
  --train-split train \
  --eval-split validation_select \
  --max-train-prompts 8 \
  --max-eval-prompts 8 \
  --epochs 1 \
  --prompts-per-batch 1 \
  --gradient-accumulation-prompts 4 \
  --eval-batch-size 1 \
  --learning-rate 3e-5 \
  --eval-every-steps 1 \
  --bootstrap-samples 200 \
  --seed 0
