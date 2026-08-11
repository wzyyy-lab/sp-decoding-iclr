#!/bin/bash
#SBATCH --job-name=tuned-l35-pilot
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/tuned-l35-pilot-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/tuned-l35-pilot-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
"$PYTHON" scripts/train_domino_tuned_target_exit.py \
  --canonical artifacts/canonical/qwen3_4b_domino_phase3_v1 \
  --source-canonical artifacts/canonical/qwen3_4b_phase3_tier1_10035436 \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "artifacts/training/domino_tuned_exit_l35_pilot_${SLURM_JOB_ID}" \
  --train-split train \
  --eval-split validation_select \
  --early-layers 35 \
  --rank 512 \
  --candidate-topk 16 \
  --beam-width 16 \
  --max-train-prompts 500 \
  --train-blocks-per-prompt 2 \
  --epochs 1 \
  --gradient-accumulation-blocks 8 \
  --learning-rate 3e-4 \
  --warmup-ratio 0.04 \
  --position-discount 0.95 \
  --seed 5
