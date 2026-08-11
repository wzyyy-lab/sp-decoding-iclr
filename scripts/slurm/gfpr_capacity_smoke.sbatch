#!/bin/bash
#SBATCH --job-name=gfpr-cap-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-cap-smoke-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-cap-smoke-%A_%a.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
ROLLOUT="$PROJECT/artifacts/canonical/gfpr_dynamic_smoke_10163938"
OUT="$PROJECT/artifacts/models/gfpr_capacity_smoke_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
EXTRA=()
ARM=frozen15
if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
  ARM=dynamic16
  EXTRA+=(--correct-position-zero)
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

echo "GFPR capacity smoke arm=$ARM output=$OUT"
"$PYTHON" scripts/train_gfpr_head.py \
  --train-rollout "$ROLLOUT" \
  --eval-rollout "$ROLLOUT" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$OUT" \
  --train-split train \
  --eval-split train \
  --epochs 20 \
  --prompts-per-batch 1 \
  --gradient-accumulation-prompts 4 \
  --eval-batch-size 1 \
  --capacity-only-allow-overlap \
  --learning-rate 1e-4 \
  --delta-l2 1e-6 \
  --eval-every-steps 5 \
  --bootstrap-samples 2000 \
  --seed 0 \
  "${EXTRA[@]}"
