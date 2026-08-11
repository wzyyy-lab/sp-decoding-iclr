#!/bin/bash
#SBATCH --job-name=gfpr-cand-cap
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-cand-cap-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-cand-cap-%A_%a.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
ROLLOUT="$PROJECT/artifacts/canonical/gfpr_dynamic_smoke_10163938"
SCOPE=calibration
if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
  SCOPE=input_rank
elif [[ "$SLURM_ARRAY_TASK_ID" == "2" ]]; then
  SCOPE=gru_rank
fi
OUT="$PROJECT/artifacts/models/gfpr_candidate_capacity_${SCOPE}_${SLURM_ARRAY_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/train_gfpr_candidate.py \
  --train-rollout "$ROLLOUT" \
  --eval-rollout "$ROLLOUT" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$OUT" \
  --train-split train \
  --eval-split train \
  --candidate-topk 16 \
  --trainable-scope "$SCOPE" \
  --epochs 20 \
  --prompts-per-batch 1 \
  --gradient-accumulation-prompts 4 \
  --base-learning-rate 1e-4 \
  --calibration-learning-rate 1e-2 \
  --delta-l2 1e-6 \
  --eval-every-steps 5 \
  --bootstrap-samples 200 \
  --capacity-only-allow-overlap \
  --seed 0
