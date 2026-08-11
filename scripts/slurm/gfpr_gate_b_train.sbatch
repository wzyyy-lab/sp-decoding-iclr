#!/bin/bash
#SBATCH --job-name=gfpr-gateb-train
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-gateb-train-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-gateb-train-%A_%a.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
TRAIN_FIXED="$PROJECT/artifacts/canonical/gfpr_phase3_train_fixed_10163980"
TRAIN_DYNAMIC="$PROJECT/artifacts/canonical/gfpr_phase3_train_dynamic_10163980"
EVAL_FIXED="$PROJECT/artifacts/canonical/gfpr_validation_select_fixed_10163965"
TRAIN="$TRAIN_FIXED"
ARM=fixed15
EXTRA=()
if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
  TRAIN="$TRAIN_DYNAMIC"
  ARM=dynamic15
elif [[ "$SLURM_ARRAY_TASK_ID" == "2" ]]; then
  TRAIN="$TRAIN_DYNAMIC"
  ARM=gfpr16
  EXTRA+=(--correct-position-zero --position-zero-learning-rate 1e-3)
fi
OUT="$PROJECT/artifacts/models/gfpr_gateb_${ARM}_${SLURM_ARRAY_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

echo "GFPR Gate B arm=$ARM train=$TRAIN output=$OUT"
"$PYTHON" scripts/train_gfpr_head.py \
  --train-rollout "$TRAIN" \
  --eval-rollout "$EVAL_FIXED" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$OUT" \
  --train-split train \
  --eval-split validation_select \
  --epochs 3 \
  --prompts-per-batch 1 \
  --gradient-accumulation-prompts 16 \
  --eval-batch-size 1 \
  --learning-rate 3e-5 \
  --trainable-scope full_head \
  --delta-l2 1e-6 \
  --eval-every-steps 50 \
  --bootstrap-samples 10000 \
  --seed 0 \
  "${EXTRA[@]}"
