#!/bin/bash
#SBATCH --job-name=gfpr-target-boundary
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-target-boundary-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-target-boundary-%A_%a.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
TRAIN="$PROJECT/artifacts/canonical/gfpr_target_teacher_train_10164397"
EVAL="$PROJECT/artifacts/canonical/gfpr_target_teacher_validation_select_10164397"

ADAPTER_RANK=64
LABEL=r64
if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
  ADAPTER_RANK=128
  LABEL=r128
fi

OUT="$PROJECT/artifacts/models/gfpr_target_boundary_${LABEL}_${SLURM_ARRAY_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/train_gfpr_candidate.py \
  --train-rollout "$TRAIN" \
  --eval-rollout "$EVAL" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$OUT" \
  --train-split train \
  --eval-split validation_select \
  --candidate-topk 16 \
  --include-released-action \
  --released-union-teacher stored_frontier \
  --trainable-scope adapter \
  --adapter-rank "$ADAPTER_RANK" \
  --use-target-boundary-feature \
  --epochs 4 \
  --prompts-per-batch 4 \
  --gradient-accumulation-prompts 16 \
  --eval-batch-size 1 \
  --base-learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --delta-l2 1e-4 \
  --frontier-weight 0 \
  --target-kl-weight 1 \
  --target-advantage-weight 0 \
  --target-temperature 2 \
  --target-protect-weight 1 \
  --target-repair-weight 4 \
  --eval-every-steps 100 \
  --bootstrap-samples 10000 \
  --save-every-eval \
  --seed 0
