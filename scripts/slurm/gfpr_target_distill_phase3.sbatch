#!/bin/bash
#SBATCH --job-name=gfpr-target-distill
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-7%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-target-distill-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-target-distill-%A_%a.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
TRAIN="$PROJECT/artifacts/canonical/gfpr_target_teacher_train_10164355"
EVAL="$PROJECT/artifacts/canonical/gfpr_target_teacher_validation_select_10164355"

BASE_LR=1e-4
ADAPTER_RANK=16
TARGET_KL=1
TARGET_ADVANTAGE=0
TEMPERATURE=1
LABEL=r16_kl_t1_lr1e4
case "$SLURM_ARRAY_TASK_ID" in
  1)
    BASE_LR=3e-4
    LABEL=r16_kl_t1_lr3e4
    ;;
  2)
    ADAPTER_RANK=64
    LABEL=r64_kl_t1_lr1e4
    ;;
  3)
    ADAPTER_RANK=64
    BASE_LR=3e-4
    LABEL=r64_kl_t1_lr3e4
    ;;
  4)
    ADAPTER_RANK=64
    TARGET_KL=0
    TARGET_ADVANTAGE=1
    LABEL=r64_adv_lr1e4
    ;;
  5)
    ADAPTER_RANK=64
    BASE_LR=3e-4
    TARGET_KL=0
    TARGET_ADVANTAGE=1
    LABEL=r64_adv_lr3e4
    ;;
  6)
    ADAPTER_RANK=64
    TARGET_ADVANTAGE=0.25
    LABEL=r64_kl_adv025_lr1e4
    ;;
  7)
    ADAPTER_RANK=64
    TEMPERATURE=2
    LABEL=r64_kl_t2_lr1e4
    ;;
esac

OUT="$PROJECT/artifacts/models/gfpr_target_distill_${LABEL}_${SLURM_ARRAY_JOB_ID}"
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
  --epochs 4 \
  --prompts-per-batch 4 \
  --gradient-accumulation-prompts 16 \
  --eval-batch-size 1 \
  --base-learning-rate "$BASE_LR" \
  --delta-l2 1e-4 \
  --frontier-weight 0 \
  --target-kl-weight "$TARGET_KL" \
  --target-advantage-weight "$TARGET_ADVANTAGE" \
  --target-temperature "$TEMPERATURE" \
  --target-huber-delta 0.5 \
  --target-protect-weight 1 \
  --target-repair-weight 4 \
  --eval-every-steps 100 \
  --bootstrap-samples 10000 \
  --save-every-eval \
  --seed 0
