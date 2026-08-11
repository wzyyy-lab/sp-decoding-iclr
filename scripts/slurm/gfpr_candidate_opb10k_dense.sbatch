#!/bin/bash
#SBATCH --job-name=gfpr-cand-10k-dense
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-3%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-cand-10k-dense-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-cand-10k-dense-%A_%a.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
TRAIN="$PROJECT/artifacts/canonical/gfpr_candidate_opb10k_fixed_10164123"
EVAL="$PROJECT/artifacts/canonical/gfpr_validation_select_fixed_10163965"

SCOPE=input_rank
BASE_LR=1e-4
CALIBRATION_LR=1e-4
DENSE_WEIGHT=1.0
LABEL=input_lr1e4_cal1e4_dense1
case "$SLURM_ARRAY_TASK_ID" in
  1)
    BASE_LR=3e-4
    LABEL=input_lr3e4_cal1e4_dense1
    ;;
  2)
    DENSE_WEIGHT=0.25
    LABEL=input_lr1e4_cal1e4_dense0p25
    ;;
  3)
    SCOPE=gru_rank
    BASE_LR=3e-5
    LABEL=gru_lr3e5_cal1e4_dense1
    ;;
esac

OUT="$PROJECT/artifacts/models/gfpr_candidate_opb10k_${LABEL}_${SLURM_ARRAY_JOB_ID}"
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
  --trainable-scope "$SCOPE" \
  --epochs 3 \
  --prompts-per-batch 4 \
  --gradient-accumulation-prompts 16 \
  --eval-batch-size 16 \
  --base-learning-rate "$BASE_LR" \
  --calibration-learning-rate "$CALIBRATION_LR" \
  --delta-l2 1e-3 \
  --dense-dpace-weight "$DENSE_WEIGHT" \
  --dpace-alpha 0.5 \
  --eval-every-steps 100 \
  --bootstrap-samples 10000 \
  --save-every-eval \
  --seed 0
