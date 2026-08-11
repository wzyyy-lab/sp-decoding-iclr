#!/bin/bash
#SBATCH --job-name=gfpr-gateb-gru-rank
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-gateb-gru-rank-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-gateb-gru-rank-%A_%a.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
TRAIN="$PROJECT/artifacts/canonical/gfpr_phase3_train_dynamic_10163980"
EVAL="$PROJECT/artifacts/canonical/gfpr_validation_select_fixed_10163965"
LR=3e-5
L2=1e-3
ARM=dynamic15_lr3e5_l2_1e3
EXTRA=()
case "$SLURM_ARRAY_TASK_ID" in
  0) ;;
  1)
    LR=1e-4
    ARM=dynamic15_lr1e4_l2_1e3
    ;;
  2)
    L2=1e-2
    ARM=dynamic15_lr3e5_l2_1e2
    ;;
  3)
    ARM=gfpr16_lr3e5_l2_1e3
    EXTRA+=(--correct-position-zero --position-zero-learning-rate 1e-3)
    ;;
  *)
    echo "unexpected task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac
OUT="$PROJECT/artifacts/models/gfpr_gateb_gru_rank_${ARM}_${SLURM_ARRAY_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

echo "GFPR Gate B gru-rank arm=$ARM lr=$LR l2=$L2 output=$OUT"
"$PYTHON" scripts/train_gfpr_head.py \
  --train-rollout "$TRAIN" \
  --eval-rollout "$EVAL" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$OUT" \
  --train-split train \
  --eval-split validation_select \
  --epochs 3 \
  --prompts-per-batch 1 \
  --gradient-accumulation-prompts 16 \
  --eval-batch-size 1 \
  --learning-rate "$LR" \
  --trainable-scope gru_rank \
  --delta-l2 "$L2" \
  --eval-every-steps 50 \
  --save-every-eval \
  --bootstrap-samples 10000 \
  --seed 0 \
  "${EXTRA[@]}"
