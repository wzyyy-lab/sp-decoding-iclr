#!/bin/bash
#SBATCH --job-name=fada-domino-cld-reset
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-cld-reset-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-cld-reset-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"

TOPKS=(4 16)
BATCHES=(64 32)
CHECKPOINTS=(
  "$PROJECT/artifacts/training/fada_cld_pretrain_100k_k4_d128_10155348/best_decoder.pt"
  "$PROJECT/artifacts/training/fada_cld_pretrain_100k_k16_d128_10155350/best_decoder.pt"
)
TOPK=${TOPKS[$SLURM_ARRAY_TASK_ID]}
BATCH=${BATCHES[$SLURM_ARRAY_TASK_ID]}
PRETRAIN_CHECKPOINT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}
OUTPUT="$PROJECT/artifacts/training/fada_domino_cld_reset_k${TOPK}_d128_${SLURM_ARRAY_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/train_domino_causal_lattice_decoder.py \
  --canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_opb10k_balanced_v1" \
  --additional-train-canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --eval-canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --init-decoder-checkpoint "$PRETRAIN_CHECKPOINT" \
  --reset-transfer-projection \
  --candidate-topk "$TOPK" \
  --model-dim 128 \
  --num-heads 4 \
  --lattice-layers 1 \
  --decoder-layers 2 \
  --objective breaker_margin \
  --epochs 2 \
  --batch-size "$BATCH" \
  --eval-batch-size 1 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --prefix-weight 1.0 \
  --residual-penalty-weight 1e-3 \
  --eval-every-steps 250 \
  --bootstrap-samples 5000 \
  --output "$OUTPUT"
