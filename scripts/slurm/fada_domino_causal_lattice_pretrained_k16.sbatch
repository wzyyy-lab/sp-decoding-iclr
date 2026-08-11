#!/bin/bash
#SBATCH --job-name=fada-domino-cld-pre-k16
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-cld-pre-k16-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-cld-pre-k16-%j.err

set -euo pipefail

: "${PRETRAIN_CHECKPOINT:?submit with PRETRAIN_CHECKPOINT=/absolute/path/best_decoder.pt}"

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/fada_domino_cld_pretrained_k16_d128_${SLURM_JOB_ID}"

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
  --candidate-topk 16 \
  --model-dim 128 \
  --num-heads 4 \
  --lattice-layers 1 \
  --decoder-layers 2 \
  --objective breaker_margin \
  --epochs 2 \
  --batch-size 32 \
  --eval-batch-size 1 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --prefix-weight 0.5 \
  --residual-penalty-weight 1e-4 \
  --eval-every-steps 250 \
  --bootstrap-samples 5000 \
  --output "$OUTPUT"
