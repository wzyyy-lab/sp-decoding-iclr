#!/bin/bash
#SBATCH --job-name=fada-cld-pre100k
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-cld-pre100k-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-cld-pre100k-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770"
OUTPUT="$PROJECT/artifacts/training/fada_cld_pretrain_100k_k4_d128_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/pretrain_causal_lattice_decoder.py \
  --train-canonical \
    "$DATA/part-000" \
    "$DATA/part-001" \
    "$DATA/part-002" \
    "$DATA/part-003" \
    "$DATA/part-004" \
    "$DATA/part-005" \
    "$DATA/part-006" \
    "$DATA/part-007" \
  --eval-canonical "$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436" \
  --target "$ASSETS/models/Qwen3-4B" \
  --candidate-topk 4 \
  --model-dim 128 \
  --num-heads 4 \
  --lattice-layers 1 \
  --decoder-layers 2 \
  --objective breaker_margin \
  --epochs 2 \
  --batch-size 128 \
  --eval-batch-size 64 \
  --learning-rate 8e-4 \
  --weight-decay 1e-4 \
  --prefix-weight 0.5 \
  --residual-penalty-weight 1e-4 \
  --eval-every-steps 500 \
  --output "$OUTPUT"
