#!/bin/bash
#SBATCH --job-name=fada-cld-pre25k
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-cld-pre25k-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-cld-pre25k-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/training/fada_cld_pretrain_25k_k4_d128_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/pretrain_causal_lattice_decoder.py \
  --train-canonical "$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_25k_seed20260730" \
  --eval-canonical "$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436" \
  --target "$ASSETS/models/Qwen3-4B" \
  --candidate-topk 4 \
  --model-dim 128 \
  --num-heads 4 \
  --lattice-layers 1 \
  --decoder-layers 2 \
  --objective breaker_margin \
  --epochs 3 \
  --batch-size 128 \
  --eval-batch-size 64 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --prefix-weight 0.5 \
  --residual-penalty-weight 1e-4 \
  --eval-every-steps 500 \
  --output "$OUTPUT"
