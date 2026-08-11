#!/bin/bash
#SBATCH --job-name=fada-domino-cld-215k
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-cld-215k-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-cld-215k-%j.err

set -euo pipefail

: "${DOMINO_MAT_JOB:?submit with DOMINO_MAT_JOB=<materialization array job id>}"

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
MAT="$PROJECT/artifacts/canonical"
PRETRAIN="$PROJECT/artifacts/training/fada_cld_pretrain_100k_k4_d128_10155348/best_decoder.pt"
OUTPUT="$PROJECT/artifacts/training/fada_domino_cld_215k_k4_d128_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/train_domino_causal_lattice_decoder.py \
  --canonical "$MAT/qwen3_4b_domino_opb100k_half_part000_${DOMINO_MAT_JOB}" \
  --additional-train-canonical \
    "$MAT/qwen3_4b_domino_opb100k_half_part001_${DOMINO_MAT_JOB}" \
    "$MAT/qwen3_4b_domino_opb100k_half_part002_${DOMINO_MAT_JOB}" \
    "$MAT/qwen3_4b_domino_opb100k_half_part003_${DOMINO_MAT_JOB}" \
    "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --eval-canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --init-decoder-checkpoint "$PRETRAIN" \
  --reset-transfer-projection \
  --candidate-topk 4 \
  --model-dim 128 \
  --num-heads 4 \
  --lattice-layers 1 \
  --decoder-layers 2 \
  --objective breaker_margin \
  --epochs 2 \
  --batch-size 64 \
  --eval-batch-size 1 \
  --learning-rate 5e-5 \
  --weight-decay 1e-4 \
  --prefix-weight 1.0 \
  --residual-penalty-weight 1e-3 \
  --eval-every-steps 500 \
  --bootstrap-samples 5000 \
  --output "$OUTPUT"
