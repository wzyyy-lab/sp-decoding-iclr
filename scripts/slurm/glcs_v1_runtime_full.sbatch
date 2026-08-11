#!/bin/bash
#SBATCH --job-name=glcs-v1-runtime
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/glcs-v1-runtime-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/glcs-v1-runtime-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
RUNTIME="$PROJECT/artifacts/canonical/plc_runtime_phase3_10158769"
OUTPUT="$PROJECT/artifacts/training/glcs_v1_runtime_full_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/train_global_lookahead_causal_selector.py \
  --train-canonical "$RUNTIME" \
  --eval-runtime "$RUNTIME" \
  --eval-split validation_select \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --candidate-topk 16 \
  --global-width 512 \
  --global-heads 8 \
  --global-layers 2 \
  --global-modes 4 \
  --feed-forward-width 1536 \
  --epochs 8 \
  --batch-size 32 \
  --eval-batch-size 32 \
  --adapter-learning-rate 2e-4 \
  --base-learning-rate 2e-5 \
  --margin-weight 1.0 \
  --margin-offset 1.0 \
  --auxiliary-weight 0.1 \
  --eval-every-steps 500 \
  --output "$OUTPUT"
