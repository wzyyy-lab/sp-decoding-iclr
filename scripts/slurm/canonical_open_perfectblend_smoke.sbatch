#!/bin/bash
#SBATCH --job-name=gcls-opb-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-opb-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-opb-smoke-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
MANIFEST="$PROJECT/artifacts/manifests/open_perfectblend_100k_v2_parts/part-000.jsonl"
OUTPUT="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_smoke_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/collect_canonical_blocks.py \
  --target "$ASSETS/models/Qwen3-4B" \
  --draft "$ASSETS/models/Qwen3-4B-DFlash-b16" \
  --manifest "$MANIFEST" \
  --block-size 16 \
  --top-k 64 \
  --anchors-per-sample 8 \
  --continuation-tokens 128 \
  --shard-blocks 256 \
  --max-samples 32 \
  --output "$OUTPUT"
