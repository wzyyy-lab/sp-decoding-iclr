#!/bin/bash
#SBATCH --job-name=gcls-opb100k
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-opb100k-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-opb100k-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
PARTS="$PROJECT/artifacts/manifests/open_perfectblend_100k_v2_parts"
RUN_ROOT="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_100k_${SLURM_ARRAY_JOB_ID}"

TASK="$SLURM_ARRAY_TASK_ID"
MANIFEST="$PARTS/part-$(printf '%03d' "$TASK").jsonl"
OUTPUT="$RUN_ROOT/part-$(printf '%03d' "$TASK")"

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
  --output "$OUTPUT"
