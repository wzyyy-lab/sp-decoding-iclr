#!/bin/bash
#SBATCH --job-name=fada-domino-mat25k
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-mat25k-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-mat25k-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
PART=$(printf '%03d' "$SLURM_ARRAY_TASK_ID")
SOURCE="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770/part-$PART"
OUTPUT="$PROJECT/artifacts/canonical/qwen3_4b_domino_opb100k_half_part${PART}_${SLURM_ARRAY_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/materialize_domino_same_anchor.py \
  --canonical "$SOURCE" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --splits train \
  --max-samples 6250 \
  --shard-blocks 512 \
  --output "$OUTPUT"
