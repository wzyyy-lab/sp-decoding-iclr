#!/bin/bash
#SBATCH --job-name=fada-domino-iter-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-iter-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-iter-smoke-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/analysis/fada_domino_iterative_smoke_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/evaluate_domino_iterative_refinement.py \
  --canonical "$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --split validation_select \
  --passes 1 2 \
  --max-samples 2 \
  --bootstrap-samples 100 \
  --output "$OUTPUT"
