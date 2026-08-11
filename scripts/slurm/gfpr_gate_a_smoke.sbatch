#!/bin/bash
#SBATCH --job-name=gfpr-gatea-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-gatea-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-gatea-smoke-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
SOURCE="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
DYNAMIC="$PROJECT/artifacts/canonical/gfpr_dynamic_smoke_${SLURM_JOB_ID}"
FIXED="$PROJECT/artifacts/canonical/gfpr_fixed_smoke_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/collect_gfpr_rollouts.py \
  --canonical "$SOURCE" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$DYNAMIC" \
  --mode dynamic \
  --splits train \
  --max-samples 8 \
  --shard-blocks 128 \
  --attn-implementation sdpa

"$PYTHON" scripts/analyze_gfpr_rollouts.py \
  --collection "$DYNAMIC" \
  --output "$PROJECT/artifacts/analysis/gfpr_dynamic_smoke_${SLURM_JOB_ID}.json" \
  --fail-on-gate

"$PYTHON" scripts/collect_gfpr_rollouts.py \
  --canonical "$SOURCE" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$FIXED" \
  --mode fixed \
  --splits train \
  --max-samples 8 \
  --shard-blocks 128 \
  --attn-implementation sdpa

"$PYTHON" scripts/analyze_gfpr_rollouts.py \
  --collection "$FIXED" \
  --output "$PROJECT/artifacts/analysis/gfpr_fixed_smoke_${SLURM_JOB_ID}.json" \
  --fail-on-gate
