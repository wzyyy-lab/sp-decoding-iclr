#!/bin/bash
#SBATCH --job-name=r047-anchor-align
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r047-anchor-align-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r047-anchor-align-%j.err

set -euo pipefail
: "${R047_COLLECTION_JOB:?submit with --export=ALL,R047_COLLECTION_JOB=<job>}"
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
COLLECTION="$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_${R047_COLLECTION_JOB}"
OUTPUT="$PROJECT/artifacts/analysis/r047_anchor_t4_alignment_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

ARGS=(
  --collection "$COLLECTION"
  --target "$ASSETS/models/Qwen3-4B"
  --output "$OUTPUT"
  --split validation_select
  --early-layers 4
  --max-blocks 24
  --minimum-cosine 0.999
  --maximum-relative-rms 0.05
  --maximum-path-change-fraction 0
  --maximum-eal-absolute-delta 0.02
  --fail-on-material-delta
)
if [[ -n "${R047_CHECKPOINT:-}" ]]; then
  ARGS+=(
    --checkpoint "$R047_CHECKPOINT"
    --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16"
  )
fi
"$PYTHON" scripts/check_anchor_early_exit_alignment.py "${ARGS[@]}"
