#!/bin/bash
#SBATCH --job-name=r055-forest-full
#SBATCH --partition=i64m1tga40u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=150G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r055-forest-full-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r055-forest-full-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/analysis/r055_padded_forest_full_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/evaluate_r055_padded_forest.py \
  --source-rollout artifacts/canonical/r047_anchor_t4_validation_select_10164718 \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --r053-report artifacts/analysis/r053_tree_budget_pareto_10165201.json \
  --output "$OUTPUT" \
  --split validation_select \
  --beam-widths 4 8 16 \
  --warmup 5 \
  --repeats 30
