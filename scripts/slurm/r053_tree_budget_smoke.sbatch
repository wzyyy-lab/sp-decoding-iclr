#!/bin/bash
#SBATCH --job-name=r053-tree-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r053-tree-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r053-tree-smoke-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/analysis/r053_tree_budget_smoke_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/evaluate_r053_tree_budget.py \
  --source-rollout artifacts/canonical/r047_anchor_t4_validation_select_10164718 \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --r052-report profile_output/r052_exact_prefix_eager_10165070.json \
  --output "$OUTPUT" \
  --split validation_select \
  --candidate-pool-topk 64 \
  --tree-support-size 16 \
  --beam-width 16 \
  --budgets 17 24 32 48 64 \
  --gamma 0.75 \
  --warmup 2 \
  --repeats 10 \
  --max-records 4

