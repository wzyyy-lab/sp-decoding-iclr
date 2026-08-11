#!/bin/bash
#SBATCH --job-name=domino-w64-l36
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-w64-l36-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-w64-l36-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
"$PYTHON" scripts/evaluate_domino_early_exit_reranker.py \
  --canonical artifacts/canonical/qwen3_4b_domino_phase3_v1 \
  --source-canonical artifacts/canonical/qwen3_4b_phase3_tier1_10035436 \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "artifacts/analysis/domino_width64_l36_validation_select_${SLURM_JOB_ID}.json" \
  --split validation_select \
  --early-layers 36 \
  --candidate-topk 16 \
  --beam-width 64 \
  --target-path-batch-size 32 \
  --score-gammas 0.25 0.5 0.75 1.0 \
  --fusion-weights 0.125 0.25 0.5 1.0 2.0 4.0
