#!/bin/bash
#SBATCH --job-name=domino-l35-gate
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-l35-gate-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-l35-gate-%j.err

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
  --output "artifacts/analysis/domino_l35_validation_gate_${SLURM_JOB_ID}.json" \
  --split validation_gate \
  --early-layers 35 \
  --candidate-topk 16 \
  --beam-width 16 \
  --score-gammas 0.25 0.5 0.75 1.0 \
  --fusion-weights 0.125 0.25 0.5 1.0 2.0 4.0
