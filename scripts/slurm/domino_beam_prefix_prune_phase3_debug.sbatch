#!/bin/bash
#SBATCH --job-name=domino-prefix-beam
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-prefix-beam-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-prefix-beam-%A_%a.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
GAMMAS=(0.5 0.75)
PRUNE_GAMMA="${GAMMAS[$SLURM_ARRAY_TASK_ID]}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
"$PYTHON" scripts/evaluate_domino_beam_search.py \
  --canonical artifacts/canonical/qwen3_4b_domino_phase3_v1 \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "artifacts/analysis/domino_beam_prefix_prune_g${PRUNE_GAMMA}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json" \
  --split validation_select \
  --candidate-topk 16 \
  --beam-widths 64 \
  --score-gammas 0.5 0.625 0.75 0.875 1.0 \
  --normalization full_vocab \
  --prune-gamma "$PRUNE_GAMMA"
