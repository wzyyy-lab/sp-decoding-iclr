#!/bin/bash
#SBATCH --job-name=gfpr-pos0-sweep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-pos0-sweep-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-pos0-sweep-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/sweep_gfpr_position_zero.py \
  --rollout "$PROJECT/artifacts/canonical/gfpr_dynamic_smoke_10163938" \
  --split train \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$PROJECT/artifacts/analysis/gfpr_pos0_sweep_smoke_${SLURM_JOB_ID}.json" \
  --scales -2 -1 -0.5 -0.25 -0.1 0 0.1 0.25 0.5 1 2
