#!/bin/bash
#SBATCH --job-name=r048-cap-train-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r048-cap-train-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r048-cap-train-smoke-%j.err

set -euo pipefail
: "${R048_SMOKE_COLLECTION_JOB:?submit with --export=ALL,R048_SMOKE_COLLECTION_JOB=<job>}"
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
COLLECTION="$PROJECT/artifacts/canonical/r048_capacity_smoke_${R048_SMOKE_COLLECTION_JOB}"
OUT="$PROJECT/artifacts/models/r048_capacity_train_smoke_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
"$PYTHON" scripts/train_r048_capacity.py \
  --collection "$COLLECTION" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$OUT" \
  --max-steps 2 --batch-size 8 --learning-rate 0.001 \
  --max-grad-norm 1 --temperature 2 --frontier-weight 4 \
  --keep-weight 1 --keep-margin 0.05 --eval-every 1 \
  --minimum-recovery 0 --required-prompts 8 --seed 0
