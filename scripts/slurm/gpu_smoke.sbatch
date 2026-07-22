#!/bin/bash
#SBATCH --job-name=sph-smoke
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-smoke-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/gpu_smoke.py \
  --target-model "$ASSETS/models/Qwen3-4B" \
  --draft-model "$ASSETS/models/Qwen3-4B-DFlash-b16" \
  --draft-model "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --max-new-tokens 32 \
  --output "$PROJECT/artifacts/smoke/gpu_smoke_${SLURM_JOB_ID}.json"
