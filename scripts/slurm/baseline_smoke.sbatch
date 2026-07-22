#!/bin/bash
#SBATCH --job-name=sph-base-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-base-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-base-smoke-%j.err

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
"$PYTHON" scripts/run_single_chain_baselines.py \
  --target "$ASSETS/models/Qwen3-4B" \
  --dflash-draft "$ASSETS/models/Qwen3-4B-DFlash-b16" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --manifest "$PROJECT/artifacts/manifests/prompts_smoke_v1.jsonl" \
  --max-new-tokens 64 \
  --block-size 16 \
  --output "$PROJECT/artifacts/baselines/qwen3_4b_smoke_${SLURM_JOB_ID}.json"
