#!/bin/bash
#SBATCH --job-name=sph-g1b-v2-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-g1b-v2-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-g1b-v2-smoke-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
: "${CANONICAL_JOB_ID:?submit with CANONICAL_JOB_ID set to the v2 collector job}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/compare_same_anchor_domino.py \
  --canonical "$PROJECT/artifacts/canonical/qwen3_4b_smoke_${CANONICAL_JOB_ID}" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --manifest "$PROJECT/artifacts/manifests/prompts_smoke_v1.jsonl" \
  --matched-horizon 15 \
  --oracle-k 8 16 \
  --max-samples 6 \
  --bootstrap-samples 1000 \
  --output "$PROJECT/artifacts/analysis/gate1b_stored_context_smoke_${SLURM_JOB_ID}.json"
