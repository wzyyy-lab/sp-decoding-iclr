#!/bin/bash
#SBATCH --job-name=sph-g1b-anchor
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-g1b-anchor-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-g1b-anchor-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/analysis/gate1b_same_anchor_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/compare_same_anchor_domino.py \
  --canonical "$PROJECT/artifacts/canonical/qwen3_4b_gate1_10022338" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --manifest "$PROJECT/artifacts/manifests/prompts_v1.jsonl" \
  --matched-horizon 15 \
  --oracle-k 8 16 \
  --bootstrap-samples 10000 \
  --fail-on-gate \
  --output "$OUTPUT"
