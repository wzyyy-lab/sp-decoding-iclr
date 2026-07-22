#!/bin/bash
#SBATCH --job-name=sph-p3-t1-data
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-p3-t1-data-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-p3-t1-data-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
RUN_DIR="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_${SLURM_JOB_ID}"
REPORT="$PROJECT/artifacts/analysis/candidate_ceiling_phase3_tier1_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/collect_canonical_blocks.py \
  --target "$ASSETS/models/Qwen3-4B" \
  --draft "$ASSETS/models/Qwen3-4B-DFlash-b16" \
  --manifest "$PROJECT/artifacts/manifests/phase3_development_v2.jsonl" \
  --block-size 16 \
  --top-k 64 \
  --anchors-per-sample 8 \
  --continuation-tokens 256 \
  --shard-blocks 256 \
  --output "$RUN_DIR"

"$PYTHON" scripts/analyze_candidate_ceiling.py \
  --input "$RUN_DIR" \
  --output "$REPORT" \
  --bootstrap-samples 1000
