#!/bin/bash
#SBATCH --job-name=sph-canon-g1
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-canon-g1-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-canon-g1-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
RUN_DIR="$PROJECT/artifacts/canonical/qwen3_4b_gate1_${SLURM_JOB_ID}"
REPORT="$PROJECT/artifacts/analysis/candidate_ceiling_gate1_${SLURM_JOB_ID}.json"
PLOT="$PROJECT/artifacts/analysis/candidate_ceiling_gate1_${SLURM_JOB_ID}.pdf"

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
  --manifest "$PROJECT/artifacts/manifests/prompts_v1.jsonl" \
  --block-size 16 \
  --top-k 64 \
  --anchors-per-sample 8 \
  --continuation-tokens 256 \
  --shard-blocks 256 \
  --output "$RUN_DIR"

"$PYTHON" scripts/analyze_candidate_ceiling.py \
  --input "$RUN_DIR" \
  --output "$REPORT" \
  --plot "$PLOT" \
  --bootstrap-samples 1000
