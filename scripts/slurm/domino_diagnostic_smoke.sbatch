#!/bin/bash
#SBATCH --job-name=sph-domino-diag
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-domino-diag-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-domino-diag-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
RUN_DIR="$PROJECT/artifacts/canonical/qwen3_4b_domino_smoke_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/collect_domino_canonical.py \
  --target "$ASSETS/models/Qwen3-4B" \
  --draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --manifest "$PROJECT/artifacts/manifests/prompts_smoke_v1.jsonl" \
  --block-size 16 \
  --top-k 64 \
  --anchors-per-sample 2 \
  --continuation-tokens 64 \
  --shard-blocks 64 \
  --output "$RUN_DIR"

"$PYTHON" scripts/analyze_candidate_ceiling.py \
  --input "$RUN_DIR" \
  --output "$PROJECT/artifacts/analysis/domino_base_ceiling_smoke_${SLURM_JOB_ID}.json" \
  --bootstrap-samples 200

"$PYTHON" scripts/analyze_domino_pollution.py \
  --input "$RUN_DIR" \
  --output "$PROJECT/artifacts/analysis/domino_pollution_smoke_${SLURM_JOB_ID}.json"
