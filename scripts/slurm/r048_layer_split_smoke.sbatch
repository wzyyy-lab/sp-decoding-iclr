#!/bin/bash
#SBATCH --job-name=r048-layer-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r048-layer-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r048-layer-smoke-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/analysis/r048_layer_split_smoke_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/profile_r048_layer_split.py \
  --collection "$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_10164718" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --candidate-profile "$PROJECT/artifacts/analysis/r048_fast_candidate_profile_10164829.json" \
  --oracle-report "$PROJECT/artifacts/analysis/r048_fast_k64_oracle_10164820.json" \
  --output "$OUTPUT" \
  --record-index 723 \
  --max-distribution-records 8 \
  --candidate-topk 64 \
  --early-layers 4 \
  --warmup 1 \
  --repeats 2 \
  --domino-graph-head-ms 2.110945281982422 \
  --minimum-throughput-ratio 1.20 \
  --maximum-token-path-mismatches 0
