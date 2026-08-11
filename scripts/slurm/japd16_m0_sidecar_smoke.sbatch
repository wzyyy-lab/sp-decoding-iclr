#!/bin/bash
#SBATCH --job-name=japd-lse-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:15:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/japd-lse-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/japd-lse-smoke-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
ROLLOUT="$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_10164718"
SIDECAR="$PROJECT/artifacts/canonical/japd_lse_validation_smoke_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

# Compliance: full16 batch1 replay; pure-base K16; no target transformer
# forward; output is a scalar-feature sidecar only, never a path/tree.
nvidia-smi
"$PYTHON" scripts/materialize_japd_base_lse.py \
  --rollout "$ROLLOUT" \
  --target "$ASSETS/models/Qwen3-4B" \
  --split validation_select \
  --mode materialize \
  --output "$SIDECAR" \
  --max-records 32 \
  --shard-size 32 \
  --require-cuda

"$PYTHON" scripts/materialize_japd_base_lse.py \
  --rollout "$ROLLOUT" \
  --target "$ASSETS/models/Qwen3-4B" \
  --split validation_select \
  --mode verify \
  --sidecar "$SIDECAR" \
  --max-records 32 \
  --require-cuda
