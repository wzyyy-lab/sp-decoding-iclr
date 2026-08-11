#!/bin/bash
#SBATCH --job-name=pcld-p0
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pcld-p0-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pcld-p0-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
ROLLOUT="$PROJECT/artifacts/canonical/r047_anchor_t4_train_10164718"
TARGET="$ASSETS/models/Qwen3-4B"
SIDECAR="$PROJECT/artifacts/canonical/pcld_sidecar_smoke_${SLURM_JOB_ID}"
MECHANICS="$PROJECT/artifacts/analysis/pcld16_mechanics_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

nvidia-smi
"$PYTHON" scripts/materialize_pcld_sidecar.py \
  --rollout "$ROLLOUT" \
  --target "$TARGET" \
  --split train \
  --group smoke32 \
  --mode materialize \
  --output "$SIDECAR" \
  --max-records 32 \
  --manual-parity-records 32 \
  --shard-size 32 \
  --require-cuda

"$PYTHON" scripts/materialize_pcld_sidecar.py \
  --rollout "$ROLLOUT" \
  --target "$TARGET" \
  --split train \
  --group smoke32 \
  --mode verify \
  --sidecar "$SIDECAR" \
  --max-records 32 \
  --require-cuda

"$PYTHON" scripts/check_pcld_mechanics.py \
  --rollout "$ROLLOUT" \
  --sidecar "$SIDECAR" \
  --target "$TARGET" \
  --split train \
  --output "$MECHANICS" \
  --records 4 \
  --seed 0 \
  --require-cuda

