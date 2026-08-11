#!/bin/bash
#SBATCH --job-name=japd-lse-full
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/japd-lse-full-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/japd-lse-full-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
ROLLOUT="$PROJECT/artifacts/canonical/r047_anchor_t4_train_10164718"
SIDECAR="$PROJECT/artifacts/canonical/japd_lse_train_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

nvidia-smi
"$PYTHON" scripts/materialize_japd_base_lse.py \
  --rollout "$ROLLOUT" \
  --target "$ASSETS/models/Qwen3-4B" \
  --split train \
  --mode materialize \
  --output "$SIDECAR" \
  --shard-size 512 \
  --require-cuda

"$PYTHON" scripts/materialize_japd_base_lse.py \
  --rollout "$ROLLOUT" \
  --target "$ASSETS/models/Qwen3-4B" \
  --split train \
  --mode verify \
  --sidecar "$SIDECAR" \
  --require-cuda
