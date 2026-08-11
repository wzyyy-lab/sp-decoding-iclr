#!/bin/bash
#SBATCH --job-name=pcld-cap
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pcld-cap-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pcld-cap-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
ROLLOUT="$PROJECT/artifacts/canonical/r047_anchor_t4_train_10164718"
MANIFEST="$PROJECT/artifacts/manifests/japd16_r047_split_20260810.json"
TARGET="$ASSETS/models/Qwen3-4B"
SIDECAR="$PROJECT/artifacts/canonical/pcld_sidecar_capacity_10168459"
SUPPORT_RECEIPT="$PROJECT/artifacts/manifests/pcld16_capacity_support_10168459.json"
OUTPUT="$PROJECT/artifacts/models/pcld16_capacity_${SLURM_JOB_ID}"
P0_MECHANICS="$PROJECT/artifacts/analysis/pcld16_mechanics_10168418.json"
P0_PROFILE="$PROJECT/profile_output/pcld16_eager_10168424.json"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

nvidia-smi
"$PYTHON" scripts/check_pcld_p1_preflight.py \
  --mechanics "$P0_MECHANICS" \
  --profile "$P0_PROFILE"

"$PYTHON" scripts/materialize_pcld_sidecar.py \
  --rollout "$ROLLOUT" \
  --target "$TARGET" \
  --split train \
  --group capacity \
  --manifest "$MANIFEST" \
  --mode verify \
  --sidecar "$SIDECAR" \
  --require-cuda

"$PYTHON" scripts/train_pcld16.py \
  --train-rollout "$ROLLOUT" \
  --eval-rollout "$ROLLOUT" \
  --train-sidecar "$SIDECAR" \
  --eval-sidecar "$SIDECAR" \
  --manifest "$MANIFEST" \
  --capacity-support-receipt "$SUPPORT_RECEIPT" \
  --target "$TARGET" \
  --output "$OUTPUT" \
  --train-split train \
  --eval-split train \
  --train-group capacity \
  --eval-group capacity \
  --scope global \
  --batch-size 8 \
  --eval-batch-size 32 \
  --max-steps 8000 \
  --eval-every-steps 250 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-steps 200 \
  --weight-decay 1e-2 \
  --gradient-clip 1 \
  --seed 0 \
  --gate capacity \
  --require-gate \
  --require-cuda \
  --require-parameter-count 3826688
