#!/bin/bash
#SBATCH --job-name=japd-cap-d256
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/japd-cap-d256-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/japd-cap-d256-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
ROLLOUT="$PROJECT/artifacts/canonical/r047_anchor_t4_train_10164718"
MANIFEST="$PROJECT/artifacts/manifests/japd16_r047_split_20260810.json"
SIDECAR="${JAPD_SIDECAR:?set JAPD_SIDECAR to the reviewed full R047 sidecar}"
OUTPUT="$PROJECT/artifacts/models/japd16_capacity_d256_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
test -f "$SIDECAR/metadata.json"

nvidia-smi
"$PYTHON" scripts/train_japd16.py \
  --train-rollout "$ROLLOUT" \
  --eval-rollout "$ROLLOUT" \
  --train-sidecar "$SIDECAR" \
  --eval-sidecar "$SIDECAR" \
  --manifest "$MANIFEST" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$OUTPUT" \
  --train-split train \
  --eval-split train \
  --train-group capacity \
  --eval-group capacity \
  --scope global \
  --objective japd \
  --model-dim 256 \
  --num-heads 8 \
  --num-layers 2 \
  --dropout 0.1 \
  --batch-size 16 \
  --eval-batch-size 16 \
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
  --require-parameter-count 4539888
