#!/bin/bash
#SBATCH --job-name=parc16-train
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/parc16-train-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/parc16-train-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets/models"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"

if [[ -z "${PARC_DATA_ROOT:-}" ]]; then
  echo "PARC_DATA_ROOT must name the completed 16-part full16 collection" >&2
  exit 2
fi
if [[ -z "${PARC_OUTPUT:-}" ]]; then
  echo "PARC_OUTPUT must name the one frozen formal-training directory" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT"

RESUME_ARGS=()
if [[ -f "$PARC_OUTPUT/latest.pt" ]]; then
  if [[ -f "$PARC_OUTPUT/report.json" ]]; then
    STATUS=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$PARC_OUTPUT/report.json")
    if [[ "$STATUS" != "interrupted_resumable" ]]; then
      echo "refusing to resume terminal PARC status: $STATUS" >&2
      exit 3
    fi
  fi
  RESUME_ARGS=(--resume "$PARC_OUTPUT/latest.pt")
fi

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/train_parc16.py \
  --target "$ASSETS/Qwen3-4B" \
  --draft "$ASSETS/Qwen3-4B-DFlash-b16" \
  --domino-draft "$ASSETS/Qwen3-4B-Domino-b16" \
  --data-root "$PARC_DATA_ROOT" \
  --output "$PARC_OUTPUT" \
  --attn-implementation sdpa \
  --batch-size 8 \
  --total-steps 180000 \
  --eval-every 10000 \
  --save-every 1000 \
  --warmup-steps 2000 \
  --head-learning-rate 0.0003 \
  --dflash-learning-rate 0.00001 \
  --weight-decay 0.0 \
  --gradient-clip 1.0 \
  --seed 0 \
  "${RESUME_ARGS[@]}"
