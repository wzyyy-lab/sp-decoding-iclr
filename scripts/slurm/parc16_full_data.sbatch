#!/bin/bash
#SBATCH --job-name=parc16-data
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-15%16
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/parc16-data-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/parc16-data-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets/models"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
MANIFEST_ROOT="$PROJECT/artifacts/manifests/parc16_opb270k_reserve_v2/train_validation_parts"
RUN_ROOT="$PROJECT/artifacts/canonical/parc16_full16_opb270k_reserve_${SLURM_ARRAY_JOB_ID}"
TASK="$SLURM_ARRAY_TASK_ID"
PADDED_TASK=$(printf '%03d' "$TASK")

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/collect_parc16_data.py \
  --target "$ASSETS/Qwen3-4B" \
  --draft "$ASSETS/Qwen3-4B-DFlash-b16" \
  --domino-draft "$ASSETS/Qwen3-4B-Domino-b16" \
  --manifest "$MANIFEST_ROOT/part-${PADDED_TASK}.jsonl" \
  --continuation-tokens 129 \
  --anchors-per-prompt 8 \
  --shard-prompts 16 \
  --attn-implementation sdpa \
  --output "$RUN_ROOT/part-${PADDED_TASK}"
