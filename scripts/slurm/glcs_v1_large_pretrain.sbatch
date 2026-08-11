#!/bin/bash
#SBATCH --job-name=glcs-v1-large
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/glcs-v1-large-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/glcs-v1-large-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
CANONICAL="$PROJECT/artifacts/canonical"
RUNTIME="$CANONICAL/plc_runtime_phase3_10158769"

EXTRA_ARGS=()
if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  TAG=frozen
  BASE_LR=0
  EXTRA_ARGS+=(--freeze-base)
else
  TAG=joint
  BASE_LR=1e-6
fi
OUTPUT="$PROJECT/artifacts/training/glcs_v1_large_${TAG}_${SLURM_ARRAY_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/train_global_lookahead_causal_selector.py \
  --train-canonical \
    "$CANONICAL/qwen3_4b_domino_opb100k_half_part000_10157195" \
    "$CANONICAL/qwen3_4b_domino_opb100k_half_part001_10157195" \
    "$CANONICAL/qwen3_4b_domino_opb100k_half_part002_10157195" \
    "$CANONICAL/qwen3_4b_domino_opb100k_half_part003_10157195" \
    "$CANONICAL/qwen3_4b_domino_opb10k_balanced_v1" \
    "$CANONICAL/qwen3_4b_domino_phase3_v1" \
  --eval-runtime "$RUNTIME" \
  --eval-split validation_select \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --candidate-topk 16 \
  --global-width 512 \
  --global-heads 8 \
  --global-layers 2 \
  --global-modes 4 \
  --feed-forward-width 1536 \
  --epochs 2 \
  --batch-size 64 \
  --eval-batch-size 32 \
  --adapter-learning-rate 1e-4 \
  --base-learning-rate "$BASE_LR" \
  --margin-weight 1.0 \
  --margin-offset 1.0 \
  --auxiliary-weight 0.1 \
  --eval-every-steps 500 \
  "${EXTRA_ARGS[@]}" \
  --output "$OUTPUT"
