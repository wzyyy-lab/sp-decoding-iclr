#!/bin/bash
#SBATCH --job-name=fada-gate-screen
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-gate-screen-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-gate-screen-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"

GATE_TYPES=(
  position position position position
  state_position state_position state_position state_position state_position state_position
)
OBJECTIVES=(
  breaker breaker breaker_margin breaker_margin
  breaker breaker breaker breaker_margin breaker_margin breaker_margin
)
LEARNING_RATES=(
  1e-3 3e-3 1e-3 3e-3
  1e-4 3e-4 1e-3 1e-4 3e-4 1e-3
)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
for TASK in {0..9}; do
  GATE_TYPE=${GATE_TYPES[$TASK]}
  OBJECTIVE=${OBJECTIVES[$TASK]}
  LEARNING_RATE=${LEARNING_RATES[$TASK]}
  OUTPUT="$PROJECT/artifacts/training/fada_gate_screen_${SLURM_JOB_ID}_${TASK}"
  "$PYTHON" scripts/train_domino_cached_gate.py \
    --canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
    --target "$ASSETS/models/Qwen3-4B" \
    --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
    --gate-type "$GATE_TYPE" \
    --objective "$OBJECTIVE" \
    --epochs 5 \
    --batch-size 16 \
    --eval-batch-size 1 \
    --learning-rate "$LEARNING_RATE" \
    --bootstrap-samples 2000 \
    --output "$OUTPUT"
done
