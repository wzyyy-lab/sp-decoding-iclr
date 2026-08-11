#!/bin/bash
#SBATCH --job-name=pgcf16-gate2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pgcf16-gate2-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pgcf16-gate2-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
EVAL="$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_10164718"
OUT="$PROJECT/artifacts/results/pgcf16_gate2_${SLURM_JOB_ID}"

: "${GLOBAL_CHECKPOINT:?submit with an explicit GLOBAL_CHECKPOINT=/path/to/global/best.pt}"
: "${LOCAL_CHECKPOINT:?submit with an explicit LOCAL_CHECKPOINT=/path/to/local/best.pt}"
test -f "$GLOBAL_CHECKPOINT"
test -f "$LOCAL_CHECKPOINT"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

nvidia-smi
"$PYTHON" scripts/evaluate_pgcf16_gate2.py \
  --global-checkpoint "$GLOBAL_CHECKPOINT" \
  --local-checkpoint "$LOCAL_CHECKPOINT" \
  --eval-rollout "$EVAL" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$OUT" \
  --eval-split validation_select \
  --eval-batch-size 8 \
  --remote-batch-size 2 \
  --require-cuda
