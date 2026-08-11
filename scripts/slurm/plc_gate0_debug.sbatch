#!/bin/bash
#SBATCH --job-name=plc-gate0
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/plc-gate0-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/plc-gate0-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
EXTRA_ARGS=()
if [[ -n "${PLC_WIDTH:-}" ]]; then
  EXTRA_ARGS+=(--width "$PLC_WIDTH")
fi
if [[ -n "${PLC_MODES:-}" ]]; then
  EXTRA_ARGS+=(--modes "$PLC_MODES")
fi
if [[ -n "${PLC_HEADS:-}" ]]; then
  EXTRA_ARGS+=(--heads "$PLC_HEADS")
fi
if [[ -n "${PLC_FFN_WIDTH:-}" ]]; then
  EXTRA_ARGS+=(--feed-forward-width "$PLC_FFN_WIDTH")
fi
if [[ -n "${PLC_GLOBAL_LAYERS:-}" ]]; then
  EXTRA_ARGS+=(--global-layers "$PLC_GLOBAL_LAYERS")
fi
if [[ "${PLC_USE_SEMANTIC_EMBEDDING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use-semantic-embedding)
fi
if [[ "${PLC_USE_FULL_HIDDEN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use-full-hidden)
fi
"$PYTHON" scripts/profile_plc_gate0.py \
  --canonical artifacts/canonical/qwen3_4b_domino_phase3_v1 \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "profile_output/plc_gate0_${SLURM_JOB_ID}.json" \
  --record-index 0 \
  --warmup 25 \
  --repeats 200 \
  "${EXTRA_ARGS[@]}"
