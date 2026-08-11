#!/bin/bash
#SBATCH --job-name=gfpr-gatea-full
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-gatea-full-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-gatea-full-%A_%a.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
SOURCE="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
MODE=fixed
ANALYZE_EXTRA=(--expected-released-eal 7.23955296404276 --expected-eal-tolerance 1e-9)
if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
  MODE=dynamic
  ANALYZE_EXTRA=()
fi
COLLECTION="$PROJECT/artifacts/canonical/gfpr_validation_select_${MODE}_${SLURM_ARRAY_JOB_ID}"
ANALYSIS="$PROJECT/artifacts/analysis/gfpr_validation_select_${MODE}_${SLURM_ARRAY_JOB_ID}.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/collect_gfpr_rollouts.py \
  --canonical "$SOURCE" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$COLLECTION" \
  --mode "$MODE" \
  --splits validation_select \
  --shard-blocks 256 \
  --attn-implementation sdpa

"$PYTHON" scripts/analyze_gfpr_rollouts.py \
  --collection "$COLLECTION" \
  --output "$ANALYSIS" \
  --fail-on-gate \
  "${ANALYZE_EXTRA[@]}"
