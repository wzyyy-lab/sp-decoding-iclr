#!/bin/bash
#SBATCH --job-name=gfpr-target-teacher
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-target-teacher-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-target-teacher-%A_%a.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
SOURCE="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
SPLIT=train
LABEL=train
if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
  SPLIT=validation_select
  LABEL=validation_select
fi
COLLECTION="$PROJECT/artifacts/canonical/gfpr_target_teacher_${LABEL}_${SLURM_ARRAY_JOB_ID}"
ANALYSIS="$PROJECT/artifacts/analysis/gfpr_target_teacher_${LABEL}_${SLURM_ARRAY_JOB_ID}.json"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/collect_gfpr_rollouts.py \
  --canonical "$SOURCE" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$COLLECTION" \
  --mode fixed \
  --splits "$SPLIT" \
  --store-target-boundary-feature \
  --shard-blocks 512 \
  --attn-implementation sdpa

"$PYTHON" scripts/analyze_gfpr_rollouts.py \
  --collection "$COLLECTION" \
  --output "$ANALYSIS" \
  --fail-on-gate
