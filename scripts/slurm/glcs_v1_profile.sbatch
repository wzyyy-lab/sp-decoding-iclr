#!/bin/bash
#SBATCH --job-name=glcs-v1-profile
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/glcs-v1-profile-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/glcs-v1-profile-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
CHECKPOINT="${GLCS_CHECKPOINT:-$PROJECT/artifacts/training/glcs_v1_baseline_10161695/best.pt}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/profile_glcs_head.py \
  --canonical "$PROJECT/artifacts/canonical/plc_runtime_phase3_10158769" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --checkpoint "$CHECKPOINT" \
  --output "$PROJECT/profile_output/glcs_v1_${SLURM_JOB_ID}.json" \
  --warmup 25 \
  --repeats 200
