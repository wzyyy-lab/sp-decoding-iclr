#!/bin/bash
#SBATCH --job-name=sph-g0-equiv
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-g0-equiv-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-g0-equiv-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/diagnostics/target_equivalence_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/diagnose_target_equivalence.py \
  --target "$ASSETS/models/Qwen3-4B" \
  --manifest "$PROJECT/artifacts/manifests/prompts_smoke_v1.jsonl" \
  --backends eager sdpa \
  --reference-backend eager \
  --max-samples 6 \
  --continuation-tokens 32 \
  --block-size 16 \
  --anchors-per-sample 2 \
  --fail-on-gate \
  --output "$OUTPUT"
