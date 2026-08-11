#!/bin/bash
#SBATCH --job-name=gcls-opb-ablate
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-opb-ablate-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-opb-ablate-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
RUN="$PROJECT/artifacts/training/gcls_v1_open_perfectblend_curve_10099772/axial_global_d64_dpace_pall_seed0"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT"

cd "$PROJECT"
nvidia-smi
"$PYTHON" scripts/evaluate_global_scope_ablation.py \
  --checkpoint "$RUN/best.pt" \
  --metrics "$RUN/metrics.json" \
  --data "$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436" \
  --target "$ASSETS/models/Qwen3-4B" \
  --batch-size 64 \
  --output "$PROJECT/artifacts/analysis/gcls_v1_opb_scope_ablation_${SLURM_JOB_ID}.json"
