#!/bin/bash
#SBATCH --job-name=domino-dels
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-dels-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-dels-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DELS="$PROJECT/third_party/DeLS-Spec/checkpoints/dels-spec/qwen3-4b"
OUTPUT="$PROJECT/artifacts/analysis/domino_dels_validation_select_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
"$PYTHON" scripts/diagnose_domino_dels_fusion.py \
  --canonical "$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --dels-local-head "$DELS/local_head.pt" \
  --dels-unigram "$DELS/loss_mask_unigram.pt" \
  --split validation_select \
  --matched-horizon 15 \
  --configs \
    domino:1:0:0 \
    dscale075:0.75:0:0 \
    dscale125:1.25:0:0 \
    joint_a010:1:0.1:0.1 \
    joint_a020:1:0.2:0.2 \
    joint_a030:1:0.3:0.3 \
    joint_g075_a015:0.75:0.15:0.15 \
    joint_g075_a030:0.75:0.3:0.3 \
    joint_g125_a015:1.25:0.15:0.15 \
    joint_a030_b015:1:0.3:0.15 \
    joint_a015_b030:1:0.15:0.3 \
  --bootstrap-samples 20000 \
  --output "$OUTPUT"
