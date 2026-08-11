#!/bin/bash
#SBATCH --job-name=r048-k-sweep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r048-k-sweep-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r048-k-sweep-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
COLLECTION="$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_10164718"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

for K in 64 128; do
  "$PYTHON" scripts/analyze_fast_r048_oracle.py \
    --collection "$COLLECTION" \
    --target "$ASSETS/models/Qwen3-4B" \
    --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
    --output "$PROJECT/artifacts/analysis/r048_fast_k${K}_oracle_${SLURM_JOB_ID}.json" \
    --split validation_select \
    --candidate-topk "$K" \
    --batch-size 1 \
    --minimum-one-repair-eal 8.325485908649174 \
    --target-eal 8.325485908649174 \
    --maximum-required-oracle-recovery 0.90 \
    --minimum-frontier-count-coverage 0.95 \
    --minimum-oracle-reward-coverage 0.95
done
