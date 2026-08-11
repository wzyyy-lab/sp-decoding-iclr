#!/bin/bash
#SBATCH --job-name=r048-k32-oracle
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r048-k32-oracle-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r048-k32-oracle-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
COLLECTION="$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_10164718"
OUTPUT="$PROJECT/artifacts/analysis/r048_fast_k32_oracle_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/analyze_fast_r048_oracle.py \
  --collection "$COLLECTION" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$OUTPUT" \
  --split validation_select \
  --candidate-topk 32 \
  --batch-size 1 \
  --minimum-one-repair-eal 8.40 \
  --expected-released-eal 7.23955296404276 \
  --expected-eal-tolerance 0.000001 \
  --fail-on-gate
