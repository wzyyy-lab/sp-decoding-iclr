#!/bin/bash
#SBATCH --job-name=gfpr-cand-opb10k
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-cand-opb10k-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gfpr-cand-opb10k-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
SOURCE="$PROJECT/artifacts/canonical/qwen3_4b_domino_opb10k_balanced_v1"
OUT="$PROJECT/artifacts/canonical/gfpr_candidate_opb10k_fixed_${SLURM_JOB_ID}"
ANALYSIS="$PROJECT/artifacts/analysis/gfpr_candidate_opb10k_fixed_${SLURM_JOB_ID}.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/materialize_gfpr_candidate_rollout.py \
  --domino-rollout "$SOURCE" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$OUT" \
  --topk 16 \
  --batch-size 64 \
  --shard-blocks 512

"$PYTHON" scripts/analyze_gfpr_rollouts.py \
  --collection "$OUT" \
  --output "$ANALYSIS" \
  --split train \
  --expected-released-eal 7.258566570942808 \
  --expected-eal-tolerance 1e-9 \
  --fail-on-gate
