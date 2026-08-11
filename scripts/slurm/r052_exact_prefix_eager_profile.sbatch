#!/bin/bash
#SBATCH --job-name=r052-prefix-profile
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r052-prefix-profile-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r052-prefix-profile-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUT="$PROJECT/profile_output/r052_exact_prefix_eager_${SLURM_JOB_ID}.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
"$PYTHON" scripts/profile_r052_exact_prefix.py \
  --source-rollout "$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_10164718" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --r051-report "$PROJECT/artifacts/analysis/r051_exact_prefix_fixed_10165029.json" \
  --output "$OUT" --split validation_select --candidate-topk 64 \
  --warmup 5 --repeats 30
