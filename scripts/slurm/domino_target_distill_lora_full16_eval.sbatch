#!/bin/bash
#SBATCH --job-name=domino-tdl-b16
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-tdl-b16-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/domino-tdl-b16-%j.err

set -euo pipefail
: "${ADAPTATION:?submit with --export=ALL,ADAPTATION=/absolute/path/to/best_adaptation.pt}"
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
OUTPUT="$PROJECT/artifacts/analysis/domino_target_distill_lora_full16_${SLURM_JOB_ID}.json"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/evaluate_domino_lora_full16.py \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --canonical artifacts/canonical/qwen3_4b_phase3_tier1_10035436 \
  --adaptation "$ADAPTATION" \
  --output "$OUTPUT" \
  --split validation_select \
  --bootstrap-samples 10000 \
  --seed 0 \
  --attn-implementation sdpa
