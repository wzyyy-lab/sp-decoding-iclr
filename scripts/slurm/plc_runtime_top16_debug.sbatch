#!/bin/bash
#SBATCH --job-name=plc-top16
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/plc-top16-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/plc-top16-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
: "${PLC_CANONICAL:?set PLC_CANONICAL to the completed runtime teacher cache}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
"$PYTHON" scripts/analyze_plc_runtime_top16.py \
  --canonical "$PLC_CANONICAL" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "artifacts/analysis/plc_runtime_top16_${SLURM_JOB_ID}.json" \
  --split validation_select \
  --batch-size 32
