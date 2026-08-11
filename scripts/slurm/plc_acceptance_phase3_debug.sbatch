#!/bin/bash
#SBATCH --job-name=plc-accept
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/plc-accept-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/plc-accept-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
: "${PLC_CANONICAL:?set PLC_CANONICAL to the completed runtime teacher cache}"
: "${PLC_INIT_CHECKPOINT:?set PLC_INIT_CHECKPOINT to a PLC imitation checkpoint}"
PRESERVE_WEIGHT="${PLC_PRESERVE_WEIGHT:-0.5}"
SURVIVAL_FLOOR="${PLC_SURVIVAL_FLOOR:-0.5}"
OBJECTIVE="${PLC_OBJECTIVE:-ce}"
MARGIN_WEIGHT="${PLC_MARGIN_WEIGHT:-1.0}"
MARGIN_OFFSET="${PLC_MARGIN_OFFSET:-1.0}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"
"$PYTHON" scripts/train_plc_acceptance.py \
  --canonical "$PLC_CANONICAL" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --init-checkpoint "$PLC_INIT_CHECKPOINT" \
  --output "artifacts/training/plc_acceptance_${SLURM_JOB_ID}" \
  --epochs 8 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --learning-rate 1e-4 \
  --preserve-weight "$PRESERVE_WEIGHT" \
  --survival-floor "$SURVIVAL_FLOOR" \
  --objective "$OBJECTIVE" \
  --margin-weight "$MARGIN_WEIGHT" \
  --margin-offset "$MARGIN_OFFSET" \
  --code-loss-weight 0.05
