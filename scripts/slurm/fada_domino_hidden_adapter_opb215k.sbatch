#!/bin/bash
#SBATCH --job-name=fada-domino-hadapt-215k
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-hadapt-215k-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-hadapt-215k-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"

DIMS=(128 256)
HEADS=(4 8)
LAYERS=(1 2)
OBJECTIVES=(decay_ce breaker_margin)
LRS=(3e-5 1e-5)
PREFIX_WEIGHTS=(0.25 0.5)
APPLICATIONS=(base_only joint)

DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}
NUM_HEADS=${HEADS[$SLURM_ARRAY_TASK_ID]}
NUM_LAYERS=${LAYERS[$SLURM_ARRAY_TASK_ID]}
OBJECTIVE=${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
PREFIX_WEIGHT=${PREFIX_WEIGHTS[$SLURM_ARRAY_TASK_ID]}
APPLICATION=${APPLICATIONS[$SLURM_ARRAY_TASK_ID]}
OUTPUT="$PROJECT/artifacts/training/fada_domino_hadapt_215k_${OBJECTIVE}_d${DIM}_${SLURM_ARRAY_JOB_ID}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/train_domino_hidden_adapter.py \
  --canonical "$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_25k_seed20260730" \
  --additional-train-canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --eval-canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --adapter-dim "$DIM" \
  --adapter-heads "$NUM_HEADS" \
  --adapter-layers "$NUM_LAYERS" \
  --adapter-application "$APPLICATION" \
  --objective "$OBJECTIVE" \
  --epochs 2 \
  --batch-size 64 \
  --eval-batch-size 1 \
  --learning-rate "$LR" \
  --weight-decay 1e-4 \
  --prefix-weight "$PREFIX_WEIGHT" \
  --adapter-output-weight 1e-4 \
  --eval-every-steps 500 \
  --bootstrap-samples 5000 \
  --output "$OUTPUT"
