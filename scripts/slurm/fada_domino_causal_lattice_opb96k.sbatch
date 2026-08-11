#!/bin/bash
#SBATCH --job-name=fada-domino-cld-96k
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-cld-96k-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fada-domino-cld-96k-%A_%a.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"

DIMS=(64 128)
HEADS=(4 4)
TOPKS=(4 4)
OBJECTIVES=(decay_ce breaker_margin)
LRS=(1e-5 1e-5)
BATCHES=(64 32)

DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}
NUM_HEADS=${HEADS[$SLURM_ARRAY_TASK_ID]}
TOPK=${TOPKS[$SLURM_ARRAY_TASK_ID]}
OBJECTIVE=${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
BATCH=${BATCHES[$SLURM_ARRAY_TASK_ID]}
OUTPUT="$PROJECT/artifacts/training/fada_domino_cld_96k_k${TOPK}_${OBJECTIVE}_d${DIM}_${SLURM_ARRAY_JOB_ID}"
INIT="$PROJECT/artifacts/training/gcls_v4_feature_100k_10132819/compact_axial_additive_d64_full_seed0/best.pt"
EXTRA_ARGS=()
if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  EXTRA_ARGS+=(--init-selector-checkpoint "$INIT")
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts"

cd "$PROJECT"
"$PYTHON" scripts/train_domino_causal_lattice_decoder.py \
  --canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_opb10k_balanced_v1" \
  --additional-train-canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --eval-canonical "$PROJECT/artifacts/canonical/qwen3_4b_domino_phase3_v1" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --candidate-topk "$TOPK" \
  --model-dim "$DIM" \
  --num-heads "$NUM_HEADS" \
  --lattice-layers 1 \
  --decoder-layers 2 \
  --objective "$OBJECTIVE" \
  --epochs 1 \
  --batch-size "$BATCH" \
  --eval-batch-size 1 \
  --learning-rate "$LR" \
  --weight-decay 1e-4 \
  --prefix-weight 0.5 \
  --residual-penalty-weight 1e-4 \
  --eval-every-steps 500 \
  --bootstrap-samples 5000 \
  "${EXTRA_ARGS[@]}" \
  --output "$OUTPUT"
