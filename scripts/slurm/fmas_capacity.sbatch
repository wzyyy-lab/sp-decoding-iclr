#!/bin/bash
#SBATCH --job-name=fmas-cap
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fmas-cap-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fmas-cap-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
MANIFEST="$PROJECT/artifacts/manifests/fmas_capacity_512_seed0.json"
RUN_ROOT="$PROJECT/artifacts/training/fmas_capacity_${SLURM_JOB_ID}"
LABEL=axial_additive_d64_action_ce_seed0

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd "$PROJECT"
echo "d60613a00fc8557f4ff227ec302ced42de6a071d030b7ae7eb9eb5120bf5b67f  $MANIFEST" | sha256sum -c -
nvidia-smi
"$PYTHON" scripts/train_first_miss_action_selector.py \
  --data "$DATA" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/$LABEL" \
  --scope global \
  --mixer axial \
  --node-encoder additive \
  --candidate-k 16 \
  --model-dim 64 \
  --num-heads 4 \
  --num-layers 1 \
  --dropout 0 \
  --batch-size 32 \
  --epochs 320 \
  --learning-rate 0.0006 \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --gradient-clip 1.0 \
  --seed 0 \
  --memorization-blocks 512 \
  --memorization-opportunity-fraction 0.5 \
  --capacity-manifest "$MANIFEST" \
  --require-capacity-gate \
  --evidence-tier capacity_probe \
  --min-action-accuracy 0.97 \
  --min-repairable-action-recall 0.95 \
  --min-oracle-gap-recovered 0.95 \
  --max-harmed-fraction 0.01 \
  --expected-train-blocks 512 \
  --expected-total-steps 5120
