#!/bin/bash
#SBATCH --job-name=sph-p3-t1-train
#SBATCH --partition=i64m1tga40u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-p3-t1-train-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-p3-t1-train-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
: "${DATA_JOB_ID:?submit with DATA_JOB_ID set to the Phase 3 collector job}"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_${DATA_JOB_ID}"
RUN_ROOT="$PROJECT/artifacts/training/phase3_tier1_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
for WEIGHT_LABEL in survival_w0 survival_w01; do
  if [[ "$WEIGHT_LABEL" == "survival_w0" ]]; then
    SURVIVAL_WEIGHT=0.0
  else
    SURVIVAL_WEIGHT=0.1
  fi
  for SEED in 0 1 2; do
    "$PYTHON" scripts/train_survival_head.py \
      --data "$DATA" \
      --target "$ASSETS/models/Qwen3-4B" \
      --candidate-k 16 \
      --head-type no_mixer \
      --normalization absorbing_crf \
      --rank 32 \
      --batch-size 64 \
      --epochs 10 \
      --learning-rate 0.0003 \
      --survival-loss-weight "$SURVIVAL_WEIGHT" \
      --seed "$SEED" \
      --train-split train \
      --validation-split validation_select \
      --gate-split validation_gate \
      --skip-test \
      --evidence-tier development \
      --output "$RUN_ROOT/$WEIGHT_LABEL/no_mixer_absorbing_crf_seed${SEED}"
  done
done

for WEIGHT_LABEL in survival_w0 survival_w01; do
  "$PYTHON" scripts/summarize_head_probe.py \
    --input "$RUN_ROOT/$WEIGHT_LABEL" \
    --output "$RUN_ROOT/${WEIGHT_LABEL}_summary.json" \
    --evaluation-split gate \
    --minimum-delta 0.2 \
    --minimum-base-delta 0.1
done
