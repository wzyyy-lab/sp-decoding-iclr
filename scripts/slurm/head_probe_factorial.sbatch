#!/bin/bash
#SBATCH --job-name=sph-head-probe
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-head-probe-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/sph-head-probe-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_gate1_10022338"
RUN_ROOT="$PROJECT/artifacts/training/probe_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

cd "$PROJECT"
nvidia-smi
for NORMALIZATION in local absorbing_crf; do
  for SEED in 0 1 2; do
    "$PYTHON" scripts/train_survival_head.py \
      --data "$DATA" \
      --target "$ASSETS/models/Qwen3-4B" \
      --candidate-k 16 \
      --head-type no_mixer \
      --normalization "$NORMALIZATION" \
      --rank 32 \
      --batch-size 16 \
      --epochs 10 \
      --learning-rate 0.0003 \
      --survival-loss-weight 0.0 \
      --seed "$SEED" \
      --train-split train \
      --validation-split validation \
      --test-split test \
      --evidence-tier development \
      --output "$RUN_ROOT/no_mixer_${NORMALIZATION}_seed${SEED}"
  done
done

"$PYTHON" scripts/summarize_head_probe.py \
  --input "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.json" \
  --minimum-delta 0.2
