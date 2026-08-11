#!/bin/bash
#SBATCH --job-name=r047-anchor-phase3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r047-anchor-phase3-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r047-anchor-phase3-%j.err

set -euo pipefail
: "${R047_COLLECTION_JOB:?submit with --export=ALL,R047_COLLECTION_JOB=<job>}"
: "${R047_ALIGNMENT_REPORT:?submit with a passed trained-checkpoint alignment report}"
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
TRAIN="$PROJECT/artifacts/canonical/r047_anchor_t4_train_${R047_COLLECTION_JOB}"
EVAL="$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_${R047_COLLECTION_JOB}"
OUT="$PROJECT/artifacts/models/r047_anchor_t4_phase3_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT/src:$PROJECT/scripts"
cd "$PROJECT"

"$PYTHON" scripts/train_gfpr_candidate.py \
  --train-rollout "$TRAIN" \
  --eval-rollout "$EVAL" \
  --target "$ASSETS/models/Qwen3-4B" \
  --domino-draft "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --output "$OUT" \
  --train-split train \
  --eval-split validation_select \
  --candidate-topk 16 \
  --include-released-action \
  --released-union-teacher stored_frontier \
  --trainable-scope adapter \
  --adapter-rank 64 \
  --use-target-anchor-early-feature \
  --target-anchor-early-exit-layer 4 \
  --required-alignment-report "$R047_ALIGNMENT_REPORT" \
  --epochs 4 \
  --prompts-per-batch 4 \
  --gradient-accumulation-prompts 16 \
  --eval-batch-size 1 \
  --base-learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --delta-l2 1e-4 \
  --frontier-weight 0 \
  --target-kl-weight 1 \
  --target-advantage-weight 0 \
  --target-temperature 2 \
  --target-protect-weight 1 \
  --target-repair-weight 4 \
  --eval-every-steps 50 \
  --bootstrap-samples 10000 \
  --save-every-eval \
  --seed 0
