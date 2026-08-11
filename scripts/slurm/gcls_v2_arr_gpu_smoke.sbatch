#!/bin/bash
#SBATCH --job-name=gcls-v2-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-smoke-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
RUN_ROOT="$PROJECT/artifacts/training/gcls_v2_smoke_${SLURM_JOB_ID}"

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
nvidia-smi
"$PYTHON" -c 'import torch,transformers; print("ENV_WITNESS", torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))'
"$PYTHON" -c 'import torch; torch.manual_seed(0); x=torch.randn(8,8,device="cuda"); y=x@x; print("KERNEL_WITNESS", tuple(y.shape), y.numel(), torch.cuda.get_device_name(0))'
"$PYTHON" scripts/train_global_direct_selector.py \
  --data "$DATA" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$RUN_ROOT/flat_compat_global_reach_lam0p1_seed0" \
  --scope global \
  --mixer flat \
  --node-encoder compatibility \
  --candidate-k 16 \
  --model-dim 128 \
  --num-heads 8 \
  --num-layers 2 \
  --dropout 0 \
  --batch-size 16 \
  --epochs 5 \
  --learning-rate 0.0006 \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --gradient-clip 1.0 \
  --loss-weighting accepted_reach \
  --base-safety-weight 0.1 \
  --base-safety-margin 0.1 \
  --seed 0 \
  --memorization-blocks 128 \
  --memorization-opportunity-fraction 0.5 \
  --skip-gate \
  --evidence-tier smoke
