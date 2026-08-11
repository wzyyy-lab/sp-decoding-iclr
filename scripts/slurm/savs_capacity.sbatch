#!/bin/bash
#SBATCH --job-name=savs-cap
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/savs-cap-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/savs-cap-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
MANIFEST="$PROJECT/artifacts/manifests/fmas_capacity_512_seed0.json"
RUN_ROOT="$PROJECT/artifacts/training/savs_capacity_${SLURM_JOB_ID}"
LABEL=axial_additive_d64_signed_value_mse_seed0

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
echo "4b3b02ffc3c413034bf445d3352e09b24e68ccb10fb16c9f63e07240818ad483  scripts/train_first_miss_value_selector.py" | sha256sum -c -
echo "caa9c5a611e7ead6d880c876e0a002b544115fc57dd2e43d46987b64195a82d2  src/sph/first_miss_value_selector.py" | sha256sum -c -
echo "2e27b4500fdd6d440078e40378c2aa56c09f09227072d7ffdc48bfaafddbcd10  scripts/train_first_miss_action_selector.py" | sha256sum -c -
echo "332fac8948c61b2287cb861988b700c8a249a91cec6f808c0010d47fe7260cef  src/sph/first_miss_action_selector.py" | sha256sum -c -
echo "58a92a38572dfca983361e257194a5d76f0a14f7eb9a196b8d7e749f584c7238  src/sph/first_miss_capacity.py" | sha256sum -c -
echo "c811701d0ec097afa86e594946857290bcfff80e2cfc2e8638f0bcdaffcc0742  src/sph/data.py" | sha256sum -c -
echo "e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110  scripts/train_global_direct_selector.py" | sha256sum -c -
echo "f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06  src/sph/global_direct_selector.py" | sha256sum -c -
nvidia-smi
"$PYTHON" scripts/train_first_miss_value_selector.py \
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
  --max-value-rmse 0.02 \
  --min-beneficial-sign-recall 0.99 \
  --min-harmful-nonpositive-recall 0.99 \
  --min-oracle-gap-recovered 0.95 \
  --max-harmed-fraction 0.01 \
  --expected-beneficial-actions 256 \
  --expected-train-blocks 512 \
  --expected-train-prompts 459 \
  --expected-train-prompt-sha256 1e2be08968b2356f71e9818a5be5b8f3ecdd12ee50299ba6212a035f8a4d2707 \
  --expected-total-steps 5120
