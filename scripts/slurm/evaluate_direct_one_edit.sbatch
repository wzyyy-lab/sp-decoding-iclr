#!/bin/bash
#SBATCH --job-name=direct-1edit
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/direct-1edit-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/direct-1edit-%j.err

set -euo pipefail
: "${DIRECT_JOB_ID:?submit with the matched Direct array job id}"

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
VALIDATION="$PROJECT/artifacts/canonical/qwen3_4b_phase3_validation_select_only_20260805"
DIRECT_RUN="$PROJECT/artifacts/training/gcls_v4_feature_100k_${DIRECT_JOB_ID}/compact_axial_additive_d64_full_seed0"
OUTPUT="$PROJECT/artifacts/analysis/fmas_gate2/direct_one_edit_${DIRECT_JOB_ID}.json"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

cd "$PROJECT"
echo "802abd7fd8715e67a6b2cab5f33056a9ce5e17fb9af34723b0dd080850c450fd  scripts/evaluate_direct_one_edit.py" | sha256sum -c -
echo "2e27b4500fdd6d440078e40378c2aa56c09f09227072d7ffdc48bfaafddbcd10  scripts/train_first_miss_action_selector.py" | sha256sum -c -
echo "332fac8948c61b2287cb861988b700c8a249a91cec6f808c0010d47fe7260cef  src/sph/first_miss_action_selector.py" | sha256sum -c -
echo "e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110  scripts/train_global_direct_selector.py" | sha256sum -c -
echo "f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06  src/sph/global_direct_selector.py" | sha256sum -c -
echo "c811701d0ec097afa86e594946857290bcfff80e2cfc2e8638f0bcdaffcc0742  src/sph/data.py" | sha256sum -c -
echo "b63be7bbfd56651aadbee57a819bfe0afb39395b1601b5ea4fc1564cc9f933d7  $VALIDATION/metadata.json" | sha256sum -c -
echo "1496caa3d71ce64de9cd3fc2c29e40be60e9b636a988c9b400a0712e3ee5e811  $VALIDATION/selected_manifest.jsonl" | sha256sum -c -
nvidia-smi
"$PYTHON" scripts/evaluate_direct_one_edit.py \
  --direct-run "$DIRECT_RUN" \
  --data "$VALIDATION" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$OUTPUT" \
  --validation-split validation_select \
  --batch-size 64
