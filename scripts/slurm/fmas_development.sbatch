#!/bin/bash
#SBATCH --job-name=fmas-dev
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fmas-dev-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/fmas-dev-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
VALIDATION="$PROJECT/artifacts/canonical/qwen3_4b_phase3_validation_select_only_20260805"
COLLECTION="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770"
RUN_ROOT="$PROJECT/artifacts/training/fmas_development_${SLURM_JOB_ID}"
LABEL=axial_additive_d64_action_ce_full_seed0

TRAIN_ARGS=()
for INDEX in {0..7}; do
  PART=$(printf '%03d' "$INDEX")
  TRAIN_ARGS+=(--train-data "$COLLECTION/part-$PART")
done

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
echo "2e27b4500fdd6d440078e40378c2aa56c09f09227072d7ffdc48bfaafddbcd10  scripts/train_first_miss_action_selector.py" | sha256sum -c -
echo "332fac8948c61b2287cb861988b700c8a249a91cec6f808c0010d47fe7260cef  src/sph/first_miss_action_selector.py" | sha256sum -c -
echo "58a92a38572dfca983361e257194a5d76f0a14f7eb9a196b8d7e749f584c7238  src/sph/first_miss_capacity.py" | sha256sum -c -
echo "e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110  scripts/train_global_direct_selector.py" | sha256sum -c -
echo "f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06  src/sph/global_direct_selector.py" | sha256sum -c -
echo "c811701d0ec097afa86e594946857290bcfff80e2cfc2e8638f0bcdaffcc0742  src/sph/data.py" | sha256sum -c -
echo "b3509d04bd5e2a22b342d985c9c1eea1715550f78dee416198f756c4ea6ff055  scripts/materialize_canonical_split.py" | sha256sum -c -
echo "b63be7bbfd56651aadbee57a819bfe0afb39395b1601b5ea4fc1564cc9f933d7  $VALIDATION/metadata.json" | sha256sum -c -
echo "1496caa3d71ce64de9cd3fc2c29e40be60e9b636a988c9b400a0712e3ee5e811  $VALIDATION/selected_manifest.jsonl" | sha256sum -c -
echo "d64492233e6112daeee5f54c88cca16dffb3a2b4f98a54ad6c5f11b877935856  $COLLECTION/part-000/metadata.json" | sha256sum -c -
echo "a55331a31c9dc6efa4a376896ec5d6f7de828f104e3ffcb10da68546425240de  $COLLECTION/part-001/metadata.json" | sha256sum -c -
echo "4d78caf63e0a70c012d7382e20abacd8abd2029cc27de9885de88043628fd7e5  $COLLECTION/part-002/metadata.json" | sha256sum -c -
echo "0e447e1a3a40635525128ea36569b3e4a7424d1056aa683a81f0943b32f74d50  $COLLECTION/part-003/metadata.json" | sha256sum -c -
echo "d7597e41b7b16f533522f0f9f0d741f8de9ad1e3c8bcc3bf8557a24d8d31b42c  $COLLECTION/part-004/metadata.json" | sha256sum -c -
echo "03099d833d1ca9eaa32dcc1f468038ab0d58d8952c4220bcd0ed5e897ba5ec7a  $COLLECTION/part-005/metadata.json" | sha256sum -c -
echo "b9dcf53d64b38b1658831e09130cfb9a1a265685ede5a45e562b42e1be016c02  $COLLECTION/part-006/metadata.json" | sha256sum -c -
echo "48f015efdb40bc8ed32a41af0e9682d2090dd3dd9bb90e6e0906dfe8c6803585  $COLLECTION/part-007/metadata.json" | sha256sum -c -
nvidia-smi
"$PYTHON" scripts/train_first_miss_action_selector.py \
  --data "$VALIDATION" \
  "${TRAIN_ARGS[@]}" \
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
  --batch-size 64 \
  --epochs 3 \
  --learning-rate 0.0006 \
  --weight-decay 0 \
  --warmup-ratio 0.04 \
  --gradient-clip 1.0 \
  --seed 0 \
  --train-subset-seed 20260730 \
  --train-split train \
  --validation-split validation_select \
  --evidence-tier development \
  --expected-train-blocks 793989 \
  --expected-train-prompts 99356 \
  --expected-total-steps 37221 \
  --expected-train-prompt-sha256 45471a62f93a488f3f7653c096bebcddb0ddae3773f6c99744bd070e348a9405 \
  --skip-final-train-diagnostic
