#!/bin/bash
#SBATCH --job-name=camrs-development
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/camrs-development-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/camrs-development-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
VALIDATION="$PROJECT/artifacts/canonical/qwen3_4b_phase3_validation_select_only_20260805"
COLLECTION="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770"
DIRECT_RUN="$PROJECT/artifacts/training/gcls_v4_feature_100k_10133585/compact_axial_additive_d64_full_seed0"
DIRECT_CONTROL="$PROJECT/artifacts/analysis/fmas_gate2/direct_one_edit_10133585.json"
OUTPUT="$PROJECT/artifacts/training/camrs_development_${SLURM_JOB_ID}/axial_additive_d64_tie_safe_max_regret_full_seed0"

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
echo "4720a07b062b6e77607ca867c97e9d8b37e64944962c25ac03c1320edb8ccf0b  scripts/train_first_miss_max_regret_development.py" | sha256sum -c -
echo "ca7e9a30e2f2450fde856ccd313b0b49e8ad31866016882a26fd33da4d980379  scripts/train_first_miss_max_regret_selector.py" | sha256sum -c -
echo "1a7a89be1ca525fc1c10ad61d22a006c2f7525210f4358699befa926f7e2d0ce  src/sph/first_miss_max_regret_selector.py" | sha256sum -c -
echo "caa9c5a611e7ead6d880c876e0a002b544115fc57dd2e43d46987b64195a82d2  src/sph/first_miss_value_selector.py" | sha256sum -c -
echo "2e27b4500fdd6d440078e40378c2aa56c09f09227072d7ffdc48bfaafddbcd10  scripts/train_first_miss_action_selector.py" | sha256sum -c -
echo "332fac8948c61b2287cb861988b700c8a249a91cec6f808c0010d47fe7260cef  src/sph/first_miss_action_selector.py" | sha256sum -c -
echo "802abd7fd8715e67a6b2cab5f33056a9ce5e17fb9af34723b0dd080850c450fd  scripts/evaluate_direct_one_edit.py" | sha256sum -c -
echo "e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110  scripts/train_global_direct_selector.py" | sha256sum -c -
echo "f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06  src/sph/global_direct_selector.py" | sha256sum -c -
echo "c811701d0ec097afa86e594946857290bcfff80e2cfc2e8638f0bcdaffcc0742  src/sph/data.py" | sha256sum -c -
echo "6815ae5cfbbc45d66df2baa22729d5b7e23faa527cfe67a6d2ea07a68cc40464  src/sph/__init__.py" | sha256sum -c -
echo "1fae55d19a6c59468c1d1be642718b6885fd2f4ac2d4cab642f34b487f79c0ab  src/sph/survival_path_head.py" | sha256sum -c -
echo "5f8a7808722744117e483959f302ab99a440c73679448277cf8d53182ab411f1  src/sph/candidate_ceiling.py" | sha256sum -c -
echo "58a92a38572dfca983361e257194a5d76f0a14f7eb9a196b8d7e749f584c7238  src/sph/first_miss_capacity.py" | sha256sum -c -
echo "4e0ed0f8208de0a7ab2e736e78a57323f72eb54e12b5b3b91063a9ef67137f22  refine-logs/first-miss-max-regret/FINAL_PROPOSAL.md" | sha256sum -c -
echo "ee60170019add2f57b04acd0dad08f56f72c9e0472e20bda5bc8ea2cdcfcdf5b  refine-logs/first-miss-max-regret/CAPACITY_RESULT_TO_CLAIM.md" | sha256sum -c -
echo "dd350d54cb73b66645323735983b5084265a168727010c25680e5bee901f5683  refine-logs/first-miss-max-regret/PRELAUNCH_CONTROL_FREEZE.md" | sha256sum -c -
echo "9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef  $DIRECT_RUN/metrics.json" | sha256sum -c -
echo "9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e  $DIRECT_RUN/best.pt" | sha256sum -c -
echo "b97c003745a96d2f2dde7e24425cf30d8e6c321257b746e922a794696e732b25  $DIRECT_CONTROL" | sha256sum -c -
echo "b63be7bbfd56651aadbee57a819bfe0afb39395b1601b5ea4fc1564cc9f933d7  $VALIDATION/metadata.json" | sha256sum -c -
echo "1496caa3d71ce64de9cd3fc2c29e40be60e9b636a988c9b400a0712e3ee5e811  $VALIDATION/selected_manifest.jsonl" | sha256sum -c -

TRAIN_METADATA_HASHES=(
  d64492233e6112daeee5f54c88cca16dffb3a2b4f98a54ad6c5f11b877935856
  a55331a31c9dc6efa4a376896ec5d6f7de828f104e3ffcb10da68546425240de
  4d78caf63e0a70c012d7382e20abacd8abd2029cc27de9885de88043628fd7e5
  0e447e1a3a40635525128ea36569b3e4a7424d1056aa683a81f0943b32f74d50
  d7597e41b7b16f533522f0f9f0d741f8de9ad1e3c8bcc3bf8557a24d8d31b42c
  03099d833d1ca9eaa32dcc1f468038ab0d58d8952c4220bcd0ed5e897ba5ec7a
  b9dcf53d64b38b1658831e09130cfb9a1a265685ede5a45e562b42e1be016c02
  48f015efdb40bc8ed32a41af0e9682d2090dd3dd9bb90e6e0906dfe8c6803585
)
for INDEX in {0..7}; do
  PART=$(printf '%03d' "$INDEX")
  echo "${TRAIN_METADATA_HASHES[$INDEX]}  $COLLECTION/part-$PART/metadata.json" | sha256sum -c -
done

nvidia-smi
"$PYTHON" scripts/train_first_miss_max_regret_development.py \
  --data "$VALIDATION" \
  "${TRAIN_ARGS[@]}" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$OUTPUT" \
  --direct-run "$DIRECT_RUN" \
  --direct-control "$DIRECT_CONTROL" \
  --expected-direct-metrics-sha256 9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef \
  --expected-direct-checkpoint-sha256 9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e \
  --expected-direct-control-sha256 b97c003745a96d2f2dde7e24425cf30d8e6c321257b746e922a794696e732b25 \
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
  --gradient-clip 1 \
  --seed 0 \
  --train-subset-seed 20260730 \
  --train-split train \
  --validation-split validation_select \
  --require-development-gate
