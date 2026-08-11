#!/bin/bash
#SBATCH --job-name=pros-outcome
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-outcome-%A_%a.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-outcome-%A_%a.err

set -euo pipefail
: "${SPLIT_MANIFEST_SHA256:?submit with the reviewed split-manifest SHA256}"
: "${SPLIT_AUDIT_SHA256:?submit with the independent split-audit SHA256}"

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
DIRECT_RUN="$PROJECT/artifacts/training/gcls_v4_feature_100k_10133585/compact_axial_additive_d64_full_seed0"
SPLIT_MANIFEST="$PROJECT/artifacts/pros_gate/r079_numeric_v2/split_manifest.json"
SPLIT_AUDIT="$PROJECT/artifacts/pros_gate/r079_numeric_v2/audits/split.json"
SOURCE_MANIFEST_SHA256=2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4

case "$SLURM_ARRAY_TASK_ID" in
  0) OUTCOME_SPLIT=fit ;;
  1) OUTCOME_SPLIT=checkpoint ;;
  *) echo "unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 64 ;;
esac
OUTPUT="$PROJECT/artifacts/pros_gate/r079_numeric_v2/outcomes/$OUTCOME_SPLIT"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$PROJECT"
echo "28545c2edd2c3e8b4a404b50f7c5f6fedbf2348a33349106d2d7feb538d853f4  scripts/materialize_direct_safety_artifacts.py" | sha256sum -c -
echo "852915273e330344c1afe472f9d0c2b4789b564e6636af897051486fbaa3a6d9  src/sph/direct_safety_artifacts.py" | sha256sum -c -
echo "3b2f0fbab3a0a683a7f5e142b88f79c96960b6427e48253861c090e85768437c  src/sph/direct_safety_numeric_policy.py" | sha256sum -c -
echo "e3bd6392f7430e60e0eef16217dc904eeb018313ae8d4f543bd089a1943739b6  src/sph/direct_safety_gate.py" | sha256sum -c -
echo "bdde815e546993edb039e675e991cf6353477a62c24ad69215f056fb545ee24b  src/sph/direct_safety_protocol.py" | sha256sum -c -
echo "802abd7fd8715e67a6b2cab5f33056a9ce5e17fb9af34723b0dd080850c450fd  scripts/evaluate_direct_one_edit.py" | sha256sum -c -
echo "e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110  scripts/train_global_direct_selector.py" | sha256sum -c -
echo "f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06  src/sph/global_direct_selector.py" | sha256sum -c -
echo "c811701d0ec097afa86e594946857290bcfff80e2cfc2e8638f0bcdaffcc0742  src/sph/data.py" | sha256sum -c -
echo "9fac5193a09c3def52b8e1360c0790d2497597f4daa8f5b67a16df5bbb1a31d4  src/sph/source_closure.py" | sha256sum -c -
echo "2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4  refine-logs/direct-safety-gate/R079_SOURCE_CLOSURE_NUMERIC_V2.json" | sha256sum -c -
echo "c93325260be458d9904d25503670a97a4d04c277837a77cb4e22f5178db0091a  scripts/verify_pros_gate_receipt.py" | sha256sum -c -
"$PYTHON" src/sph/source_closure.py \
  --project "$PROJECT" \
  --expected-manifest-sha256 "$SOURCE_MANIFEST_SHA256"
echo "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320  $DATA/metadata.json" | sha256sum -c -
echo "$SPLIT_MANIFEST_SHA256  $SPLIT_MANIFEST" | sha256sum -c -
echo "$SPLIT_AUDIT_SHA256  $SPLIT_AUDIT" | sha256sum -c -
"$PYTHON" scripts/verify_pros_gate_receipt.py split \
  --project "$PROJECT" \
  --receipt "$SPLIT_AUDIT" \
  --expected-receipt-sha256 "$SPLIT_AUDIT_SHA256" \
  --split-manifest-sha256 "$SPLIT_MANIFEST_SHA256" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256"
echo "9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef  $DIRECT_RUN/metrics.json" | sha256sum -c -
echo "9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e  $DIRECT_RUN/best.pt" | sha256sum -c -
test "$(stat -c %s artifacts/manifests/open_perfectblend_100k_v2.jsonl)" -eq 92590866
test "$(stat -c %s artifacts/manifests/phase3_development_v3.jsonl)" -eq 1418202
test "$(stat -c %s artifacts/manifests/phase3_reserved_test_v3.jsonl)" -eq 429359
echo "b05087a56e8e717605415026421f7bae23092eb7cb9509361a36932f80260e3a  artifacts/manifests/open_perfectblend_100k_v2.jsonl" | sha256sum -c -
echo "e16374068e9c8904214fbf282b4adb6187a0b099db5c37e79660fc46a2801d01  artifacts/manifests/phase3_development_v3.jsonl" | sha256sum -c -
echo "ae25467fbb52b7091c8d9a5f98776b11ccf76e87e781850b5638734548a53bb4  artifacts/manifests/phase3_reserved_test_v3.jsonl" | sha256sum -c -
nvidia-smi

"$PYTHON" scripts/materialize_direct_safety_artifacts.py outcomes \
  --data "$DATA" \
  --split-manifest "$SPLIT_MANIFEST" \
  --expected-split-manifest-sha256 "$SPLIT_MANIFEST_SHA256" \
  --split "$OUTCOME_SPLIT" \
  --direct-run "$DIRECT_RUN" \
  --target "$ASSETS/models/Qwen3-4B" \
  --output "$OUTPUT" \
  --batch-size 32 \
  --expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --exclusion "producer_train=$PROJECT/artifacts/manifests/open_perfectblend_100k_v2.jsonl" \
  --exclusion "validation=$PROJECT/artifacts/manifests/phase3_development_v3.jsonl" \
  --exclusion "reserved=$PROJECT/artifacts/manifests/phase3_reserved_test_v3.jsonl"

sha256sum "$OUTPUT/metadata.json"
