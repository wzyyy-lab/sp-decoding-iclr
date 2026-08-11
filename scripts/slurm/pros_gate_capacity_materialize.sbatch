#!/bin/bash
#SBATCH --job-name=pros-cap-data
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-cap-data-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-cap-data-%j.err

set -euo pipefail
: "${SPLIT_MANIFEST_SHA256:?provide reviewed split-manifest SHA256}"
: "${FIT_METADATA_SHA256:?provide reviewed fit metadata SHA256}"
: "${CHECKPOINT_METADATA_SHA256:?provide reviewed checkpoint metadata SHA256}"
: "${OUTCOMES_AUDIT_SHA256:?provide independent outcome-audit SHA256}"

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
ROOT="$PROJECT/artifacts/pros_gate/r079_numeric_v2"
OUTCOMES_AUDIT="$ROOT/audits/outcomes.json"
SPLIT_MANIFEST="$ROOT/split_manifest.json"
FIT_METADATA="$ROOT/outcomes/fit/metadata.json"
CHECKPOINT_METADATA="$ROOT/outcomes/checkpoint/metadata.json"
SOURCE_MANIFEST_SHA256=2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$PROJECT"
echo "28545c2edd2c3e8b4a404b50f7c5f6fedbf2348a33349106d2d7feb538d853f4  scripts/materialize_direct_safety_artifacts.py" | sha256sum -c -
echo "852915273e330344c1afe472f9d0c2b4789b564e6636af897051486fbaa3a6d9  src/sph/direct_safety_artifacts.py" | sha256sum -c -
echo "3b2f0fbab3a0a683a7f5e142b88f79c96960b6427e48253861c090e85768437c  src/sph/direct_safety_numeric_policy.py" | sha256sum -c -
echo "bdde815e546993edb039e675e991cf6353477a62c24ad69215f056fb545ee24b  src/sph/direct_safety_protocol.py" | sha256sum -c -
echo "9fac5193a09c3def52b8e1360c0790d2497597f4daa8f5b67a16df5bbb1a31d4  src/sph/source_closure.py" | sha256sum -c -
echo "2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4  refine-logs/direct-safety-gate/R079_SOURCE_CLOSURE_NUMERIC_V2.json" | sha256sum -c -
echo "c93325260be458d9904d25503670a97a4d04c277837a77cb4e22f5178db0091a  scripts/verify_pros_gate_receipt.py" | sha256sum -c -
"$PYTHON" src/sph/source_closure.py \
  --project "$PROJECT" \
  --expected-manifest-sha256 "$SOURCE_MANIFEST_SHA256"
echo "$SPLIT_MANIFEST_SHA256  $SPLIT_MANIFEST" | sha256sum -c -
echo "$FIT_METADATA_SHA256  $FIT_METADATA" | sha256sum -c -
echo "$CHECKPOINT_METADATA_SHA256  $CHECKPOINT_METADATA" | sha256sum -c -
echo "$OUTCOMES_AUDIT_SHA256  $OUTCOMES_AUDIT" | sha256sum -c -
"$PYTHON" scripts/verify_pros_gate_receipt.py outcomes \
  --receipt "$OUTCOMES_AUDIT" \
  --expected-receipt-sha256 "$OUTCOMES_AUDIT_SHA256" \
  --split-manifest-sha256 "$SPLIT_MANIFEST_SHA256" \
  --fit-metadata-sha256 "$FIT_METADATA_SHA256" \
  --checkpoint-metadata-sha256 "$CHECKPOINT_METADATA_SHA256" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256"

"$PYTHON" scripts/materialize_direct_safety_artifacts.py capacity \
  --fit-bundle "$ROOT/outcomes/fit" \
  --expected-fit-metadata-sha256 "$FIT_METADATA_SHA256" \
  --output "$ROOT/capacity" \
  --producer-checkpoint-sha256 9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e \
  --producer-metrics-sha256 9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef \
  --canonical-metadata-sha256 0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320 \
  --split-manifest-sha256 "$SPLIT_MANIFEST_SHA256" \
  --expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA256"

sha256sum "$ROOT/capacity/metadata.json"
