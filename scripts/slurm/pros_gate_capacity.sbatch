#!/bin/bash
#SBATCH --job-name=pros-cap
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-cap-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-cap-%j.err

set -euo pipefail
: "${CAPACITY_METADATA_SHA256:?provide independently audited capacity metadata SHA256}"
: "${CAPACITY_AUDIT_SHA256:?provide independent capacity-artifact audit SHA256}"
: "${FIT_METADATA_SHA256:?provide reviewed fit metadata SHA256}"
: "${SPLIT_MANIFEST_SHA256:?provide reviewed split-manifest SHA256}"

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
CAPACITY="$PROJECT/artifacts/pros_gate/r079_numeric_v2/capacity"
OUTPUT="$PROJECT/artifacts/training/pros_gate_capacity_${SLURM_JOB_ID}/seed0"
CAPACITY_AUDIT="$PROJECT/artifacts/pros_gate/r079_numeric_v2/audits/capacity.json"
FIT_METADATA="$PROJECT/artifacts/pros_gate/r079_numeric_v2/outcomes/fit/metadata.json"
SPLIT_MANIFEST="$PROJECT/artifacts/pros_gate/r079_numeric_v2/split_manifest.json"
SOURCE_MANIFEST_SHA256=2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$PROJECT"
echo "2a8ea20a466096f540fc0d001f9dea0a766dc5c1773d351408f9e12d4b57e2f6  scripts/train_direct_safety_capacity.py" | sha256sum -c -
echo "852915273e330344c1afe472f9d0c2b4789b564e6636af897051486fbaa3a6d9  src/sph/direct_safety_artifacts.py" | sha256sum -c -
echo "3b2f0fbab3a0a683a7f5e142b88f79c96960b6427e48253861c090e85768437c  src/sph/direct_safety_numeric_policy.py" | sha256sum -c -
echo "e3bd6392f7430e60e0eef16217dc904eeb018313ae8d4f543bd089a1943739b6  src/sph/direct_safety_gate.py" | sha256sum -c -
echo "bdde815e546993edb039e675e991cf6353477a62c24ad69215f056fb545ee24b  src/sph/direct_safety_protocol.py" | sha256sum -c -
echo "9fac5193a09c3def52b8e1360c0790d2497597f4daa8f5b67a16df5bbb1a31d4  src/sph/source_closure.py" | sha256sum -c -
echo "2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4  refine-logs/direct-safety-gate/R079_SOURCE_CLOSURE_NUMERIC_V2.json" | sha256sum -c -
echo "c93325260be458d9904d25503670a97a4d04c277837a77cb4e22f5178db0091a  scripts/verify_pros_gate_receipt.py" | sha256sum -c -
"$PYTHON" src/sph/source_closure.py \
  --project "$PROJECT" \
  --expected-manifest-sha256 "$SOURCE_MANIFEST_SHA256"
echo "$CAPACITY_METADATA_SHA256  $CAPACITY/metadata.json" | sha256sum -c -
echo "$FIT_METADATA_SHA256  $FIT_METADATA" | sha256sum -c -
echo "$SPLIT_MANIFEST_SHA256  $SPLIT_MANIFEST" | sha256sum -c -
echo "$CAPACITY_AUDIT_SHA256  $CAPACITY_AUDIT" | sha256sum -c -
"$PYTHON" scripts/verify_pros_gate_receipt.py capacity \
  --receipt "$CAPACITY_AUDIT" \
  --expected-receipt-sha256 "$CAPACITY_AUDIT_SHA256" \
  --capacity-metadata-sha256 "$CAPACITY_METADATA_SHA256" \
  --fit-metadata-sha256 "$FIT_METADATA_SHA256" \
  --split-manifest-sha256 "$SPLIT_MANIFEST_SHA256" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256"
nvidia-smi

"$PYTHON" scripts/train_direct_safety_capacity.py \
  --capacity-bundle "$CAPACITY" \
  --expected-capacity-metadata-sha256 "$CAPACITY_METADATA_SHA256" \
  --expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --output "$OUTPUT"
