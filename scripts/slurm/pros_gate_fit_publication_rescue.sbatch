#!/bin/bash
#SBATCH --job-name=pros-fit-pub
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-fit-pub-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-fit-pub-%j.err

set -euo pipefail
umask 077

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
FIT="$PROJECT/artifacts/pros_gate/r079_numeric_v2/outcomes/fit"
CHECKPOINT="$PROJECT/artifacts/pros_gate/r079_numeric_v2/outcomes/checkpoint"
OUTCOMES_AUDIT="$PROJECT/artifacts/pros_gate/r079_numeric_v2/audits/outcomes.json"
CAPACITY_ADJUDICATION="$PROJECT/artifacts/adjudication/pros_gate_capacity_10138104/capacity_adjudication_v2.json"
SOURCE_MANIFEST="$PROJECT/refine-logs/direct-safety-gate/R082_SOURCE_CLOSURE_PUBLICATION_RESCUE_V1.json"
SOURCE_MANIFEST_SHA256=f36291a961ea793dbaa888950bc4312d8b53954fcc5ecdb01a5caad4af97e184
RUN_ROOT="$PROJECT/artifacts/training/pros_gate_fit_${SLURM_JOB_ID}"
OUTPUT="$RUN_ROOT/seed0"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$PROJECT"
read -r WRAPPER_SHA256 _ < <(sha256sum "$0")
echo "wrapper_sha256=$WRAPPER_SHA256"

echo "9bc069a5d70ff6f074e323a076c065eb7ae8ff903f4ed83cd1f6a317b790bff4  scripts/train_direct_safety_fit.py" | sha256sum -c -
echo "844e2aac1149d88af5710a3186ba3e6f8384f6dc346a2ad151a534e9ca74e4dd  scripts/verify_pros_gate_receipt.py" | sha256sum -c -
echo "852915273e330344c1afe472f9d0c2b4789b564e6636af897051486fbaa3a6d9  src/sph/direct_safety_artifacts.py" | sha256sum -c -
echo "a8aa727c0dc6e82c7d0c1a2ac65f2cef78941ac4619f02bf0740d3bb03d7b30f  src/sph/direct_safety_gate.py" | sha256sum -c -
echo "47f91341c7d222fa8a62d55b9aad16d75e54582d30e632aa8b9d63e6a678a0ea  src/sph/direct_safety_protocol.py" | sha256sum -c -
echo "3b2f0fbab3a0a683a7f5e142b88f79c96960b6427e48253861c090e85768437c  src/sph/direct_safety_numeric_policy.py" | sha256sum -c -
echo "9fac5193a09c3def52b8e1360c0790d2497597f4daa8f5b67a16df5bbb1a31d4  src/sph/source_closure.py" | sha256sum -c -
echo "f36291a961ea793dbaa888950bc4312d8b53954fcc5ecdb01a5caad4af97e184  refine-logs/direct-safety-gate/R082_SOURCE_CLOSURE_PUBLICATION_RESCUE_V1.json" | sha256sum -c -
"$PYTHON" src/sph/source_closure.py \
  --project "$PROJECT" \
  --manifest "$SOURCE_MANIFEST" \
  --expected-manifest-sha256 "$SOURCE_MANIFEST_SHA256"

# This exact-protocol GPFS smoke precedes every records.pt hash/open and every
# GPU update.  Its committed dummy bundle and parent pending receipt remain in
# this job-scoped directory for audit.
mkdir -p "$RUN_ROOT"
"$PYTHON" -c 'import json,sys; from pathlib import Path; from scripts.train_direct_safety_fit import publication_filesystem_smoke; print(json.dumps(publication_filesystem_smoke(Path(sys.argv[1]), job_id=sys.argv[2], wrapper_sha256=sys.argv[3], source_manifest=Path(sys.argv[4]), source_manifest_sha256=sys.argv[5]), sort_keys=True))' "$RUN_ROOT" "$SLURM_JOB_ID" "$WRAPPER_SHA256" "$SOURCE_MANIFEST" "$SOURCE_MANIFEST_SHA256"

echo "061069ed644b7fd700d7b65586622c02ef878c611ab6a549968f78bba8425f98  artifacts/pros_gate/r079_numeric_v2/outcomes/fit/metadata.json" | sha256sum -c -
echo "645007ec2665e141813b09e4bd1e35c33337b4b32e27655ae088c52c89fbcc6b  artifacts/pros_gate/r079_numeric_v2/outcomes/fit/records.pt" | sha256sum -c -
echo "cdc879ef861608c6f26e004e4dd27826554f4c5631929e68f8dc57e8ea753047  artifacts/pros_gate/r079_numeric_v2/outcomes/checkpoint/metadata.json" | sha256sum -c -
echo "203cadf7141684b91b7c41d70fc0222098898ca2f9200825ccd60fc8ecbb93a2  artifacts/pros_gate/r079_numeric_v2/outcomes/checkpoint/records.pt" | sha256sum -c -
echo "29b0c83a26b3f7fa30830e596cb070176b5c2738f6c4d2728b0ebde7bc87f36d  artifacts/pros_gate/r079_numeric_v2/audits/outcomes.json" | sha256sum -c -
echo "17ac807e5b599c45e414958786fade47d7b2a0e1fd5603c3726d531eb143a352  artifacts/adjudication/pros_gate_capacity_10138104/capacity_adjudication_v2.json" | sha256sum -c -

"$PYTHON" scripts/verify_pros_gate_receipt.py outcomes \
  --receipt "$OUTCOMES_AUDIT" \
  --expected-receipt-sha256 29b0c83a26b3f7fa30830e596cb070176b5c2738f6c4d2728b0ebde7bc87f36d \
  --split-manifest-sha256 7a572670867bbf6f811aad58c7ed1365f179083cda8187f5402221d554d1f1c0 \
  --fit-metadata-sha256 061069ed644b7fd700d7b65586622c02ef878c611ab6a549968f78bba8425f98 \
  --checkpoint-metadata-sha256 cdc879ef861608c6f26e004e4dd27826554f4c5631929e68f8dc57e8ea753047 \
  --source-manifest-sha256 2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4
"$PYTHON" scripts/verify_pros_gate_receipt.py capacity-adjudication \
  --receipt "$CAPACITY_ADJUDICATION" \
  --expected-receipt-sha256 17ac807e5b599c45e414958786fade47d7b2a0e1fd5603c3726d531eb143a352
nvidia-smi

"$PYTHON" scripts/train_direct_safety_fit.py \
  --fit-bundle "$FIT" \
  --checkpoint-bundle "$CHECKPOINT" \
  --outcomes-audit-receipt "$OUTCOMES_AUDIT" \
  --capacity-adjudication-receipt "$CAPACITY_ADJUDICATION" \
  --source-manifest "$SOURCE_MANIFEST" \
  --expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --expected-wrapper-sha256 "$WRAPPER_SHA256" \
  --output "$OUTPUT"
