#!/bin/bash
#SBATCH --job-name=pros-split
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-split-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-split-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
OUTPUT="$PROJECT/artifacts/pros_gate/r079_numeric_v2/split_manifest.json"
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
echo "c811701d0ec097afa86e594946857290bcfff80e2cfc2e8638f0bcdaffcc0742  src/sph/data.py" | sha256sum -c -
echo "9fac5193a09c3def52b8e1360c0790d2497597f4daa8f5b67a16df5bbb1a31d4  src/sph/source_closure.py" | sha256sum -c -
echo "2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4  refine-logs/direct-safety-gate/R079_SOURCE_CLOSURE_NUMERIC_V2.json" | sha256sum -c -
"$PYTHON" src/sph/source_closure.py \
  --project "$PROJECT" \
  --expected-manifest-sha256 "$SOURCE_MANIFEST_SHA256"
echo "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320  $DATA/metadata.json" | sha256sum -c -
test "$(stat -c %s artifacts/manifests/open_perfectblend_100k_v2.jsonl)" -eq 92590866
test "$(stat -c %s artifacts/manifests/phase3_development_v3.jsonl)" -eq 1418202
test "$(stat -c %s artifacts/manifests/phase3_reserved_test_v3.jsonl)" -eq 429359
echo "b05087a56e8e717605415026421f7bae23092eb7cb9509361a36932f80260e3a  artifacts/manifests/open_perfectblend_100k_v2.jsonl" | sha256sum -c -
echo "e16374068e9c8904214fbf282b4adb6187a0b099db5c37e79660fc46a2801d01  artifacts/manifests/phase3_development_v3.jsonl" | sha256sum -c -
echo "ae25467fbb52b7091c8d9a5f98776b11ccf76e87e781850b5638734548a53bb4  artifacts/manifests/phase3_reserved_test_v3.jsonl" | sha256sum -c -

"$PYTHON" scripts/materialize_direct_safety_artifacts.py split \
  --data "$DATA" \
  --output "$OUTPUT" \
  --expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --exclusion "producer_train=$PROJECT/artifacts/manifests/open_perfectblend_100k_v2.jsonl" \
  --exclusion "validation=$PROJECT/artifacts/manifests/phase3_development_v3.jsonl" \
  --exclusion "reserved=$PROJECT/artifacts/manifests/phase3_reserved_test_v3.jsonl"

sha256sum "$OUTPUT"
