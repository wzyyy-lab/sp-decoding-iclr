#!/bin/bash
#SBATCH --job-name=pros-ulp-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-ulp-smoke-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-ulp-smoke-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
SOURCE_MANIFEST_SHA256=2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"
export PROS_REQUIRE_CUDA=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$PROJECT"
echo "852915273e330344c1afe472f9d0c2b4789b564e6636af897051486fbaa3a6d9  src/sph/direct_safety_artifacts.py" | sha256sum -c -
echo "3b2f0fbab3a0a683a7f5e142b88f79c96960b6427e48253861c090e85768437c  src/sph/direct_safety_numeric_policy.py" | sha256sum -c -
echo "102a87a5ea9dbfedaf7fc6429fac79023bfd7e7ebbee062e332f4ad4ad5a863d  scripts/audit_direct_safety_artifacts.py" | sha256sum -c -
echo "9fac5193a09c3def52b8e1360c0790d2497597f4daa8f5b67a16df5bbb1a31d4  src/sph/source_closure.py" | sha256sum -c -
echo "b552f16dd8f9afab8df570609391f443571ca199ed6e13b2a00844b8f2c4a4c5  tests/test_direct_safety_numeric_policy_v2.py" | sha256sum -c -
echo "2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4  refine-logs/direct-safety-gate/R079_SOURCE_CLOSURE_NUMERIC_V2.json" | sha256sum -c -
"$PYTHON" src/sph/source_closure.py \
  --project "$PROJECT" \
  --expected-manifest-sha256 "$SOURCE_MANIFEST_SHA256"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA allocation unavailable"'
"$PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_direct_safety_numeric_policy_v2.py::test_cuda_same_device_and_portable_independent_roundtrip
