#!/bin/bash
#SBATCH --job-name=pros-numdiag
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-numdiag-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-numdiag-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
DIRECT_RUN="$PROJECT/artifacts/training/gcls_v4_feature_100k_10133585/compact_axial_additive_d64_full_seed0"
SPLIT_MANIFEST="$PROJECT/artifacts/pros_gate/r079_numeric_rescue/split_manifest.json"
SPLIT_MANIFEST_SHA256=413264e4fa6473c3363b1f16a73f7e03eaa237dadbc4a3a8c07ad91a841d5d9c
SOURCE_MANIFEST="$PROJECT/refine-logs/direct-safety-gate/R079_CONTINUOUS_NUMERIC_DIAGNOSTIC_SOURCE_CLOSURE.json"
SOURCE_MANIFEST_SHA256=34d6f0c37caabf6039675b437ab708f2efe515fdb400bcbecc2fe1604ecf3fc3

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTHONPATH="$PROJECT/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$PROJECT"
echo "d7e3f0d763b35e997b50f532e2a57bc399df26c93c60a70af6b5e76a27f4083d  scripts/diagnose_direct_safety_numeric_portability.py" | sha256sum -c - >/dev/null
echo "841e214ba39640e305dfe15f340c2319e4bef485bd27219c559e8563a58b3ab2  tests/test_direct_safety_numeric_portability.py" | sha256sum -c - >/dev/null
echo "c811701d0ec097afa86e594946857290bcfff80e2cfc2e8638f0bcdaffcc0742  src/sph/data.py" | sha256sum -c - >/dev/null
echo "e3bd6392f7430e60e0eef16217dc904eeb018313ae8d4f543bd089a1943739b6  src/sph/direct_safety_gate.py" | sha256sum -c - >/dev/null
echo "f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06  src/sph/global_direct_selector.py" | sha256sum -c - >/dev/null
echo "332fac8948c61b2287cb861988b700c8a249a91cec6f808c0010d47fe7260cef  src/sph/first_miss_action_selector.py" | sha256sum -c - >/dev/null
echo "18c99279fb301b881db076d63627574b3c77205df8e91112373bd3579bee8a81  src/sph/source_closure.py" | sha256sum -c - >/dev/null
echo "$SOURCE_MANIFEST_SHA256  $SOURCE_MANIFEST" | sha256sum -c - >/dev/null
"$PYTHON" src/sph/source_closure.py \
  --project "$PROJECT" \
  --manifest "$SOURCE_MANIFEST" \
  --expected-manifest-sha256 "$SOURCE_MANIFEST_SHA256" >/dev/null
echo "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320  $DATA/metadata.json" | sha256sum -c - >/dev/null
echo "$SPLIT_MANIFEST_SHA256  $SPLIT_MANIFEST" | sha256sum -c - >/dev/null
echo "9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef  $DIRECT_RUN/metrics.json" | sha256sum -c - >/dev/null
echo "9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e  $DIRECT_RUN/best.pt" | sha256sum -c - >/dev/null
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA allocation unavailable"' >/dev/null

"$PYTHON" scripts/diagnose_direct_safety_numeric_portability.py \
  --data "$DATA" \
  --split-manifest "$SPLIT_MANIFEST" \
  --expected-split-manifest-sha256 "$SPLIT_MANIFEST_SHA256" \
  --direct-run "$DIRECT_RUN" \
  --target "$ASSETS/models/Qwen3-4B" \
  --batch-size 32
