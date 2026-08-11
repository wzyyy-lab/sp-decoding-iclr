#!/bin/bash
#SBATCH --job-name=gcls-v2-mat25k
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-mat25k-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-mat25k-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
COLLECTION="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770"
OUTPUT="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_25k_seed20260730"

SOURCE_ARGS=()
LOG_ARGS=()
for INDEX in {0..7}; do
  PART=$(printf '%03d' "$INDEX")
  SOURCE_ARGS+=(--source "$COLLECTION/part-$PART")
  LOG_ARGS+=(--collection-log "$PROJECT/artifacts/logs/gcls-opb100k-10099770_${INDEX}.out")
done

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd "$PROJECT"
"$PYTHON" scripts/materialize_canonical_prompt_subset.py \
  "${SOURCE_ARGS[@]}" \
  "${LOG_ARGS[@]}" \
  --output "$OUTPUT" \
  --max-prompts 25000 \
  --seed 20260730 \
  --shard-blocks 256 \
  --expected-source-prompts 99356
