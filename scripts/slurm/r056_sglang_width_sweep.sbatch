#!/bin/bash
#SBATCH --job-name=r056-width-sweep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r056-width-sweep-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r056-width-sweep-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
SGLANG=/hpc2hdd/home/zwang668/sglang-ddtree
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON=/hpc2hdd/home/zwang668/.venvs/ddtree-env/bin/python
VENV=/hpc2hdd/home/zwang668/.venvs/ddtree-env
SOURCE="$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_10164718"
PORT="${R056_PORT:-30461}"
MAX_PROMPTS="${R056_MAX_PROMPTS:-24}"
MAX_NEW_TOKENS="${R056_MAX_NEW_TOKENS:-64}"
PREFIX="$PROJECT/artifacts/analysis/r056_width_sweep_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$SGLANG/python:$PROJECT/scripts"
export CUDA_HOME=/hpc2ssd/softwares/cuda/cuda-12.8
export PATH="$CUDA_HOME/bin:$VENV/bin:/opt/compilers/gcc/131/bin:$PATH"
export LD_LIBRARY_PATH="/opt/compilers/gcc/131/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=/opt/compilers/gcc/131/bin/gcc
export CXX=/opt/compilers/gcc/131/bin/g++

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

run_mode() {
  local algorithm="$1"
  local label="$2"
  local width="$3"
  local server_log="$PROJECT/artifacts/logs/r056-width-${SLURM_JOB_ID}-${label}.server.log"
  local output="${PREFIX}_${label}.json"
  local args=(
    --model-path "$ASSETS/models/Qwen3-4B"
    --disable-overlap-schedule
    --disable-radix-cache
    --disable-piecewise-cuda-graph
    --random-seed 20260810
    --dtype bfloat16
    --tp-size 1
    --max-running-requests 1
    --cuda-graph-bs 1
    --cuda-graph-max-bs 1
    --page-size 1
    --port "$PORT"
    --mem-fraction-static 0.72
    --trust-remote-code
    --log-level warning
  )
  if [[ "$algorithm" != "TARGET" ]]; then
    args+=(
      --speculative-algorithm "$algorithm"
      --speculative-draft-model-path "$ASSETS/models/Qwen3-4B-Domino-b16"
    )
  fi
  if [[ "$algorithm" == "DOMINO_FOREST" ]]; then
    args+=(--speculative-domino-forest-width "$width")
  fi

  cd "$SGLANG"
  "$PYTHON" -m sglang.launch_server "${args[@]}" >"$server_log" 2>&1 &
  SERVER_PID=$!
  local ready=0
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      wait "$SERVER_PID"
      return 1
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "R056 $label server did not become healthy" >&2
    return 1
  fi

  cd "$PROJECT"
  "$PYTHON" scripts/benchmark_r056_sglang.py \
    --source-rollout "$SOURCE" \
    --base-url "http://127.0.0.1:${PORT}" \
    --mode "$label" \
    --output "$output" \
    --max-prompts "$MAX_PROMPTS" \
    --max-new-tokens "$MAX_NEW_TOKENS"

  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  sleep 2
}

run_mode TARGET target 0
run_mode DOMINO domino 1
for width in 4 5 6 7 8; do
  run_mode DOMINO_FOREST "forest_w${width}" "$width"
done

cd "$PROJECT"
"$PYTHON" scripts/analyze_r056_width_sweep.py \
  --target "${PREFIX}_target.json" \
  --domino "${PREFIX}_domino.json" \
  --forest \
    "${PREFIX}_forest_w4.json" \
    "${PREFIX}_forest_w5.json" \
    "${PREFIX}_forest_w6.json" \
    "${PREFIX}_forest_w7.json" \
    "${PREFIX}_forest_w8.json" \
  --output "${PREFIX}_analysis.json" \
  --bootstrap-draws 10000
