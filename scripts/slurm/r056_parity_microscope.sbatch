#!/bin/bash
#SBATCH --job-name=r056-parity-micro
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r056-parity-micro-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r056-parity-micro-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
SGLANG=/hpc2hdd/home/zwang668/sglang-ddtree
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON=/hpc2hdd/home/zwang668/.venvs/ddtree-env/bin/python
VENV=/hpc2hdd/home/zwang668/.venvs/ddtree-env
SOURCE="$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_10164718"
PORT="${R056_PORT:-30462}"
PREFIX="$PROJECT/artifacts/analysis/r056_parity_microscope_${SLURM_JOB_ID}"
TRACE_SITE="$PROJECT/scripts/r056_trace_site"
TRACE_RIDS="code_alpaca_train:04146bc88a2c42dde02a,code_alpaca_train:0784355798fd55a7bb2a,sharegpt:0jblW2u_0,sharegpt:1KitqZW_15"
SAMPLE_IDS=(
  code_alpaca_train:04146bc88a2c42dde02a
  code_alpaca_train:0784355798fd55a7bb2a
  sharegpt:0jblW2u_0
  sharegpt:1KitqZW_15
)

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
  local server_log="$PROJECT/artifacts/logs/r056-parity-${SLURM_JOB_ID}-${label}.server.log"
  local output="${PREFIX}_${label}.json"
  local trace="${PREFIX}_${label}.jsonl"
  local authority="${PREFIX}_target.json"
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
  if [[ "$algorithm" == "TARGET" ]]; then
    R056_TRACE_PATH="$trace" \
    R056_TRACE_MODE=target \
    R056_TRACE_RIDS="$TRACE_RIDS" \
    PYTHONPATH="$TRACE_SITE:$SGLANG/python:$PROJECT/scripts" \
      "$PYTHON" -m sglang.launch_server "${args[@]}" >"$server_log" 2>&1 &
  else
    R056_TRACE_PATH="$trace" \
    R056_TRACE_MODE="$label" \
    R056_TRACE_RIDS="$TRACE_RIDS" \
    R056_TRACE_AUTHORITY_JSON="$authority" \
    PYTHONPATH="$TRACE_SITE:$SGLANG/python:$PROJECT/scripts" \
      "$PYTHON" -m sglang.launch_server "${args[@]}" >"$server_log" 2>&1 &
  fi
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
    echo "R056 $label trace server did not become healthy" >&2
    return 1
  fi

  cd "$PROJECT"
  "$PYTHON" scripts/benchmark_r056_sglang.py \
    --source-rollout "$SOURCE" \
    --base-url "http://127.0.0.1:${PORT}" \
    --mode "$label" \
    --output "$output" \
    --sample-ids "${SAMPLE_IDS[@]}" \
    --max-new-tokens 64

  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  sleep 2
}

run_mode TARGET target 0
run_mode DOMINO domino 1
run_mode DOMINO_FOREST forest_w4 4
run_mode DOMINO_FOREST forest_w6 6
run_mode DOMINO_FOREST forest_w8 8

echo "R056 parity microscope artifacts: ${PREFIX}_*"
