#!/bin/bash
#SBATCH --job-name=r056-e2e-screen
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r056-e2e-screen-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r056-e2e-screen-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
SGLANG=/hpc2hdd/home/zwang668/sglang-ddtree
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON=/hpc2hdd/home/zwang668/.venvs/ddtree-env/bin/python
VENV=/hpc2hdd/home/zwang668/.venvs/ddtree-env
SOURCE="$PROJECT/artifacts/canonical/r047_anchor_t4_validation_select_10164718"
PORT="${R056_PORT:-30460}"
MAX_PROMPTS="${R056_MAX_PROMPTS:-12}"
MAX_NEW_TOKENS="${R056_MAX_NEW_TOKENS:-64}"
PREFIX="$PROJECT/artifacts/analysis/r056_e2e_screen_${SLURM_JOB_ID}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$SGLANG/python"
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
  local server_log="$PROJECT/artifacts/logs/r056-e2e-${SLURM_JOB_ID}-${label}.server.log"
  local output="${PREFIX}_${label}.json"
  local args=(
    --model-path "$ASSETS/models/Qwen3-4B"
    --disable-overlap-schedule
    --disable-radix-cache
    --dtype bfloat16
    --tp-size 1
    --max-running-requests 1
    --cuda-graph-bs 1
    --cuda-graph-max-bs 1
    --disable-piecewise-cuda-graph
    --random-seed 20260810
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

# One target-only lossless authority followed by an ABBA order for drift control.
run_mode TARGET target
run_mode DOMINO domino_a
run_mode DOMINO_FOREST forest_a
run_mode DOMINO_FOREST forest_b
run_mode DOMINO domino_b

cd "$PROJECT"
"$PYTHON" scripts/analyze_r056_sglang_e2e.py \
  --target "${PREFIX}_target.json" \
  --domino "${PREFIX}_domino_a.json" "${PREFIX}_domino_b.json" \
  --forest "${PREFIX}_forest_a.json" "${PREFIX}_forest_b.json" \
  --output "${PREFIX}_analysis.json" \
  --bootstrap-draws 10000 \
  --required-ratio 1.15
