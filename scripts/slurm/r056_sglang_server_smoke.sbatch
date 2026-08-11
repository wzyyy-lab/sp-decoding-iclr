#!/bin/bash
#SBATCH --job-name=r056-server
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r056-server-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/r056-server-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
SGLANG=/hpc2hdd/home/zwang668/sglang-ddtree
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON=/hpc2hdd/home/zwang668/.venvs/ddtree-env/bin/python
VENV=/hpc2hdd/home/zwang668/.venvs/ddtree-env
MODE="${R056_MODE:-DOMINO}"
PORT="${R056_PORT:-30456}"
RESPONSE="$PROJECT/artifacts/analysis/r056_${MODE,,}_server_smoke_${SLURM_JOB_ID}.json"

if [[ "$MODE" != "DOMINO" && "$MODE" != "DOMINO_FOREST" ]]; then
  echo "R056_MODE must be DOMINO or DOMINO_FOREST" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$SGLANG/python"
export CUDA_HOME=/hpc2ssd/softwares/cuda/cuda-12.8
export PATH="$CUDA_HOME/bin:$VENV/bin:/opt/compilers/gcc/131/bin:$PATH"
export LD_LIBRARY_PATH="/opt/compilers/gcc/131/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=/opt/compilers/gcc/131/bin/gcc
export CXX=/opt/compilers/gcc/131/bin/g++
cd "$SGLANG"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$PYTHON" -m sglang.launch_server \
  --model-path "$ASSETS/models/Qwen3-4B" \
  --speculative-algorithm "$MODE" \
  --speculative-draft-model-path "$ASSETS/models/Qwen3-4B-Domino-b16" \
  --disable-overlap-schedule \
  --dtype bfloat16 \
  --tp-size 1 \
  --max-running-requests 1 \
  --cuda-graph-bs 1 \
  --cuda-graph-max-bs 1 \
  --page-size 1 \
  --port "$PORT" \
  --mem-fraction-static 0.72 \
  --trust-remote-code \
  --log-level info &
SERVER_PID=$!

READY=0
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID"
    exit 1
  fi
  sleep 1
done
if [[ "$READY" -ne 1 ]]; then
  echo "R056 server did not become healthy" >&2
  exit 1
fi

curl -fsS -X POST "http://127.0.0.1:${PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Give a concise explanation of why speculative decoding is lossless.","sampling_params":{"temperature":0.0,"max_new_tokens":64}}' \
  -o "$RESPONSE"

"$PYTHON" - "$RESPONSE" "$MODE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2]
payload = json.loads(path.read_text())
meta = payload.get("meta_info", {})
if not payload.get("text"):
    raise RuntimeError(f"{mode} returned empty text")
if int(meta.get("spec_verify_ct", 0)) <= 0:
    raise RuntimeError(f"{mode} did not execute speculative verification")
print(json.dumps({
    "mode": mode,
    "response_path": str(path),
    "completion_tokens": meta.get("completion_tokens"),
    "spec_verify_ct": meta.get("spec_verify_ct"),
    "spec_accept_length": meta.get("spec_accept_length"),
    "text": payload["text"],
}, indent=2), flush=True)
PY
