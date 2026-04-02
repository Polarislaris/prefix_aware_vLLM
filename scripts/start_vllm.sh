#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-testkey}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-3}"

ROOT_DIR="${ROOT_DIR:-/home/xy68/kvcache_lab}"
VLLM_HOME="${VLLM_HOME:-/scratch/xy68/vllm-exp}"
VLLM_ENV="${VLLM_ENV:-${VLLM_HOME}/env}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ENV}/bin/vllm}"
HF_HOME="${HF_HOME:-${VLLM_HOME}/hf-home}"
VLLM_LIB_DIR="${VLLM_LIB_DIR:-${VLLM_ENV}/lib}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"

mkdir -p "${LOG_DIR}" "${HF_HOME}"

if [[ ! -d "${VLLM_HOME}" ]]; then
  echo "[ERROR] VLLM_HOME not found: ${VLLM_HOME}" >&2
  exit 1
fi

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "[ERROR] vLLM binary not found or not executable: ${VLLM_BIN}" >&2
  exit 1
fi

export HF_HOME
export VLLM_NO_USAGE_STATS=1

# Force vLLM to prefer libraries from mounted env, avoiding host CXXABI mismatch.
export PATH="${VLLM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${VLLM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${VLLM_LIB_DIR}/libstdc++.so.6"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/vllm_${TS}.log"
PID_FILE="${LOG_DIR}/vllm.pid"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found. Please run this on a GPU compute node (inside srun)." >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && ps -p "${old_pid}" >/dev/null 2>&1; then
    echo "[INFO] Found existing vLLM process PID=${old_pid}, stopping it..."
    kill "${old_pid}" || true
    sleep 2
  fi
fi

if pgrep -af "${VLLM_BIN} serve" >/dev/null 2>&1; then
  echo "[INFO] Cleaning leftover vLLM serve process..."
  pkill -f "${VLLM_BIN} serve" || true
  sleep 2
fi

echo "[INFO] Starting vLLM..."
echo "[INFO] VLLM_HOME=${VLLM_HOME}"
echo "[INFO] Model=${MODEL} Host=${HOST} Port=${PORT}"
echo "[INFO] Log=${LOG_FILE}"

cd "${VLLM_HOME}"

nohup "${VLLM_BIN}" serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --dtype "${DTYPE}" \
  --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  > "${LOG_FILE}" 2>&1 &

new_pid=$!
echo "${new_pid}" > "${PID_FILE}"

sleep "${STARTUP_WAIT_SECONDS}"
if ps -p "${new_pid}" >/dev/null 2>&1; then
  echo "[OK] vLLM started. PID=${new_pid}"
  echo "[INFO] Follow logs: tail -f ${LOG_FILE}"
  echo "[INFO] Health check: env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/curl http://127.0.0.1:${PORT}/v1/models -H \"Authorization: Bearer ${API_KEY}\""
else
  echo "[ERROR] vLLM failed to start. See log: ${LOG_FILE}" >&2
  exit 1
fi
