#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xy68/kvcache_lab}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/xy68/vllm-exp/env/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-${ROOT_DIR}/scripts/prefix_cache_benchmark.py}"
START_VLLM_SCRIPT="${START_VLLM_SCRIPT:-${ROOT_DIR}/scripts/start_vllm.sh}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-testkey}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
REQUESTS="${REQUESTS:-50}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
WARMUP="${WARMUP:-2}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/vllm_shared_prefix_prompts_100.json}"
AUTO_START_VLLM="${AUTO_START_VLLM:-1}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-300}"
VLLM_HOME="${VLLM_HOME:-/scratch/xy68/vllm-exp}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-2}"
WAIT_STATUS_INTERVAL_SECONDS="${WAIT_STATUS_INTERVAL_SECONDS:-10}"

health_check() {
  /usr/bin/curl -fs \
    --connect-timeout 2 \
    --max-time 5 \
    "${BASE_URL}/models" \
    -H "Authorization: Bearer ${API_KEY}" \
    >/dev/null 2>&1
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] Benchmark script not found: ${SCRIPT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[ERROR] Prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

echo "[INFO] Checking vLLM health at ${BASE_URL}/models"
if ! health_check; then
  if [[ "${AUTO_START_VLLM}" == "1" ]]; then
    if [[ ! -x "${START_VLLM_SCRIPT}" ]]; then
      echo "[ERROR] vLLM not ready and start script is not executable: ${START_VLLM_SCRIPT}" >&2
      exit 1
    fi

    echo "[INFO] vLLM endpoint not ready, starting service via ${START_VLLM_SCRIPT}"
    VLLM_HOME="${VLLM_HOME}" "${START_VLLM_SCRIPT}"

    deadline=$((SECONDS + VLLM_WAIT_SECONDS))
    last_status_ts=0
    until health_check; do
      if (( SECONDS >= deadline )); then
        echo "[ERROR] vLLM endpoint did not become ready within ${VLLM_WAIT_SECONDS}s" >&2
        exit 1
      fi

      if (( last_status_ts == 0 || SECONDS - last_status_ts >= WAIT_STATUS_INTERVAL_SECONDS )); then
        remaining=$((deadline - SECONDS))
        echo "[INFO] Waiting for vLLM warmup... ${remaining}s left"
        last_status_ts=${SECONDS}
      fi

      sleep "${WAIT_POLL_SECONDS}"
    done
    echo "[INFO] vLLM is ready"
  else
    echo "[ERROR] vLLM endpoint is not ready. Start vLLM first or set AUTO_START_VLLM=1." >&2
    exit 1
  fi
fi

echo "[INFO] Running prefix cache benchmark..."
"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --model "${MODEL}" \
  --requests "${REQUESTS}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --warmup "${WARMUP}" \
  --prompt-file "${PROMPT_FILE}" \
  --output-dir "${ROOT_DIR}/data"

echo "[OK] Done. Results are in ${ROOT_DIR}/data"
