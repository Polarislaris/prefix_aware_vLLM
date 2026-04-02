#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/xy68/kvcache_lab}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/xy68/vllm-exp/env/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-${ROOT_DIR}/scripts/prefix_similarity_window_benchmark.py}"
START_VLLM_SCRIPT="${START_VLLM_SCRIPT:-${ROOT_DIR}/scripts/start_vllm.sh}"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-testkey}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/vllm_shared_prefix_10types_100.json}"

SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
WARMUP="${WARMUP:-1}"
EXACT_PREFIX_CHARS="${EXACT_PREFIX_CHARS:-120}"
SIM_PREFIX_TOKENS="${SIM_PREFIX_TOKENS:-24}"
SIM_THRESHOLD_BITS="${SIM_THRESHOLD_BITS:-8}"
WINDOW_MS_LIST="${WINDOW_MS_LIST:-50,100}"
ROUTE2_GROUPING="${ROUTE2_GROUPING:-similarity}"
ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS:-5}"
COST_PER_TOKEN_MS="${COST_PER_TOKEN_MS:-0.03}"
CONCURRENCY="${CONCURRENCY:-5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

AUTO_START_VLLM="${AUTO_START_VLLM:-1}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-300}"
VLLM_HOME="${VLLM_HOME:-/scratch/xy68/vllm-exp}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-2}"
WAIT_STATUS_INTERVAL_SECONDS="${WAIT_STATUS_INTERVAL_SECONDS:-10}"

DRY_RUN="${DRY_RUN:-0}"

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

if [[ "${DRY_RUN}" != "1" ]]; then
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
fi

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --model "${MODEL}"
  --prompt-file "${PROMPT_FILE}"
  --sample-size "${SAMPLE_SIZE}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --warmup "${WARMUP}"
  --exact-prefix-chars "${EXACT_PREFIX_CHARS}"
  --sim-prefix-tokens "${SIM_PREFIX_TOKENS}"
  --sim-threshold-bits "${SIM_THRESHOLD_BITS}"
  --window-ms-list "${WINDOW_MS_LIST}"
  --route2-grouping "${ROUTE2_GROUPING}"
  --arrival-gap-ms "${ARRIVAL_GAP_MS}"
  --cost-per-token-ms "${COST_PER_TOKEN_MS}"
  --concurrency "${CONCURRENCY}"
  --progress-every "${PROGRESS_EVERY}"
  --output-dir "${ROOT_DIR}/data"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "[INFO] Running prefix similarity benchmark..."
"${CMD[@]}"

echo "[OK] Done. CSV results are in ${ROOT_DIR}/data"
