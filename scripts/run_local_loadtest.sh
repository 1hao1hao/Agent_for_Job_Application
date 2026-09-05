#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/reports/loadtest/local-deterministic}"
DURATION="${DURATION:-30s}"
HOST="${HOST:-http://127.0.0.1:8010}"
PORT="${PORT:-8010}"
CONCURRENCIES="${CONCURRENCIES:-1 5 10 20 50}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOCUST_PYTHON="${LOCUST_PYTHON:-${PYTHON_BIN}}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}"
export EVALRAG_PROJECT_ROOT="${ROOT_DIR}"
export EVALRAG_LOADTEST_MODE="${EVALRAG_LOADTEST_MODE:-deterministic}"

"${PYTHON_BIN}" -m uvicorn loadtest.app:create_loadtest_app \
  --factory --host 127.0.0.1 --port "${PORT}" --workers 1 \
  >"${OUTPUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${MONITOR_PID:-0}" "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if curl --silent --fail "${HOST}/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl --silent --fail "${HOST}/health" >"${OUTPUT_DIR}/health.json"

echo "timestamp,cpu_percent,rss_kb" >"${OUTPUT_DIR}/server_resources.csv"
(
  while kill -0 "${SERVER_PID}" 2>/dev/null; do
    ps -p "${SERVER_PID}" -o %cpu=,rss= | awk -v now="$(date --iso-8601=seconds)" \
      '{print now "," $1 "," $2}' >>"${OUTPUT_DIR}/server_resources.csv"
    sleep 1
  done
) &
MONITOR_PID=$!

for users in ${CONCURRENCIES}; do
  "${LOCUST_PYTHON}" -m locust -f loadtest/locustfile.py \
    --headless --only-summary --host "${HOST}" \
    --users "${users}" --spawn-rate "${users}" --run-time "${DURATION}" \
    --csv "${OUTPUT_DIR}/c${users}" --html "${OUTPUT_DIR}/c${users}.html" \
    >"${OUTPUT_DIR}/c${users}.log" 2>&1
done

echo "load-test artifacts: ${OUTPUT_DIR}"
