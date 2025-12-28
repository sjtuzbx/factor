#!/usr/bin/env bash
set -euo pipefail

# Note: when updating financial fields, also update ann_date/end_date columns.
# Use --only-meta in fill_balance_fina_daily.py to refresh only the date columns.
DAILY_PATH="${1:-daily.hdf}"
BATCH_SIZE="${2:-200}"
CHECKPOINT="${3:-fill_balance_fina_daily.progress}"
LOG_FILE="${4:-fill_balance_fina_daily.log}"

if [[ ! -f "${DAILY_PATH}" ]]; then
  echo "daily.hdf not found: ${DAILY_PATH}" | tee -a "${LOG_FILE}"
  exit 1
fi

TOTAL=$(python - <<'PY'
import h5py as h5
with h5.File("daily.hdf", "r") as f:
    print(f["symbols"].shape[0])
PY
)

START=0
if [[ -f "${CHECKPOINT}" ]]; then
  START=$(cat "${CHECKPOINT}" | tr -d ' \n' || echo 0)
  if [[ -z "${START}" ]]; then
    START=0
  fi
fi

echo "total symbols: ${TOTAL}, start at: ${START}, batch: ${BATCH_SIZE}" | tee -a "${LOG_FILE}"

while [[ "${START}" -lt "${TOTAL}" ]]; do
  END=$((START + BATCH_SIZE))
  if [[ "${END}" -gt "${TOTAL}" ]]; then
    END="${TOTAL}"
  fi

  echo "batch ${START} -> ${END}" | tee -a "${LOG_FILE}"
  python fill_balance_fina_daily.py \
    --daily "${DAILY_PATH}" \
    --start-index "${START}" \
    --end-index "${END}" \
    --checkpoint "${CHECKPOINT}" \
    --flush-every 50 \
    >> "${LOG_FILE}" 2>&1

  START="${END}"
  echo "${START}" > "${CHECKPOINT}"
done

echo "done" | tee -a "${LOG_FILE}"
