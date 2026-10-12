#!/bin/bash
set -euo pipefail

TICKER="${TICKER:-SBER}"
FROM_DATE="${FROM_DATE:-2026-10-01}"
PAGE_SIZE=500
MAX_PAGES=50
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/data/raw"
ticker_lower="$(echo "$TICKER" | tr '[:upper:]' '[:lower:]')"
OUT_FILE="${OUT_DIR}/${ticker_lower}.csv"
TMP_FILE="${OUT_FILE}.tmp"
BASE_URL="https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/${TICKER}/candles.csv?interval=24&from=${FROM_DATE}"

mkdir -p "$OUT_DIR"
trap 'rm -f "$TMP_FILE"' EXIT

start=0
page_num=0
total=0
while true; do
    page_num=$((page_num + 1))
    if [ "$page_num" -gt "$MAX_PAGES" ]; then
        echo "$(date '+%F %T') ERROR: more than ${MAX_PAGES} pages for ${TICKER}, stopping" >&2
        exit 1
    fi

    page=$(curl -sSf --retry 3 --max-time 30 "${BASE_URL}&start=${start}")
    rows=$(echo "$page" | tail -n +4 | grep -v '^$' || true)
    if [ -z "$rows" ]; then
        break
    fi

    if [ "$start" -eq 0 ]; then
        echo "$page" | sed -n '3p' > "$TMP_FILE"
    fi
    echo "$rows" >> "$TMP_FILE"

    n=$(echo "$rows" | wc -l | tr -d ' ')
    total=$((total + n))
    if [ "$n" -lt "$PAGE_SIZE" ]; then
        break
    fi
    start=$((start + PAGE_SIZE))
    sleep 0.5
done

if [ "$total" -eq 0 ]; then
    echo "$(date '+%F %T') ERROR: no candles for ${TICKER} from ${FROM_DATE}" >&2
    exit 1
fi

mv "$TMP_FILE" "$OUT_FILE"
echo "$(date '+%F %T') ${TICKER}: ${total} candles from ${FROM_DATE}, requests=${page_num} -> ${OUT_FILE}"