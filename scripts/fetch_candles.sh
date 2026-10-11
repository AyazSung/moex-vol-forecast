#!/bin/bash
set -euo pipefail

TICKER="${TICKER:-SBER}"
FROM_DATE="${FROM_DATE:-2026-10-01}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/data/raw"
ticker_lower="$(echo "$TICKER" | tr '[:upper:]' '[:lower:]')"
OUT_FILE="${OUT_DIR}/${ticker_lower}.csv"
URL="https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/${TICKER}/candles.csv?interval=24&from=${FROM_DATE}"

mkdir -p "$OUT_DIR"
curl -sSf --retry 3 --max-time 30 "$URL" -o "$OUT_FILE"

rows=$(tail -n +4 "$OUT_FILE" | grep -c -v '^$' || true)
if [ "$rows" -eq 0 ]; then
    echo "$(date '+%F %T') ERROR: no candles for ${TICKER} from ${FROM_DATE}" >&2
    exit 1
fi

echo "$(date '+%F %T') ${TICKER}: ${rows} candles from ${FROM_DATE} -> ${OUT_FILE}"