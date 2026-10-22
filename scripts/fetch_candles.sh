#!/bin/bash
# fetch_candles.sh — загрузчик дневных свечей MOEX на bash (день 2 курса).
# Качает свечи по страницам и атомарно записывает их в data/raw/<тикер>.csv.
#
# ВНИМАНИЕ: учебный скрипт. Он ЗАМЕНЯЕТ файл свечами с FROM_DATE — короткий
# период сотрёт историю. Python-загрузчик с этапа 4 дописывает свежие свечи к
# истории; для работы используй его: docker compose run --rm loader
#
# Запуск:   ./scripts/fetch_candles.sh
# Настройки через переменные окружения:
#           TICKER=GAZP FROM_DATE=2020-01-01 ./scripts/fetch_candles.sh

# Строгий режим:
#   -e          — остановиться при любой ошибке;
#   -u          — ошибка при обращении к несуществующей переменной;
#   -o pipefail — пайп a | b считается упавшим, если упала любая его часть.
set -euo pipefail

# Настройки: значение из окружения, а если его нет — значение по умолчанию.
TICKER="${TICKER:-SBER}"
FROM_DATE="${FROM_DATE:-2026-10-01}"
# MOEX отдаёт не больше 500 строк за запрос, поэтому качаем по страницам.
PAGE_SIZE=500
# Предохранитель от бесконечного цикла, если API сломается.
MAX_PAGES=50
# Абсолютный путь к проекту: папка скрипта и на уровень выше.
# Нужен, чтобы скрипт работал из любой папки (например, из cron).
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/data/raw"
# Имя файла в нижнем регистре: SBER -> sber. tr работает на любом bash.
ticker_lower="$(echo "$TICKER" | tr '[:upper:]' '[:lower:]')"
OUT_FILE="${OUT_DIR}/${ticker_lower}.csv"
# Временный файл рядом с основным: пишем сюда, в конце одним mv подменяем основной.
TMP_FILE="${OUT_FILE}.tmp"
BASE_URL="https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/${TICKER}/candles.csv?interval=24&from=${FROM_DATE}"

mkdir -p "$OUT_DIR"
# При любом завершении скрипта (успех, ошибка, kill) удалить временный файл.
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

    # Одна страница. curl: -s тихо, -S но ошибки показывать, -f HTTP 4xx/5xx = ошибка;
    # --retry 3 — до трёх повторов; --max-time 30 — не дольше 30 секунд.
    page=$(curl -sSf --retry 3 --max-time 30 "${BASE_URL}&start=${start}")
    # Строки с данными: пропустить 3 служебные строки ответа и убрать пустые.
    # || true — если строк 0, grep возвращает код 1;
    # без этого set -e молча остановил бы скрипт.
    rows=$(echo "$page" | tail -n +4 | grep -v '^$' || true)
    if [ -z "$rows" ]; then
        break
    fi

    # На первой странице записать заголовок — третью строку ответа.
    if [ "$start" -eq 0 ]; then
        echo "$page" | sed -n '3p' > "$TMP_FILE"
    fi
    echo "$rows" >> "$TMP_FILE"

    # Сколько строк пришло; меньше 500 — это последняя страница.
    n=$(echo "$rows" | wc -l | tr -d ' ')
    total=$((total + n))
    if [ "$n" -lt "$PAGE_SIZE" ]; then
        break
    fi
    # Следующая страница: сдвинуть смещение на 500 и подождать полсекунды.
    start=$((start + PAGE_SIZE))
    sleep 0.5
done

# Ничего не скачано (например, нет такого тикера) — ошибка в stderr и код 1.
if [ "$total" -eq 0 ]; then
    echo "$(date '+%F %T') ERROR: no candles for ${TICKER} from ${FROM_DATE}" >&2
    exit 1
fi

# Атомарная подмена: основной файл заменяется временным одним действием.
mv "$TMP_FILE" "$OUT_FILE"
echo "$(date '+%F %T') ${TICKER}: ${total} candles from ${FROM_DATE}, requests=${page_num} -> ${OUT_FILE}"
