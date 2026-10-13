"""MOEX ISS client: daily candles, downloaded page by page."""

# Модуль общается с MOEX ISS — открытым API Московской биржи — и возвращает свечи
# таблицей pandas. Файлы здесь не пишутся: это делает storage.py.

import logging
import time
from datetime import date

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Логгер с именем модуля (moex_vol_forecast.moex): в логе видно, откуда строка.
logger = logging.getLogger(__name__)

# Адрес API дневных свечей; {ticker} подставляется через URL.format(ticker=...).
URL = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
# MOEX отдаёт не больше 500 строк за один запрос, поэтому качаем по страницам.
PAGE_SIZE = 500
# Предохранитель: не больше 50 страниц (25 000 свечей). Если API сломается и начнёт
# всегда отдавать полные страницы, цикл не станет бесконечным.
MAX_PAGES = 50
# Пауза между запросами, чтобы не засыпать чужой сервер и не получить блокировку.
PAUSE_SECONDS = 0.5
# Сколько секунд ждать ответа. Без таймаута запрос может висеть вечно.
TIMEOUT_SECONDS = 30


def make_session() -> requests.Session:
    """HTTP session that retries network errors and 429/5xx answers."""
    # Retry — правила повторов: до 3 попыток с растущими паузами 0.5, 1, 2 секунды.
    # Повторяем при сетевых сбоях и ответах 429 («слишком много запросов»)
    # и 5xx («ошибка сервера»): такие ошибки часто проходят сами.
    retry = Retry(
        total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)
    )
    # Session переиспользует соединение с сервером и хранит настройки повторов.
    session = requests.Session()
    # Применить правила повторов ко всем адресам, начинающимся с https://.
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_candles(
    ticker: str, from_date: date, session: requests.Session | None = None
) -> pd.DataFrame:
    """Download all daily candles for ticker since from_date."""
    # session можно передать снаружи. В работе он создаётся здесь (настоящий),
    # а в тестах передаётся поддельный — так пагинацию проверяют без интернета.
    session = session or make_session()
    url = URL.format(ticker=ticker)
    frames = []  # сюда складываем таблицы-страницы, в конце склеим в одну
    start = 0  # смещение: сколько строк пропустить от начала (0, 500, 1000, ...)
    page_num = 0
    while True:
        page_num += 1
        if page_num > MAX_PAGES:
            raise RuntimeError(f"more than {MAX_PAGES} pages for {ticker}, stopping")

        # Параметры запроса; requests сам соберёт из них ?interval=24&from=...&start=...
        #   interval=24  — дневные свечи;
        #   iss.meta=off — не присылать служебные описания столбцов.
        params = {
            "interval": 24,
            "from": from_date.isoformat(),
            "start": start,
            "iss.meta": "off",
        }
        response = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
        # Ответ 4xx/5xx превратить в исключение HTTPError (как curl -f в bash-версии).
        response.raise_for_status()
        # Ответ MOEX: {"candles": {"columns": [...], "data": [[...], [...], ...]}}
        block = response.json()["candles"]
        rows = block["data"]
        logger.info("page %d: %d rows (start=%d)", page_num, len(rows), start)

        if rows:
            frames.append(pd.DataFrame(rows, columns=block["columns"]))
        # Пришло меньше 500 строк — это последняя страница.
        if len(rows) < PAGE_SIZE:
            break
        start += PAGE_SIZE
        time.sleep(PAUSE_SECONDS)

    # Ничего не скачалось (например, нет такого тикера) — пустая таблица.
    # Что с этим делать, решает cli.py.
    if not frames:
        return pd.DataFrame()
    # Склеить страницы в одну таблицу; ignore_index — нумерация строк заново, 0..N-1.
    return pd.concat(frames, ignore_index=True)
