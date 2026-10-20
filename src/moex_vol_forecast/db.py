"""Save candles and forecasts to PostgreSQL."""

# Сохранение свечей и прогнозов в базу данных PostgreSQL.
# Таблица candles: одна строка = одна дневная свеча одного тикера.
# Запись идемпотентная: повторная загрузка тех же дней не создаёт дублей,
# а обновляет строки («upsert» = update + insert).

import logging

import pandas as pd
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

# Создать таблицу, если её ещё нет (IF NOT EXISTS), — безопасно выполнять каждый раз.
# PRIMARY KEY (ticker, trade_date) — в таблице не может быть двух строк
# с одним тикером и одной датой: на этом держится защита от дублей.
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS candles (
    ticker     text             NOT NULL,
    trade_date date             NOT NULL,
    open       double precision NOT NULL,
    high       double precision NOT NULL,
    low        double precision NOT NULL,
    close      double precision NOT NULL,
    volume     bigint           NOT NULL,
    value      double precision NOT NULL,
    PRIMARY KEY (ticker, trade_date)
)
"""

# Вставить строку, а если строка с такими ticker и trade_date уже есть (конфликт
# по первичному ключу) — обновить её. EXCLUDED — строка, которую пытались вставить.
# %s — места для значений: psycopg подставит их сам и безопасно. SQL никогда
# не склеивают из строк руками — это защита от SQL-инъекций.
UPSERT = """
INSERT INTO candles (ticker, trade_date, open, high, low, close, volume, value)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, trade_date) DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    value = EXCLUDED.value
"""


def save_candles(df: pd.DataFrame, ticker: str, database_url: str) -> int:
    """Upsert candles of one ticker into the candles table; return row count."""
    # Дата свечи — из столбца begin ("2026-10-01 00:00:00"): берём только дату.
    trade_dates = pd.to_datetime(df["begin"]).dt.date
    # Готовим строки для вставки. float(...) и int(...) переводят числа pandas
    # в обычные числа Python — с ними psycopg работает без сюрпризов.
    rows = [
        (ticker, day, float(o), float(h), float(lo), float(c), int(v), float(val))
        for day, o, h, lo, c, v, val in zip(
            trade_dates,
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            df["value"],
            strict=True,
        )
    ]
    # with psycopg.connect(...) — открыть соединение с базой. В конце блока все
    # изменения сохраняются одним целым (commit); если внутри ошибка — отменяются
    # целиком (rollback). Это транзакция: в базу не попадёт «половина» загрузки.
    with psycopg.connect(database_url) as conn:
        conn.execute(CREATE_TABLE)
        with conn.cursor() as cur:
            # executemany — выполнить один и тот же запрос для каждой строки.
            cur.executemany(UPSERT, rows)
    logger.info("upserted %d rows for %s into PostgreSQL", len(rows), ticker)
    return len(rows)


# ---------- прогнозы ----------
# Таблица forecasts: один прогноз = тикер + день свечи, по которой он посчитан
# (as_of) + версия модели из реестра. Версия в ключе: если перевесить чемпиона
# и запустить прогноз заново, прогнозы разных моделей лягут рядом, а не затрут
# друг друга — потом их можно сравнить с тем, что случилось на самом деле.
# created_at — когда строку записали; DEFAULT now() — база подставит время сама.
CREATE_FORECASTS = """
CREATE TABLE IF NOT EXISTS forecasts (
    ticker        text             NOT NULL,
    as_of         date             NOT NULL,
    model_version integer          NOT NULL,
    forecast_pk   double precision NOT NULL,
    created_at    timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, as_of, model_version)
)
"""

UPSERT_FORECAST = """
INSERT INTO forecasts (ticker, as_of, model_version, forecast_pk)
VALUES (%s, %s, %s, %s)
ON CONFLICT (ticker, as_of, model_version) DO UPDATE SET
    forecast_pk = EXCLUDED.forecast_pk,
    created_at = now()
"""

# Самый свежий прогноз тикера: сначала по дню свечи, при равенстве — по времени
# записи. LIMIT 1 — только одна строка.
LATEST_FORECAST = """
SELECT ticker, as_of, model_version, forecast_pk, created_at
FROM forecasts
WHERE ticker = %s
ORDER BY as_of DESC, created_at DESC
LIMIT 1
"""


def save_forecasts(df: pd.DataFrame, model_version: str, database_url: str) -> int:
    """Upsert forecasts (columns ticker, date, forecast_pk); return row count."""
    rows = [
        (ticker, day.date(), int(model_version), float(value))
        for ticker, day, value in zip(
            df["ticker"], df["date"], df["forecast_pk"], strict=True
        )
    ]
    with psycopg.connect(database_url) as conn:
        conn.execute(CREATE_FORECASTS)
        with conn.cursor() as cur:
            cur.executemany(UPSERT_FORECAST, rows)
    logger.info(
        "upserted %d forecasts of model version %s into PostgreSQL",
        len(rows),
        model_version,
    )
    return len(rows)


def latest_forecast(ticker: str, database_url: str) -> dict | None:
    """The newest stored forecast of a ticker as a dict, or None if there is none."""
    # row_factory=dict_row — строки приходят словарями {"ticker": ..., ...},
    # а не кортежами: так их удобно превращать в ответ API.
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        # Таблицы может ещё не быть, если пакетный прогноз ни разу не запускался.
        conn.execute(CREATE_FORECASTS)
        return conn.execute(LATEST_FORECAST, (ticker,)).fetchone()
