"""Save candles to PostgreSQL."""

# Сохранение свечей в базу данных PostgreSQL.
# Таблица candles: одна строка = одна дневная свеча одного тикера.
# Запись идемпотентная: повторная загрузка тех же дней не создаёт дублей,
# а обновляет строки («upsert» = update + insert).

import logging

import pandas as pd
import psycopg

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
