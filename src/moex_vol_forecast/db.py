"""Save candles, forecasts and forecast quality to PostgreSQL."""

# Сохранение свечей, прогнозов и качества прогнозов в базу данных PostgreSQL.
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

# Дописать прогноз, только если такого ещё нет (этап 6, backfill-forecasts).
# Отличие от UPSERT_FORECAST выше — последняя строка. Если тикер, день и версия
# уже есть в таблице (конфликт по первичному ключу), DO NOTHING — ничего не
# делать; UPSERT_FORECAST в этом случае перезаписал бы прогноз. Прогнозы,
# которые записал ежедневный DAG, — то, что модель действительно выдала; при
# дозаполнении их трогать нельзя.
INSERT_FORECAST_IF_NEW = """
INSERT INTO forecasts (ticker, as_of, model_version, forecast_pk)
VALUES (%s, %s, %s, %s)
ON CONFLICT (ticker, as_of, model_version) DO NOTHING
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

# Все прогнозы целиком — для проверки качества (этап 6, quality.py): каждый
# прогноз сверяется с тем, что случилось на самом деле.
ALL_FORECASTS = """
SELECT ticker, as_of, model_version, forecast_pk, created_at
FROM forecasts
"""


def forecast_rows(df: pd.DataFrame, model_version: str) -> list[tuple]:
    """Rows for the forecasts table from columns ticker, date, forecast_pk."""
    # Таблица pandas -> список кортежей (тикер, день, версия, прогноз) для
    # executemany. Нужно и save_forecasts, и add_missing_forecasts, поэтому
    # вынесено в отдельную функцию, а не повторено дважды.
    return [
        (ticker, day.date(), int(model_version), float(value))
        for ticker, day, value in zip(
            df["ticker"], df["date"], df["forecast_pk"], strict=True
        )
    ]


def save_forecasts(df: pd.DataFrame, model_version: str, database_url: str) -> int:
    """Upsert forecasts (columns ticker, date, forecast_pk); return row count."""
    rows = forecast_rows(df, model_version)
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


def add_missing_forecasts(
    df: pd.DataFrame, model_version: str, database_url: str
) -> int:
    """Insert only forecasts that are not stored yet; return how many were added."""
    # Как save_forecasts, но с INSERT_FORECAST_IF_NEW: уже записанные прогнозы
    # остаются как были. Нужно для backfill-forecasts (этап 6).
    rows = forecast_rows(df, model_version)
    with psycopg.connect(database_url) as conn:
        conn.execute(CREATE_FORECASTS)
        with conn.cursor() as cur:
            cur.executemany(INSERT_FORECAST_IF_NEW, rows)
            # rowcount — сколько строк запрос действительно вставил. Прогнозы,
            # которые уже были в таблице, пропущены и не считаются: при повторном
            # запуске здесь будет 0.
            added = cur.rowcount
    logger.info(
        "added %d of %d forecasts of model version %s into PostgreSQL",
        added,
        len(rows),
        model_version,
    )
    return added


def load_forecasts(database_url: str) -> pd.DataFrame:
    """All stored forecasts; as_of as a pandas date, to match feature tables."""
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        conn.execute(CREATE_FORECASTS)
        rows = conn.execute(ALL_FORECASTS).fetchall()
    # Строки пришли словарями (dict_row) — из них сразу получается таблица pandas.
    # columns=... — чтобы и у пустой таблицы были нужные столбцы.
    forecasts = pd.DataFrame(
        rows, columns=["ticker", "as_of", "model_version", "forecast_pk", "created_at"]
    )
    # День свечи из базы приходит датой Python, а в таблицах признаков дни —
    # даты pandas (Timestamp). Без приведения к одному типу merge откажется
    # объединять таблицы: ValueError «You are trying to merge on object and
    # datetime64».
    forecasts["as_of"] = pd.to_datetime(forecasts["as_of"])
    return forecasts


def latest_forecast(ticker: str, database_url: str) -> dict | None:
    """The newest stored forecast of a ticker as a dict, or None if there is none."""
    # row_factory=dict_row — строки приходят словарями {"ticker": ..., ...},
    # а не кортежами: так их удобно превращать в ответ API.
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        # Таблицы может ещё не быть, если пакетный прогноз ни разу не запускался.
        conn.execute(CREATE_FORECASTS)
        return conn.execute(LATEST_FORECAST, (ticker,)).fetchone()


# ---------- качество прогнозов (этап 6) ----------
# Таблица forecast_quality — итог проверки качества (quality.py). Одна строка —
# прогнозы одной версии модели за один день свечи (as_of), сверенные с тем, что
# случилось на следующий торговый день:
#   tickers           — сколько тикеров сверено (обычно все 21);
#   rmse              — ошибка модели за день: корень из среднего квадрата промаха;
#   naive_rmse        — то же у наивного прогноза «завтра как сегодня»;
#   rmse_to_naive     — rmse / naive_rmse: меньше 1 — в этот день модель была
#                       точнее наивного прогноза;
#   rmse_to_naive_20d — то же отношение за 20 последних дней этой версии. Пусто
#                       (NULL), пока у версии меньше 20 дней. По нему судят о
#                       качестве: за один день отношение сильно скачет.
# Пример строки: день 2026-10-21, версия 1, 21 тикер, rmse 0.00644,
# naive_rmse 0.00875, rmse_to_naive 0.736.
# Первичный ключ — день и версия: у версии за день ровно одна строка, и
# повторный пересчёт её обновляет (upsert), а не добавляет дубль.
CREATE_QUALITY = """
CREATE TABLE IF NOT EXISTS forecast_quality (
    as_of             date             NOT NULL,
    model_version     integer          NOT NULL,
    tickers           integer          NOT NULL,
    rmse              double precision NOT NULL,
    naive_rmse        double precision NOT NULL,
    rmse_to_naive     double precision NOT NULL,
    rmse_to_naive_20d double precision,
    PRIMARY KEY (as_of, model_version)
)
"""

UPSERT_QUALITY = """
INSERT INTO forecast_quality (
    as_of, model_version, tickers, rmse, naive_rmse, rmse_to_naive, rmse_to_naive_20d
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (as_of, model_version) DO UPDATE SET
    tickers = EXCLUDED.tickers,
    rmse = EXCLUDED.rmse,
    naive_rmse = EXCLUDED.naive_rmse,
    rmse_to_naive = EXCLUDED.rmse_to_naive,
    rmse_to_naive_20d = EXCLUDED.rmse_to_naive_20d
"""


def save_quality(days: pd.DataFrame, database_url: str) -> int:
    """Upsert daily forecast quality (one row per day and model version)."""
    # Готовим строки для вставки, как в save_candles: числа pandas переводим в
    # обычные числа Python. Пустое значение pandas (NaN) — в None: в базе это
    # NULL, «пусто». Так хранится rmse_to_naive_20d, пока 20 дней не набралось.
    rows = [
        (
            day.date(),
            int(version),
            int(tickers),
            float(rmse),
            float(naive_rmse),
            float(ratio),
            None if pd.isna(ratio_20d) else float(ratio_20d),
        )
        for day, version, tickers, rmse, naive_rmse, ratio, ratio_20d in zip(
            days["as_of"],
            days["model_version"],
            days["tickers"],
            days["rmse"],
            days["naive_rmse"],
            days["rmse_to_naive"],
            days["rmse_to_naive_20d"],
            strict=True,
        )
    ]
    with psycopg.connect(database_url) as conn:
        conn.execute(CREATE_QUALITY)
        with conn.cursor() as cur:
            cur.executemany(UPSERT_QUALITY, rows)
    logger.info("upserted %d rows into forecast_quality", len(rows))
    return len(rows)
