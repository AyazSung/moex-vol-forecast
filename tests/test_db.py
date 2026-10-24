# Тесты сохранения в PostgreSQL (db.py). Это интеграционные тесты: им нужна
# настоящая база. Без неё они пропускаются (skipped), а не падают.
# Запуск с базой из docker compose:
#   docker compose up -d db
#   TEST_DATABASE_URL=postgresql://moex:moex@localhost:5432/moex uv run pytest

import os

import numpy as np
import pandas as pd
import psycopg
import pytest

from moex_vol_forecast.db import (
    add_missing_forecasts,
    latest_forecast,
    load_forecasts,
    save_candles,
    save_drift,
    save_forecasts,
    save_quality,
)

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

# pytestmark — отметка для всех тестов файла: пропустить, если адрес базы не задан.
pytestmark = pytest.mark.skipif(
    DATABASE_URL is None, reason="TEST_DATABASE_URL is not set"
)

# Тикер, которого нет на бирже, — чтобы тест не задел настоящие данные.
TICKER = "TEST"


def make_candles(close):
    """Two fake daily candles with the given close price."""
    return pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "close": [close, close],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "value": [100.0, 100.0],
            "volume": [10, 10],
            "begin": ["2026-10-01 00:00:00", "2026-10-02 00:00:00"],
            "end": ["2026-10-01 23:59:59", "2026-10-02 23:59:59"],
        }
    )


def count_and_max_close():
    with psycopg.connect(DATABASE_URL) as conn:
        query = "SELECT count(*), max(close) FROM candles WHERE ticker = %s"
        return conn.execute(query, (TICKER,)).fetchone()


def test_upsert_is_idempotent_and_updates():
    try:
        # Загружаем одни и те же два дня дважды, второй раз — с другой ценой.
        save_candles(make_candles(1.5), TICKER, DATABASE_URL)
        save_candles(make_candles(2.5), TICKER, DATABASE_URL)

        # Строк по-прежнему 2 (дублей нет), а цена обновилась на новую.
        assert count_and_max_close() == (2, 2.5)
    finally:
        # Убираем за собой тестовые строки при любом исходе теста.
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute("DELETE FROM candles WHERE ticker = %s", (TICKER,))


def make_forecasts(value):
    """Forecasts for two days, as predict-vol builds them."""
    return pd.DataFrame(
        {
            "ticker": [TICKER, TICKER],
            "date": pd.to_datetime(["2026-10-01", "2026-10-02"]),
            "forecast_pk": [value, value],
        }
    )


def test_forecasts_upsert_and_latest():
    try:
        # Два прогноза версии 1, затем те же дни версией 2, затем версия 2 ещё раз.
        save_forecasts(make_forecasts(0.01), "1", DATABASE_URL)
        save_forecasts(make_forecasts(0.02), "2", DATABASE_URL)
        save_forecasts(make_forecasts(0.03), "2", DATABASE_URL)

        latest = latest_forecast(TICKER, DATABASE_URL)
        # Последний день свечи, последняя записанная версия, обновлённое значение.
        assert latest["as_of"].isoformat() == "2026-10-02"
        assert latest["model_version"] == 2
        assert latest["forecast_pk"] == 0.03
        # У тикера без прогнозов — None.
        assert latest_forecast("NO_SUCH_TICKER", DATABASE_URL) is None
    finally:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute("DELETE FROM forecasts WHERE ticker = %s", (TICKER,))


def test_backfill_keeps_stored_forecasts():
    # Дозаполнение (backfill-forecasts) дописывает недостающие прогнозы и не
    # трогает уже записанные.
    try:
        # Ежедневный DAG записал прогнозы версии 1 на два дня, по 0.01.
        save_forecasts(make_forecasts(0.01), "1", DATABASE_URL)
        # Дозаполнение приносит те же два дня, но с другим значением (0.05), и
        # ещё один день, которого в таблице нет.
        new_day = pd.DataFrame(
            {
                "ticker": [TICKER],
                "date": pd.to_datetime(["2026-10-05"]),
                "forecast_pk": [0.05],
            }
        )
        more = pd.concat([make_forecasts(0.05), new_day])

        added = add_missing_forecasts(more, "1", DATABASE_URL)

        # Добавился только новый день. Два прогноза, записанные раньше, остались
        # 0.01: то, что модель выдала вживую, дозаполнение не переписывает.
        assert added == 1
        stored = load_forecasts(DATABASE_URL)
        mine = stored[stored["ticker"] == TICKER].sort_values("as_of")
        assert mine["forecast_pk"].tolist() == [0.01, 0.01, 0.05]
    finally:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute("DELETE FROM forecasts WHERE ticker = %s", (TICKER,))


def test_quality_upsert():
    # Запись итогов проверки качества: повторная запись обновляет строки, а не
    # добавляет дубли, и пустое окно (NaN) хранится как NULL.
    # Версия 0 — такой в реестре нет (нумерация с 1): тест не заденет настоящие
    # строки таблицы.
    days = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2026-10-01", "2026-10-02"]),
            "model_version": [0, 0],
            "tickers": [21, 21],
            "rmse": [0.004, 0.005],
            "naive_rmse": [0.005, 0.005],
            "rmse_to_naive": [0.8, 1.0],
            "rmse_to_naive_20d": [np.nan, 0.9],
        }
    )
    query = (
        "SELECT tickers, rmse_to_naive_20d FROM forecast_quality "
        "WHERE model_version = 0 ORDER BY as_of"
    )
    try:
        save_quality(days, DATABASE_URL)
        # Второй раз, с другим числом тикеров: строки обновляются, дублей нет.
        save_quality(days.assign(tickers=20), DATABASE_URL)

        with psycopg.connect(DATABASE_URL) as conn:
            rows = conn.execute(query).fetchall()
        # Пустое значение pandas (NaN) легло в базу как NULL — в Python это None.
        assert rows == [(20, None), (20, 0.9)]
    finally:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute("DELETE FROM forecast_quality WHERE model_version = 0")


def test_drift_upsert():
    # Запись итогов проверки дрейфа: повторная запись обновляет строки, а не
    # добавляет дубли. Версия 0 — такой в реестре нет: настоящие строки не задеты.
    table = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2026-10-01", "2026-10-01"]),
            "feature": ["pk_1", "pk_22"],
            "psi": [0.5, 1.2],
            "psi_limit": [2.7, 7.0],
            "outside": [0, 0],
        }
    )
    query = (
        "SELECT feature, psi, outside FROM feature_drift "
        "WHERE model_version = 0 ORDER BY feature"
    )
    try:
        save_drift(table, "0", DATABASE_URL)
        # Второй раз — у pk_1 один тикер вышел за границы.
        save_drift(table.assign(outside=[1, 0]), "0", DATABASE_URL)

        with psycopg.connect(DATABASE_URL) as conn:
            rows = conn.execute(query).fetchall()
        assert rows == [("pk_1", 0.5, 1), ("pk_22", 1.2, 0)]
    finally:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute("DELETE FROM feature_drift WHERE model_version = 0")
