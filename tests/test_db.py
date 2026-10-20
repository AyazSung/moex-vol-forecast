# Тесты сохранения в PostgreSQL (db.py). Это интеграционные тесты: им нужна
# настоящая база. Без неё они пропускаются (skipped), а не падают.
# Запуск с базой из docker compose:
#   docker compose up -d db
#   TEST_DATABASE_URL=postgresql://moex:moex@localhost:5432/moex uv run pytest

import os

import pandas as pd
import psycopg
import pytest

from moex_vol_forecast.db import latest_forecast, save_candles, save_forecasts

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
