"""Command line entry point: fetch candles and save them to data/raw."""

# Точка входа: команда `fetch-candles` (см. [project.scripts] в pyproject.toml)
# вызывает main(). Здесь всё собирается вместе: настройки -> загрузка -> запись,
# плюс логи, обработка ошибок и коды возврата.

import logging
import signal
import sys

import psycopg
import requests

from moex_vol_forecast.config import load_settings
from moex_vol_forecast.db import save_candles
from moex_vol_forecast.logging_setup import setup_logging
from moex_vol_forecast.moex import fetch_candles
from moex_vol_forecast.storage import write_csv_atomic

logger = logging.getLogger(__name__)


def exit_on_sigterm(signum: int, frame: object) -> None:
    """Turn SIGTERM into a normal exit so that finally blocks still run."""
    # По умолчанию Python умирает от SIGTERM сразу, и блоки finally не выполняются.
    # Превращаем сигнал в обычный выход с кодом 128 + 15 = 143: finally в storage.py
    # успевает убрать временный файл. Так останавливают kill и docker stop.
    raise SystemExit(128 + signum)


def main() -> None:
    # Настройка логов — один раз, в точке входа (модули только пишут в лог).
    # Сама настройка — в logging_setup.py, её же использует import-candles.
    setup_logging()
    # При получении SIGTERM вызвать exit_on_sigterm.
    signal.signal(signal.SIGTERM, exit_on_sigterm)

    try:
        settings = load_settings()
        candles = fetch_candles(settings.ticker, settings.from_date)
        if candles.empty:
            raise ValueError(
                f"no candles for {settings.ticker} from {settings.from_date}"
            )
        # Путь к файлу: <папка данных>/raw/<тикер>.csv, например data/raw/sber.csv.
        # "/" у Path склеивает части пути.
        out_path = settings.data_dir / "raw" / f"{settings.ticker.lower()}.csv"
        write_csv_atomic(candles, out_path)
        # Задан адрес базы — дополнительно сохранить свечи в PostgreSQL.
        if settings.database_url:
            save_candles(candles, settings.ticker, settings.database_url)
    # Ожидаемые ошибки: плохие настройки и пустые данные (ValueError), предохранитель
    # (RuntimeError), диск и права (OSError), сеть и HTTP (RequestException),
    # база данных недоступна или отказала (psycopg.Error).
    # Для них — одна понятная строка ERROR и код возврата 1:
    # его видят cron, Docker и CI.
    # Всё остальное — баг: Python сам выведет трассировку и тоже выйдет с кодом 1.
    except (
        ValueError,
        RuntimeError,
        OSError,
        requests.RequestException,
        psycopg.Error,
    ) as err:
        logger.error("%s", err)
        sys.exit(1)

    logger.info(
        "%s: %d candles from %s", settings.ticker, len(candles), settings.from_date
    )
