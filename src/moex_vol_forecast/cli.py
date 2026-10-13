"""Command line entry point: fetch candles and save them to data/raw."""

# Точка входа: команда `fetch-candles` (см. [project.scripts] в pyproject.toml)
# вызывает main(). Здесь всё собирается вместе: настройки -> загрузка -> запись,
# плюс логи, обработка ошибок и коды возврата.

import logging
import os
import signal
import sys

import requests

from moex_vol_forecast.config import load_settings
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
    # Настройка логов — один раз, здесь, в точке входа (модули только пишут в лог).
    #   level   — порог важности из LOG_LEVEL: DEBUG < INFO < WARNING < ERROR;
    #   format  — время, уровень, имя модуля и сообщение;
    #   datefmt — формат времени.
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
    # Ожидаемые ошибки: плохие настройки и пустые данные (ValueError), предохранитель
    # (RuntimeError), диск и права (OSError), сеть и HTTP (RequestException).
    # Для них — одна понятная строка ERROR и код возврата 1:
    # его видят cron, Docker и CI.
    # Всё остальное — баг: Python сам выведет трассировку и тоже выйдет с кодом 1.
    except (ValueError, RuntimeError, OSError, requests.RequestException) as err:
        logger.error("%s", err)
        sys.exit(1)

    logger.info(
        "%s: %d candles from %s", settings.ticker, len(candles), settings.from_date
    )
