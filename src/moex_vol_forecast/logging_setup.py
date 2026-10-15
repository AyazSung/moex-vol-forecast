"""Logging setup shared by all command line entry points."""

# Настройка логов вынесена в отдельный модуль: её вызывают обе команды —
# fetch-candles (cli.py) и import-candles (import_csv.py). Одна функция вместо двух
# одинаковых кусков кода: поменял формат здесь — поменялся везде.

import logging
import os


def setup_logging() -> None:
    """Configure log level and format once, at program start."""
    # level   — порог важности из LOG_LEVEL: DEBUG < INFO < WARNING < ERROR;
    # format  — время, уровень, имя модуля и сообщение;
    # datefmt — формат времени.
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
