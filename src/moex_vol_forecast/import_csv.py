"""Import already downloaded CSV candles into PostgreSQL."""

# Команда `import-candles`: берёт все файлы <DATA_DIR>/raw/<тикер>.csv, уже скачанные
# загрузчиком, и сохраняет их в базу. Интернет не нужен — только файлы на диске.
# Использует ту же функцию save_candles, что и fetch-candles: модули переиспользуются.

import logging
import sys

import psycopg

from moex_vol_forecast.config import load_settings
from moex_vol_forecast.db import save_candles
from moex_vol_forecast.logging_setup import setup_logging
from moex_vol_forecast.storage import read_candles

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    settings = load_settings()
    # Без адреса базы импортировать некуда — понятная ошибка и код возврата 1.
    if not settings.database_url:
        logger.error("DATABASE_URL is not set")
        sys.exit(1)

    # glob("*.csv") — все файлы, имя которых заканчивается на .csv; sorted — по алфавиту.
    files = sorted((settings.data_dir / "raw").glob("*.csv"))
    if not files:
        logger.error("no CSV files in %s", settings.data_dir / "raw")
        sys.exit(1)

    total = 0
    try:
        for path in files:
            # path.stem — имя файла без расширения: sber.csv -> sber -> SBER.
            ticker = path.stem.upper()
            # read_candles — тем же способом, что загрузчик: числа ровно как в файле.
            df = read_candles(path)
            total += save_candles(df, ticker, settings.database_url)
    # Ожидаемые ошибки: файл не читается (OSError), в файле не те данные
    # (ValueError: например, обрезанный CSV) или база недоступна (psycopg.Error).
    except (OSError, ValueError, psycopg.Error) as err:
        logger.error("%s", err)
        sys.exit(1)

    logger.info("imported %d rows from %d files", total, len(files))
