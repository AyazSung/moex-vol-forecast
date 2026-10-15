"""Loader settings, read from environment variables."""

# Все настройки загрузчика берутся из переменных окружения. Код при этом не меняется,
# а значения задаёт тот, кто запускает: терминал, файл .env, Docker или cron.

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Папка проекта: от этого файла (src/moex_vol_forecast/config.py) три уровня вверх.
#   parents[0] = .../moex-vol-forecast/src/moex_vol_forecast
#   parents[1] = .../moex-vol-forecast/src
#   parents[2] = .../moex-vol-forecast   <- папка проекта
# Это верно только при локальном запуске, когда код лежит в src. В Docker-образе
# код лежит внутри .venv, поэтому там папка данных задаётся явно через DATA_DIR.
PROJECT_DIR = Path(__file__).resolve().parents[2]


# «Коробка» с настройками. @dataclass сам создаёт __init__ по полям ниже.
# frozen=True — после создания значения менять нельзя, чтобы их случайно не испортить.
@dataclass(frozen=True)
class Settings:
    ticker: str  # тикер акции, например SBER
    from_date: date  # с какой даты скачивать свечи
    data_dir: Path  # папка для данных; файлы ложатся в data_dir/raw/<тикер>.csv
    database_url: str | None  # адрес базы PostgreSQL; None — в базу не пишем


def load_settings() -> Settings:
    """Build settings from environment variables, falling back to defaults."""
    # os.environ.get("ИМЯ", умолчание) — значение переменной окружения, а если её нет —
    # значение по умолчанию. То же, что ${ИМЯ:-умолчание} в bash.
    ticker = os.environ.get("TICKER", "SBER").upper()
    # fromisoformat превращает строку "2026-10-01" в дату и сразу её проверяет:
    # неправильная дата даёт ValueError ещё до запросов к бирже («падать сразу»).
    from_date = date.fromisoformat(os.environ.get("FROM_DATE", "2026-10-01"))
    # DATA_DIR задана (например, в Docker: /app/data) — берём её;
    # не задана (локальный запуск) — папка data в проекте.
    # Из окружения приходит строка, а умолчание — уже Path; Path(...) приводит оба
    # случая к одному типу, чтобы дальше склеивать путь через "/".
    data_dir = Path(os.environ.get("DATA_DIR", PROJECT_DIR / "data"))
    # Адрес базы вида postgresql://пользователь:пароль@хост:порт/имя_базы.
    # Не задан — get вернёт None, и загрузчик пишет только CSV, как раньше.
    database_url = os.environ.get("DATABASE_URL")
    return Settings(
        ticker=ticker,
        from_date=from_date,
        data_dir=data_dir,
        database_url=database_url,
    )
