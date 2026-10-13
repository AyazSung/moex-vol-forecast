"""Atomic file writing: readers never see a half-written file."""

# Запись файла не мгновенная. Если программа упадёт посреди записи, файл останется
# недописанным. Поэтому пишем во временный файл рядом и только в конце одним
# действием подменяем основной («атомарно» — целиком или никак).

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write df to a temporary file next to path, then swap it in one step."""
    # Создать папку вместе с родительскими; exist_ok — не ругаться, если уже есть
    # (аналог mkdir -p).
    path.parent.mkdir(parents=True, exist_ok=True)
    # Временный файл в той же папке: sber.csv -> sber.csv.tmp.
    # В той же — потому что подмена на одном диске атомарна.
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        # 1) Записать всё во временный файл (разделитель ";", без номеров строк).
        df.to_csv(tmp_path, sep=";", index=False)
        # 2) Одним действием заменить основной файл временным (аналог mv в bash).
        os.replace(tmp_path, path)
    finally:
        # 3) При любом исходе убрать временный файл, если он остался.
        #    После успешной замены его уже нет — missing_ok=True, это не ошибка.
        tmp_path.unlink(missing_ok=True)
    logger.info("saved %d rows to %s", len(df), path)
