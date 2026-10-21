"""Atomic file writing and merging of candle history."""

# Запись файла не мгновенная. Если программа упадёт посреди записи, файл останется
# недописанным. Поэтому пишем во временный файл рядом и только в конце одним
# действием подменяем основной («атомарно» — целиком или никак).
#
# read_candles и merge_candles — дозагрузка без потери истории: свежие свечи
# добавляются к уже скачанным, а не заменяют их (этап 4: ежедневная загрузка
# за последнюю неделю).

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


def read_candles(path: Path) -> pd.DataFrame:
    """Read a candles CSV written by write_csv_atomic, numbers exactly as written."""
    # float_precision="round_trip" — читать числа ровно такими, какими их записали.
    # Обычный разбор pandas быстрее, но может ошибиться в 17-й значащей цифре:
    # оборот 11541818590.300005 прочитался бы как 11541818590.300003, и при
    # ежедневной перезаписи файла старые значения понемногу «плыли» бы.
    return pd.read_csv(path, sep=";", float_precision="round_trip")


def merge_candles(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add new candles to the old ones; a day present in both is taken from new."""
    # Из старых свечей убираем дни, которые пришли заново: их значения могли
    # уточниться (например, вчера свеча была ещё незакрытой). isin — «входит ли
    # значение в список», ~ — логическое «не»: оставить строки, чьего begin НЕТ
    # среди новых.
    kept = old[~old["begin"].isin(new["begin"])]
    # Склеить старое и новое и упорядочить по началу свечи. begin — строка вида
    # "2026-10-21 00:00:00": по алфавиту такие строки идут в порядке дат.
    merged = pd.concat([kept, new], ignore_index=True)
    return merged.sort_values("begin").reset_index(drop=True)
