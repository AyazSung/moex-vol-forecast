"""Build model features from daily candles."""

# Превращаем дневные свечи в таблицу для обучения модели.
# Одна строка = один тикер в один торговый день:
#   признаки (features) — что известно к концу этого дня;
#   цель (target)       — волатильность СЛЕДУЮЩЕГО дня, её и будем прогнозировать.
#
# Мера дневной волатильности — оценка Паркинсона по максимуму и минимуму дня:
#   pk = sqrt( ln(high / low)^2 / (4 * ln 2) )
# Она точнее, чем модуль доходности, потому что использует весь диапазон цен за день.

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Признаки модели, все известны к концу дня t:
#   pk_1         — волатильность самого дня t;
#   pk_5, pk_22  — средняя волатильность за 5 и 22 торговых дня (неделя и месяц);
#   abs_ret_1    — модуль доходности дня t (насколько сильно сдвинулась цена закрытия);
#   volume_ratio — объём дня t относительно среднего объёма за 22 дня.
# pk_1, pk_5, pk_22 — классическая модель HAR (Corsi, 2009): дневная, недельная
# и месячная компоненты волатильности.
FEATURES = ["pk_1", "pk_5", "pk_22", "abs_ret_1", "volume_ratio"]
TARGET = "target"


def load_candles(raw_dir: Path) -> pd.DataFrame:
    """Read all <ticker>.csv files from raw_dir into one table."""
    frames = []
    # Все файлы data/raw/*.csv; имя файла — тикер (sber.csv -> SBER).
    for path in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(path, sep=";")
        df["ticker"] = path.stem.upper()
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no CSV files in {raw_dir}")
    candles = pd.concat(frames, ignore_index=True)
    # Дата свечи из столбца begin ("2026-10-01 00:00:00") — только день, без времени.
    candles["date"] = pd.to_datetime(candles["begin"]).dt.normalize()
    return candles[["ticker", "date", "open", "high", "low", "close", "volume"]]


def clean_candles(candles: pd.DataFrame, today: date) -> pd.DataFrame:
    """Drop rows that would mislead the model."""
    # 1) Торги выходного дня: объёмы в разы меньше будних, цены «тонкие».
    #    dayofweek: 0 — понедельник, ..., 5 — суббота, 6 — воскресенье.
    weekday = candles["date"].dt.dayofweek < 5
    # 2) Свеча сегодняшнего дня ещё не закрыта: high, low, close не окончательные.
    finished = candles["date"] < pd.Timestamp(today)
    # 3) Защита от ошибочных цен: логарифм от нуля и отрицательных чисел не считается.
    valid = (candles["low"] > 0) & (candles["high"] >= candles["low"])
    return candles[weekday & finished & valid]


def add_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Add feature columns and the next-day target, ticker by ticker."""
    # Сортируем по тикеру и дате: окна и сдвиги ниже считаются по порядку дней.
    df = candles.sort_values(["ticker", "date"]).reset_index(drop=True)
    # groupby("ticker") — считать всё отдельно для каждой акции, чтобы окна
    # и сдвиги не «перетекали» с одного тикера на другой.
    by_ticker = df.groupby("ticker")

    df["pk"] = np.sqrt(np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2)))
    # Логарифмическая доходность: ln(close_t / close_{t-1}); shift(1) — значение
    # предыдущего дня того же тикера.
    df["ret"] = np.log(df["close"] / by_ticker["close"].shift(1))

    df["pk_1"] = df["pk"]
    # rolling(n).mean() — скользящее среднее за последние n строк (включая текущую).
    # transform возвращает результат той же длины, что и таблица, — по каждому тикеру.
    df["pk_5"] = by_ticker["pk"].transform(lambda s: s.rolling(5).mean())
    df["pk_22"] = by_ticker["pk"].transform(lambda s: s.rolling(22).mean())
    df["abs_ret_1"] = df["ret"].abs()
    mean_volume = by_ticker["volume"].transform(lambda s: s.rolling(22).mean())
    df["volume_ratio"] = df["volume"] / mean_volume

    # Цель — волатильность следующего торгового дня: shift(-1) берёт значение
    # из СЛЕДУЮЩЕЙ строки того же тикера. Это единственное место, где мы
    # заглядываем в будущее, — и только для ответа, не для признаков.
    df[TARGET] = by_ticker["pk"].shift(-1)

    return df


def build_dataset(raw_dir: Path, today: date) -> pd.DataFrame:
    """Training rows: all features and the target are known."""
    df = add_features(clean_candles(load_candles(raw_dir), today))
    # Первые 22 дня каждого тикера (не хватает истории для окон) и последний день
    # (нет следующего дня для цели) дают пропуски — для обучения такие строки убираем.
    dataset = df.dropna(subset=[*FEATURES, TARGET]).reset_index(drop=True)
    logger.info(
        "dataset: %d rows, %d tickers, %s .. %s",
        len(dataset),
        dataset["ticker"].nunique(),
        dataset["date"].min().date(),
        dataset["date"].max().date(),
    )
    return dataset


def build_latest(raw_dir: Path, today: date) -> pd.DataFrame:
    """For prediction: the last finished day of every ticker (target unknown)."""
    df = add_features(clean_candles(load_candles(raw_dir), today))
    df = df.dropna(subset=FEATURES)
    # tail(1) в каждой группе — последняя строка тикера, то есть самый свежий день.
    return df.groupby("ticker").tail(1).reset_index(drop=True)
