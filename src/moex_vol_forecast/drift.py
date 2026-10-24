"""Check whether the model's input features drift away from its training data."""

# Команда `check-drift` (этап 6): похожи ли признаки, которые модель видит
# сейчас, на те, на которых она училась.
#
# Зачем. Проверка качества (evaluate-forecasts) узнаёт о проблеме, только когда
# придёт факт, и судит по 20 дням. А признаки известны сразу, в ту же ночь.
# Если на вход модели пошли данные, каких она при обучении не видела, её
# прогнозам стоит доверять меньше — и узнать об этом хочется сразу.
#
# Для каждого признака модели-чемпиона команда считает два показателя.
#
#   1. PSI (population stability index, «индекс стабильности распределения») —
#      насколько распределение признака за последние 20 торговых дней
#      отличается от распределения на данных обучения. 0 — не отличается,
#      чем больше — тем сильнее отличие. Как он считается — в функции psi ниже.
#
#   2. outside — у скольких тикеров значение признака в этот день вышло за
#      границы обучения: меньше минимума или больше максимума, которые модель
#      видела. Таких значений модель не встречала ни разу, и что она на них
#      ответит, никто не проверял.
#
# Порог PSI — не «учебный» 0.25. Этот порог придуман для кредитного скоринга,
# где каждая заявка — отдельный человек. У нас 20 дней подряд — одна и та же
# рыночная «погода»: спокойный месяц весь спокойный. Поэтому даже на данных
# обучения PSI месячного окна у признаков волатильности обычно больше 1 —
# с порогом 0.25 тревога звенела бы почти каждый день. Порог берём из истории
# самого признака: 99-й перцентиль PSI всех 20-дневных окон периода обучения.
# Выше него — месяц, какой модель при обучении почти не встречала.
#
# Проверка не прошла (PSI выше порога или есть значения за границами) —
# строки ERROR и код 1: задача Airflow становится красной. Прогноз при этом не
# останавливается: в необычный день он нужен больше всего, а красная задача —
# повод посмотреть на данные внимательнее.
#
# Результат по дням — в таблице feature_drift (db.py).
#
# Примеры (в Docker, как остальные задачи проекта):
#   docker compose run --rm predictor check-drift
#   docker compose run --rm predictor check-drift --model-version 2 --as-of 2024-07-16

import argparse
import logging
import sys
from datetime import date

import numpy as np
import pandas as pd
import psycopg
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from moex_vol_forecast.config import load_settings
from moex_vol_forecast.db import save_drift
from moex_vol_forecast.features import build_features
from moex_vol_forecast.logging_setup import setup_logging
from moex_vol_forecast.tracking import (
    CHAMPION,
    MODEL_NAME,
    first_test_day,
    load_version,
    setup_tracking,
)

logger = logging.getLogger(__name__)

# Окно — 20 торговых дней, около месяца, как в проверке качества (quality.py).
WINDOW = 20
# Число корзин, на которые делим значения признака.
BINS = 10
# Порог PSI — этот перцентиль PSI окон периода обучения.
LIMIT_QUANTILE = 0.99
# Доля для пустой корзины вместо нуля: ln(0) не считается.
EPS = 1e-4


# ---------- PSI: корзины, доли, формула ----------
def bin_edges(reference: pd.Series) -> np.ndarray:
    """Edges of 10 bins with an equal share of the reference values in each."""
    # Границы корзин — децили данных обучения: в каждую корзину попадает по 10%
    # значений обучения. Крайние корзины открыты: всё, что меньше первого дециля
    # или больше девятого, попадает в них.
    inner = np.quantile(reference, np.linspace(0.1, 0.9, BINS - 1))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def daily_counts(rows: pd.DataFrame, feature: str, edges: np.ndarray) -> pd.DataFrame:
    """How many values of the feature fall into each bin, one row per day."""
    # Номер корзины для каждого значения: searchsorted находит, между какими
    # границами оно лежит (0 — первая корзина, 9 — последняя).
    bins = np.searchsorted(edges, rows[feature], side="right") - 1
    # crosstab — таблица «день × корзина» с числом значений в каждой клетке.
    counts = pd.crosstab(rows["date"], bins)
    # Если в какую-то корзину не попало ни одного значения, её столбца нет —
    # добавляем нулевые, чтобы корзин всегда было 10.
    return counts.reindex(columns=range(BINS), fill_value=0).sort_index()


def psi(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """PSI between bin shares; the last axis is the bins."""
    # Для каждой корзины: (доля сейчас − доля в обучении) × ln(доля сейчас / доля
    # в обучении), и всё сложить. Если доли совпадают, каждое слагаемое — 0.
    # Если в корзине сейчас больше, чем в обучении, обе скобки положительны;
    # если меньше — обе отрицательны. Поэтому слагаемые всегда ≥ 0, и любое
    # расхождение только увеличивает сумму.
    # Пустая корзина дала бы ln(0) — нулевые доли заменяем крошечной EPS.
    current = np.clip(current, EPS, None)
    reference = np.clip(reference, EPS, None)
    return np.sum((current - reference) * np.log(current / reference), axis=-1)


def window_psi(counts: pd.DataFrame, reference_shares: np.ndarray) -> pd.Series:
    """PSI of every 20-day window against the reference, by the window's last day."""
    # Скользящая сумма счётчиков за 20 дней — распределение признака в окне:
    # сколько значений окна попало в каждую корзину. Первые 19 дней окно ещё не
    # набралось — их убираем (dropna).
    window = counts.rolling(WINDOW).sum().dropna()
    # Доли: счётчики корзин делим на число значений в окне (сумма по строке).
    values = window.to_numpy()
    shares = values / values.sum(axis=1, keepdims=True)
    return pd.Series(psi(shares, reference_shares), index=window.index)


# ---------- таблица по дням для одной версии модели ----------
def drift_table(
    features: pd.DataFrame, columns: list[str], test_start: pd.Timestamp
) -> pd.DataFrame:
    """PSI, its limit and out-of-range counts per day and feature since test_start."""
    # Данные обучения версии — всё до начала её тестового периода.
    train = features[features["date"] < test_start]
    parts = []
    for feature in columns:
        edges = bin_edges(train[feature])
        # Доли корзин на данных обучения: по построению около 10% в каждой.
        reference_shares = daily_counts(train, feature, edges).sum().to_numpy()
        reference_shares = reference_shares / reference_shares.sum()
        # PSI всех 20-дневных окон истории — и внутри обучения, и после.
        all_psi = window_psi(daily_counts(features, feature, edges), reference_shares)
        # Порог: каким бывал PSI месячных окон внутри периода обучения.
        limit = all_psi[all_psi.index < test_start].quantile(LIMIT_QUANTILE)
        # Сколько тикеров в каждый день вне границ обучения [минимум, максимум].
        low, high = train[feature].min(), train[feature].max()
        is_outside = (features[feature] < low) | (features[feature] > high)
        outside = is_outside.groupby(features["date"]).sum()
        # В таблицу — дни тестового периода: на них версия не училась.
        recent = all_psi[all_psi.index >= test_start]
        parts.append(
            pd.DataFrame(
                {
                    "as_of": recent.index,
                    "feature": feature,
                    "psi": recent.to_numpy(),
                    "psi_limit": limit,
                    "outside": outside.reindex(recent.index).to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def outside_values(
    features: pd.DataFrame, train: pd.DataFrame, columns: list[str], day: pd.Timestamp
) -> list[str]:
    """Readable list of the day's values outside the training range."""
    # Для лога: какие именно тикеры и значения вышли за границы, и какие это
    # границы, — чтобы сразу было видно, что случилось.
    lines = []
    today = features[features["date"] == day]
    for feature in columns:
        low, high = train[feature].min(), train[feature].max()
        for row in today.itertuples():
            value = getattr(row, feature)
            if value < low or value > high:
                lines.append(
                    f"{row.ticker} {feature} = {value:.4f}, "
                    f"training range {low:.4f} .. {high:.4f}"
                )
    return lines


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Check input drift of the champion.")
    # Номер версии модели; не указан — чемпион.
    parser.add_argument("--model-version", type=int)
    # День, который проверить; не указан — последний день с данными.
    parser.add_argument("--as-of", type=date.fromisoformat)
    args = parser.parse_args()

    settings = load_settings()
    if not settings.database_url:
        logger.error("DATABASE_URL is not set")
        sys.exit(1)
    setup_tracking()
    try:
        client = MlflowClient()
        if args.model_version is None:
            version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version
        else:
            version = str(args.model_version)
        # Начало тестового периода версии: до него — её данные обучения.
        test_start = pd.Timestamp(first_test_day(client, version))
        # Какие признаки видит модель: HAR — три, бустинг — пять (сигнатура MLflow).
        columns = load_version(version).columns
        features = build_features(settings.data_dir / "raw", today=date.today())
        table = drift_table(features, columns, test_start)
        save_drift(table, version, settings.database_url)
    # Ожидаемые ошибки: нет чемпиона или версии (MlflowException), нет параметра
    # test_start (KeyError), нет файлов со свечами (OSError), плохие данные
    # (ValueError), база недоступна (psycopg.Error).
    except (MlflowException, OSError, KeyError, ValueError, psycopg.Error) as err:
        logger.error("cannot check drift: %s", err)
        sys.exit(1)

    # Какой день проверяем: заданный или последний.
    day = pd.Timestamp(args.as_of) if args.as_of else table["as_of"].max()
    rows = table[table["as_of"] == day]
    if rows.empty:
        logger.error(
            "no drift data for %s: not a trading day or before %s",
            day.date(),
            test_start.date(),
        )
        sys.exit(1)

    logger.info(
        "model version %s, features %s, training data before %s",
        version,
        ", ".join(columns),
        test_start.date(),
    )
    # Сколько тикеров в проверяемый день — чтобы писать «0 of 21 tickers».
    tickers = int((features["date"] == day).sum())
    problems = []
    for row in rows.itertuples():
        logger.info(
            "%s: PSI %.2f over %d days up to %s (limit %.2f), "
            "%d of %d tickers outside the training range",
            row.feature,
            row.psi,
            WINDOW,
            day.date(),
            row.psi_limit,
            row.outside,
            tickers,
        )
        if row.psi > row.psi_limit:
            problems.append(
                f"{row.feature}: PSI {row.psi:.2f} is above the limit "
                f"{row.psi_limit:.2f}"
            )
    train = features[features["date"] < test_start]
    problems += outside_values(features, train, columns, day)

    if problems:
        for problem in problems:
            logger.error("%s", problem)
        sys.exit(1)
    logger.info("input features of version %s look like its training data", version)
