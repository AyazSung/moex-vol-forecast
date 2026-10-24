"""Compare stored forecasts with what actually happened."""

# Команда `evaluate-forecasts` (этап 6): насколько хорошо модель прогнозирует в
# работе — не на истории, как при обучении, а на тех прогнозах, которые она
# действительно выдала.
#
# Как это устроено. Каждую ночь DAG считает прогноз волатильности на следующий
# торговый день и сохраняет его в таблицу forecasts. Например, по свече
# 21 октября модель (версия 1) предсказала волатильность SBER на 22 октября:
# 0.01103. В таблице forecasts у этого прогноза as_of = 2026-10-21 — день свечи,
# по которой он посчитан. Вечером 22 октября свеча закрылась, и стало известно,
# какой волатильность была на самом деле: 0.00630. Модель промахнулась на 0.00473.
#
# Много это или мало? Сравниваем с самым простым прогнозом — наивным: «завтра
# будет как сегодня», то есть волатильность 21 октября, 0.00965. Он промахнулся
# на 0.00335: по SBER в тот день наивный прогноз оказался точнее модели. По
# одному тикеру за один день выводов не делают, поэтому ошибки считаются по всем
# тикерам и по многим дням.
#
# Что делает команда, по шагам (функции ниже идут в том же порядке):
#   1. add_actuals — берёт прогнозы, для которых факт уже известен, и
#      приписывает к каждому факт и наивный прогноз.
#   2. quality_by_day — для каждого дня и каждой версии модели считает ошибку
#      модели и ошибку наивного прогноза по всем тикерам (RMSE) и их отношение
#      rmse_to_naive. Меньше 1 — модель лучше наивного прогноза. Это та же
#      метрика, что при обучении и выборе чемпиона, поэтому числа можно
#      сравнивать с теми, что записаны в MLflow.
#   3. add_20_day_ratio — то же отношение за 20 последних дней версии. За один
#      день оно скачет, за 20 дней — устойчивое; по нему и судят о качестве.
#   4. current_version и check_quality — какая версия сейчас в работе и хороша
#      ли она. Если за 20 дней она хуже порога --max-ratio (по умолчанию 1.0 —
#      хуже наивного прогноза), команда пишет ERROR и возвращает код 1: задача
#      Airflow падает, и проблема видна сразу.
# main связывает шаги и записывает результат в таблицу forecast_quality (db.py).
#
# Пример (в Docker, как остальные задачи проекта):
#   docker compose run --rm predictor evaluate-forecasts
#   docker compose run --rm predictor evaluate-forecasts --max-ratio 0.9

import argparse
import logging
import sys
from datetime import date

import numpy as np
import pandas as pd
import psycopg

from moex_vol_forecast.config import load_settings
from moex_vol_forecast.db import load_forecasts, save_quality
from moex_vol_forecast.features import TARGET, build_dataset
from moex_vol_forecast.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# Сколько последних дней версии учитывает проверка качества: 20 торговых дней —
# это около месяца.
WINDOW = 20


# ---------- шаг 1: факт и наивный прогноз к каждому прогнозу ----------
def add_actuals(forecasts: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """Forecasts whose outcome is known, with the actual and the naive value."""
    # dataset — та же таблица, на которой учится модель (build_dataset, этап 2).
    # Одна строка — один тикер в один день, и в ней уже есть всё нужное:
    #   pk_1   — волатильность этого дня; это и есть наивный прогноз на завтра;
    #   target — волатильность следующего торгового дня; это факт.
    # У SBER за 21 октября: pk_1 = 0.00965, target = 0.00630.
    # Последнего дня каждого тикера в dataset нет: следующий день ещё не
    # наступил, и факт неизвестен.
    #
    # Почему не отдельный SQL-запрос к таблице candles: факт и наивный прогноз
    # должны быть посчитаны ровно так же, как цель при обучении, — те же
    # выходные выброшены, та же формула Паркинсона. Вторая, отдельная формула
    # могла бы незаметно разойтись с первой, и проверка мерила бы не то, на чём
    # модель училась.
    #
    # Переименуем столбцы, чтобы дальше было видно, что есть что:
    #   date -> as_of (как в таблице forecasts), pk_1 -> naive_pk, target -> actual_pk.
    outcomes = dataset.rename(
        columns={"date": "as_of", "pk_1": "naive_pk", TARGET: "actual_pk"}
    )[["ticker", "as_of", "naive_pk", "actual_pk"]]
    # merge — объединить две таблицы по тикеру и дню, как JOIN в SQL. Остаются
    # только пары, которые нашлись в обеих таблицах. Поэтому прогноз на завтра,
    # у которого факта ещё нет, сюда не попадёт — его сверят в следующий раз.
    # Если за один день есть прогнозы двух версий, факт приписывается обоим.
    return forecasts.merge(outcomes, on=["ticker", "as_of"])


# ---------- шаг 2: ошибки за каждый день ----------
def quality_by_day(checked: pd.DataFrame) -> pd.DataFrame:
    """Errors of the model and of the naive forecast, one row per day and version."""
    # Квадрат промаха каждого прогноза — модели и наивного. В квадрат, как в
    # RMSE: знак промаха не важен, а большие промахи весят больше маленьких.
    checked = checked.assign(
        model_sq_error=(checked["forecast_pk"] - checked["actual_pk"]) ** 2,
        naive_sq_error=(checked["naive_pk"] - checked["actual_pk"]) ** 2,
    )
    # Все тикеры одного дня одной версии сворачиваем в одну строку:
    #   tickers   — сколько тикеров сверено;
    #   model_sse — сумма квадратов промахов модели за день (SSE — sum of
    #               squared errors, «сумма квадратов ошибок»);
    #   naive_sse — то же у наивного прогноза.
    # agg(имя=(столбец, функция)) — несколько итогов по группе за один раз.
    days = (
        checked.groupby(["model_version", "as_of"])
        .agg(
            tickers=("ticker", "size"),
            model_sse=("model_sq_error", "sum"),
            naive_sse=("naive_sq_error", "sum"),
        )
        .reset_index()
    )
    # RMSE — корень из среднего квадрата промаха: sqrt(сумма квадратов / число).
    # Пример — версия 1, 21 октября: 21 тикер, rmse = 0.00644, naive_rmse = 0.00875,
    # rmse_to_naive = 0.736: модель промахивалась примерно на четверть меньше
    # наивного прогноза.
    days["rmse"] = np.sqrt(days["model_sse"] / days["tickers"])
    days["naive_rmse"] = np.sqrt(days["naive_sse"] / days["tickers"])
    days["rmse_to_naive"] = days["rmse"] / days["naive_rmse"]
    return days


# ---------- шаг 3: та же метрика за 20 дней ----------
def add_20_day_ratio(days: pd.DataFrame) -> pd.DataFrame:
    """Add rmse_to_naive over the last 20 days of each model version."""
    # Метрика за 20 дней — это не среднее двадцати дневных значений. Все
    # прогнозы этих дней считаются вместе, как будто это один большой день:
    #   rmse_to_naive_20d = sqrt(сумма model_sse за 20 дней / сумма naive_sse)
    # Число прогнозов в формулу не входит: у модели и у наивного прогноза оно
    # одинаковое и сокращается. Так день с крупными промахами весит столько,
    # сколько в нём промахов, а не «один из двадцати».
    #
    # groupby("model_version") — у каждой версии своё окно: дни разных моделей не
    # смешиваются. rolling(WINDOW).sum() — сумма за текущий день и 19 предыдущих
    # дней этой версии. Пока у версии меньше 20 дней — пусто (NaN).
    by_version = days.groupby("model_version")
    model_sse_20d = by_version["model_sse"].transform(lambda s: s.rolling(WINDOW).sum())
    naive_sse_20d = by_version["naive_sse"].transform(lambda s: s.rolling(WINDOW).sum())
    return days.assign(rmse_to_naive_20d=np.sqrt(model_sse_20d / naive_sse_20d))


# ---------- шаг 4: решение по версии, которая сейчас в работе ----------
def current_version(forecasts: pd.DataFrame) -> int:
    """Model version that makes forecasts now: the version of the newest one."""
    # Самый свежий прогноз — с самым поздним днём свечи (as_of), а среди
    # прогнозов одного дня — записанный последним (created_at). Так же выбирает
    # прогноз API (/forecasts/{ticker}), поэтому его версия — та, что сейчас в
    # работе. Прогнозы, которые дописывает backfill-forecasts, — за прошлые дни,
    # поэтому на этот выбор не влияют, даже если записаны позже всех.
    newest = forecasts.sort_values(["as_of", "created_at"]).iloc[-1]
    return int(newest["model_version"])


def check_quality(
    days: pd.DataFrame, version: int, max_ratio: float
) -> tuple[bool, str]:
    """Is the version good enough over its last 20 days? Returns (ok, log line)."""
    mine = days[days["model_version"] == version]
    # Дни, где окно из 20 дней уже набралось (значение не пустое).
    full = mine.dropna(subset=["rmse_to_naive_20d"])
    # Дней меньше 20 — судить рано. Это не ошибка: проверка просто ждёт.
    if full.empty:
        return True, (
            f"version {version}: {len(mine)} of {WINDOW} days with known outcome, "
            "not judged yet"
        )
    # Судим по самому свежему окну — 20 дней до последнего сверенного дня.
    last = full.iloc[-1]
    ratio = last["rmse_to_naive_20d"]
    text = (
        f"version {version}: rmse_to_naive {ratio:.4f} "
        f"over {WINDOW} days up to {last['as_of'].date()}"
    )
    if ratio > max_ratio:
        return False, f"{text} is above the limit {max_ratio}"
    return True, f"{text}, limit {max_ratio}"


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Compare forecasts with outcomes.")
    # Порог проверки: выше него — ERROR и код 1. 1.0 — модель хуже наивного.
    parser.add_argument("--max-ratio", type=float, default=1.0)
    args = parser.parse_args()

    settings = load_settings()
    if not settings.database_url:
        logger.error("DATABASE_URL is not set")
        sys.exit(1)
    try:
        # Прогнозы — из базы, свечи — из data/raw, как при обучении.
        forecasts = load_forecasts(settings.database_url)
        dataset = build_dataset(settings.data_dir / "raw", today=date.today())
        # Шаги 1–3: факт к прогнозам, ошибки по дням, окно 20 дней.
        days = add_20_day_ratio(quality_by_day(add_actuals(forecasts, dataset)))
        save_quality(days, settings.database_url)
    # Ожидаемые ошибки: нет файлов со свечами или они не читаются (OSError),
    # плохие данные (ValueError), база недоступна (psycopg.Error).
    except (OSError, ValueError, psycopg.Error) as err:
        logger.error("cannot evaluate forecasts: %s", err)
        sys.exit(1)

    if days.empty:
        logger.info("no forecasts with known outcome yet")
        return
    # Сводка по каждой версии: метрика за все сверенные дни, первый и последний
    # из них и число прогнозов. Её можно сравнить с rmse_to_naive версии в MLflow.
    for version, group in days.groupby("model_version"):
        logger.info(
            "version %s: rmse_to_naive %.4f over %s .. %s, %d forecasts",
            version,
            np.sqrt(group["model_sse"].sum() / group["naive_sse"].sum()),
            group["as_of"].min().date(),
            group["as_of"].max().date(),
            group["tickers"].sum(),
        )

    # Шаг 4: решение по версии, которая сейчас в работе.
    ok, message = check_quality(days, current_version(forecasts), args.max_ratio)
    if not ok:
        logger.error("%s", message)
        sys.exit(1)
    logger.info("%s", message)
