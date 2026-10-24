"""Fill in past forecasts of a model version over its test period."""

# Команда `backfill-forecasts` (этап 6): дописать прогнозы модели за прошлые дни.
#
# Зачем. Проверка качества (evaluate-forecasts, quality.py) судит о модели по её
# прогнозам за 20 последних дней. Новый чемпион начинает с нуля: в первый день
# у него один прогноз, и ещё месяц о его качестве ничего не известно. Месяц без
# проверки — слишком долго.
#
# Решение. У каждой версии модели есть тестовый период — дни, на которых она не
# училась. Начало этого периода train-model записывает в MLflow, в параметр
# test_start запуска. Например, версия 6 училась на свечах до 23 июля 2026 года
# (test_start = 2026-07-23), а все дни с 23 июля она «не видела». Команда
# считает прогнозы версии на каждый из этих дней — так, как будто версия уже
# тогда работала в ежедневном DAG, — и дописывает их в таблицу forecasts.
# После этого проверке качества есть по чему судить.
#
# Почему такие прогнозы честные. Признаки дня as_of считаются только по свечам
# до этого дня включительно (features.py, этап 2), а ответа модель не знает:
# этих дней не было в обучении. Это ровно то, что она выдала бы вживую.
# Раньше test_start дописывать нельзя: на тех днях модель училась и «помнит
# ответы», её ошибка там была бы обманчиво маленькой.
#
# Прогнозы, которые уже есть в таблице (например, записанные ежедневным DAG),
# команда не трогает — дописывает только недостающие. Поэтому запускать её
# можно сколько угодно раз.
#
# Примеры (в Docker, как остальные задачи проекта):
#   docker compose run --rm predictor backfill-forecasts                    # чемпион
#   docker compose run --rm predictor backfill-forecasts --model-version 1  # версия 1

import argparse
import logging
import sys
from datetime import date

import pandas as pd
import psycopg
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from moex_vol_forecast.config import load_settings
from moex_vol_forecast.db import add_missing_forecasts
from moex_vol_forecast.features import build_dataset
from moex_vol_forecast.logging_setup import setup_logging
from moex_vol_forecast.tracking import (
    CHAMPION,
    MODEL_NAME,
    first_test_day,
    load_version,
    setup_tracking,
)

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Backfill forecasts of a version.")
    # Номер версии модели в реестре MLflow. Не указан (None) — чемпион.
    # Имя ключа не --version: так обычно называют ключ «показать версию программы».
    parser.add_argument("--model-version", type=int)
    args = parser.parse_args()

    settings = load_settings()
    if not settings.database_url:
        logger.error("DATABASE_URL is not set")
        sys.exit(1)
    setup_tracking()
    try:
        client = MlflowClient()
        # Какую версию дописывать: указанную или ту, что сейчас под псевдонимом
        # champion в реестре.
        if args.model_version is None:
            version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version
        else:
            version = str(args.model_version)
        # С какого дня версия не видела данных: параметр test_start её запуска.
        start = first_test_day(client, version)
        model = load_version(version)
        # build_dataset — таблица для обучения: признаки каждого тикера на каждый
        # день. Берём из неё дни с начала тестового периода. В этой таблице нет
        # последнего дня каждого тикера (у него ещё нет следующего дня), поэтому
        # все дописанные прогнозы сразу можно сверить с фактом. Прогноз на завтра
        # по-прежнему считает ежедневный DAG.
        dataset = build_dataset(settings.data_dir / "raw", today=date.today())
        history = dataset[dataset["date"] >= pd.Timestamp(start)].copy()
        # Прогноз модели для каждой строки — так же, как в predict-vol.
        history["forecast_pk"] = model.predict(history)
        logger.info(
            "version %s: %d forecasts over the test period from %s",
            version,
            len(history),
            start,
        )
        # В базу — только те, которых там ещё нет (db.py, add_missing_forecasts).
        add_missing_forecasts(history, version, settings.database_url)
    # Ожидаемые ошибки: нет чемпиона или такой версии в реестре (MlflowException),
    # у запуска нет параметра test_start (KeyError), нет файлов со свечами
    # (OSError), плохие данные (ValueError), база недоступна (psycopg.Error).
    except (MlflowException, OSError, KeyError, ValueError, psycopg.Error) as err:
        logger.error("cannot backfill forecasts: %s", err)
        sys.exit(1)
