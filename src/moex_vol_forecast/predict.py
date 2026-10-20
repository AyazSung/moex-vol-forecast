"""Forecast next-day volatility with the champion model from the MLflow registry."""

# Команда `predict-vol`: берёт из реестра MLflow модель с псевдонимом champion
# и считает прогноз волатильности на следующий торговый день для каждого тикера.
# Это пакетный прогноз (batch scoring): модель применяется сразу ко всем тикерам,
# результат сохраняется в CSV, а если задан DATABASE_URL — ещё и в базу,
# вместе с номером версии модели.

import logging
import sys
from datetime import date

import psycopg
from mlflow.exceptions import MlflowException

from moex_vol_forecast.config import load_settings
from moex_vol_forecast.db import save_forecasts
from moex_vol_forecast.features import build_latest
from moex_vol_forecast.logging_setup import setup_logging
from moex_vol_forecast.storage import write_csv_atomic
from moex_vol_forecast.tracking import (
    CHAMPION,
    MODEL_NAME,
    load_champion,
    setup_tracking,
)

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    settings = load_settings()
    setup_tracking()
    try:
        champion = load_champion()
    except MlflowException as err:
        logger.error("cannot load %s@%s: %s", MODEL_NAME, CHAMPION, err)
        sys.exit(1)
    logger.info("champion: %s version %s", MODEL_NAME, champion.version)

    latest = build_latest(settings.data_dir / "raw", today=date.today())
    latest["forecast_pk"] = champion.predict(latest)

    result = latest[["ticker", "date", "pk_1", "forecast_pk"]]
    print(result.to_string(index=False))
    # Сохраняем прогноз той же атомарной записью, что и свечи.
    out_path = settings.data_dir / "predictions" / f"{date.today()}.csv"
    write_csv_atomic(result, out_path)

    if settings.database_url:
        try:
            save_forecasts(result, champion.version, settings.database_url)
        except psycopg.Error as err:
            logger.error("cannot save forecasts to PostgreSQL: %s", err)
            sys.exit(1)
