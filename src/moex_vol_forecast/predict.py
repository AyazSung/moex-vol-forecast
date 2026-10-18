"""Forecast next-day volatility with the champion model from the MLflow registry."""

# Команда `predict-vol`: берёт из реестра MLflow модель с псевдонимом champion
# и считает прогноз волатильности на следующий торговый день для каждого тикера.
# Это прообраз пакетного скоринга (этап 3): обученная модель применяется
# к свежим данным, а результат сохраняется в файл.

import logging
import sys
from datetime import date

import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException

from moex_vol_forecast.config import load_settings
from moex_vol_forecast.features import build_latest
from moex_vol_forecast.logging_setup import setup_logging
from moex_vol_forecast.storage import write_csv_atomic
from moex_vol_forecast.tracking import MODEL_NAME, setup_tracking

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    settings = load_settings()
    setup_tracking()
    try:
        # models:/<имя>@<псевдоним> — модель из реестра по псевдониму. Псевдоним
        # champion переставляют на лучшую версию, а код прогноза не меняется.
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    except MlflowException as err:
        logger.error("cannot load %s@champion: %s", MODEL_NAME, err)
        sys.exit(1)

    # Какие столбцы ждёт модель — MLflow сохранил это при обучении (input_example).
    columns = model.metadata.get_input_schema().input_names()
    latest = build_latest(settings.data_dir / "raw", today=date.today())
    latest["forecast_pk"] = model.predict(latest[columns])

    result = latest[["ticker", "date", "pk_1", "forecast_pk"]]
    print(result.to_string(index=False))
    # Сохраняем прогноз той же атомарной записью, что и свечи.
    out_path = settings.data_dir / "predictions" / f"{date.today()}.csv"
    write_csv_atomic(result, out_path)
