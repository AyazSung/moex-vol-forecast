"""Where MLflow keeps experiments, runs and registered models."""

# Общие настройки MLflow для обучения, пакетного прогноза и API.
# По умолчанию MLflow хранит всё в базе SQLite — одном файле mlflow.db в корне
# проекта, а файлы моделей — в папке mlruns рядом (этап 2).
# С этапа 3 адрес задают переменной MLFLOW_TRACKING_URI, например
# http://localhost:5001 — сервер MLflow из docker compose.

import os
from dataclasses import dataclass

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

from moex_vol_forecast.config import PROJECT_DIR

# Эксперимент — «папка» для запусков одной задачи; в UI это список слева.
EXPERIMENT = "moex-vol"
# Имя модели в реестре моделей MLflow.
MODEL_NAME = "moex-vol"
# Псевдоним версии, которой строятся прогнозы.
CHAMPION = "champion"


def setup_tracking() -> None:
    """Point MLflow at MLFLOW_TRACKING_URI or the project's mlflow.db."""
    default_uri = f"sqlite:///{PROJECT_DIR / 'mlflow.db'}"
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default_uri))


# frozen=True — объект нельзя изменить после создания, как кортеж.
@dataclass(frozen=True)
class Champion:
    """The champion model together with its registry version and input columns."""

    model: mlflow.pyfunc.PyFuncModel
    version: str
    columns: list[str]

    def predict(self, features: pd.DataFrame):
        """Forecast for every row; extra columns in `features` are ignored."""
        # Модели отдаём ровно те столбцы, которые записаны в её сигнатуре:
        # HAR ждёт три признака, бустинг — пять.
        return self.model.predict(features[self.columns])


def load_champion() -> Champion:
    """Load the model version that currently has the champion alias."""
    client = MlflowClient()
    # Сначала узнаём номер версии под псевдонимом, потом грузим именно эту версию.
    # Так номер в прогнозе точно совпадает с моделью, которой он посчитан,
    # даже если псевдоним перевесят в ту же секунду.
    version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    # Какие столбцы ждёт модель — MLflow сохранил это при обучении (input_example).
    columns = model.metadata.get_input_schema().input_names()
    return Champion(model=model, version=version, columns=columns)
