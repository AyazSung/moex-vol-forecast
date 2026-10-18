"""Where MLflow keeps experiments, runs and registered models."""

# Общие настройки MLflow для обучения и прогноза.
# MLflow хранит всё в базе SQLite — одном файле mlflow.db в корне проекта,
# а файлы моделей (артефакты) — в папке mlruns рядом. Обе в .gitignore.
# Адрес можно поменять переменной MLFLOW_TRACKING_URI (например, на сервер MLflow).

import os

import mlflow

from moex_vol_forecast.config import PROJECT_DIR

# Эксперимент — «папка» для запусков одной задачи; в UI это список слева.
EXPERIMENT = "moex-vol"
# Имя модели в реестре моделей MLflow.
MODEL_NAME = "moex-vol"


def setup_tracking() -> None:
    """Point MLflow at the project's mlflow.db (or MLFLOW_TRACKING_URI)."""
    default_uri = f"sqlite:///{PROJECT_DIR / 'mlflow.db'}"
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default_uri))
