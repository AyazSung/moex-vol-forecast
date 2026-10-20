# Тесты API (api.py) без MLflow и без базы: вместо настоящей модели-чемпиона
# подставляем маленькую фальшивую. TestClient отправляет запросы приложению
# прямо в памяти — сервер запускать не нужно.

import numpy as np
import pytest
from fastapi.testclient import TestClient

from moex_vol_forecast import api
from moex_vol_forecast.tracking import Champion

FEATURES = {
    "pk_1": 0.02,
    "pk_5": 0.015,
    "pk_22": 0.014,
    "abs_ret_1": 0.01,
    "volume_ratio": 1.2,
}


class FakeModel:
    """Remembers which columns it received; forecast = half of pk_1."""

    def __init__(self):
        self.columns = None

    def predict(self, df):
        self.columns = list(df.columns)
        return np.asarray(df["pk_1"]) / 2


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def client(model, monkeypatch):
    champion = Champion(model=model, version="7", columns=["pk_1", "pk_5", "pk_22"])
    # При старте api вызывает load_champion — подменяем её на «вернуть фальшивку».
    monkeypatch.setattr(api, "load_champion", lambda: champion)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # with — запускает приложение (lifespan) и останавливает его после теста.
    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info(client):
    info = client.get("/model").json()

    assert info["version"] == "7"
    assert info["features"] == ["pk_1", "pk_5", "pk_22"]


def test_predict_uses_only_model_columns(client, model):
    response = client.post("/predict", json=FEATURES)

    assert response.status_code == 200
    assert response.json() == {"forecast_pk": 0.01, "model_version": "7"}
    # Модель получила только свои три столбца, а не все пять полей запроса.
    assert model.columns == ["pk_1", "pk_5", "pk_22"]


def test_predict_rejects_bad_input(client):
    # Волатильность не бывает отрицательной; поля volume_ratio нет вовсе.
    bad = {**FEATURES, "pk_1": -0.1}
    del bad["volume_ratio"]

    response = client.post("/predict", json=bad)

    # 422 — запрос не прошёл проверку; в ответе — какие поля и почему.
    assert response.status_code == 422
    fields = {error["loc"][-1] for error in response.json()["detail"]}
    assert fields == {"pk_1", "volume_ratio"}


def test_forecasts_without_database(client):
    response = client.get("/forecasts/sber")

    assert response.status_code == 503


def test_docs_work_without_internet(client):
    # Страница документации ссылается на свои файлы, а не на CDN в интернете.
    page = client.get("/docs").text
    assert "/static/swagger-ui-bundle.js" in page
    assert "cdn.jsdelivr.net" not in page
    assert client.get("/static/swagger-ui-bundle.js").status_code == 200
