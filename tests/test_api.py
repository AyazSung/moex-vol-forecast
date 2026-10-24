# Тесты API (api.py) без MLflow и без базы: вместо настоящей модели-чемпиона
# подставляем маленькую фальшивую. TestClient отправляет запросы приложению
# прямо в памяти — сервер запускать не нужно.

import numpy as np
import psycopg
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


def test_forecasts_when_database_is_down(client, monkeypatch):
    # База задана, но не отвечает: вместо 500 с трассировкой — понятный 503.
    def database_down(ticker, database_url):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setenv("DATABASE_URL", "postgresql://moex:moex@localhost:1/moex")
    monkeypatch.setattr(api, "latest_forecast", database_down)

    response = client.get("/forecasts/sber")

    assert response.status_code == 503
    assert response.json() == {"detail": "database is unavailable"}


def test_docs_work_without_internet(client):
    # Страница документации ссылается на свои файлы, а не на CDN в интернете.
    page = client.get("/docs").text
    assert "/static/swagger-ui-bundle.js" in page
    assert "cdn.jsdelivr.net" not in page
    assert client.get("/static/swagger-ui-bundle.js").status_code == 200


# ---------- метрики для Prometheus (этап 6) ----------
def metric_value(client, name):
    """Value of one metric line from /metrics; 0 if there is no such line yet."""
    # Строка метрики выглядит так: имя{метки} значение. Счётчики общие для всех
    # тестов, поэтому тесты сравнивают значение «до» и «после», а не с нулём.
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[-1])
    return 0.0


def test_metrics_count_requests_by_path_template(client):
    name = 'api_requests_total{method="GET",path="/forecasts/{ticker}",status="503"}'
    before = metric_value(client, name)

    client.get("/forecasts/sber")
    client.get("/forecasts/gazp")

    # Два разных тикера — одна строка метрики: адрес записан шаблоном. Без базы
    # ответ 503, и код ответа тоже попал в метки.
    assert metric_value(client, name) == before + 2


def test_metrics_label_service_and_unknown_paths(client):
    openapi = 'api_requests_total{method="GET",path="/openapi.json",status="200"}'
    unknown = 'api_requests_total{method="GET",path="unmatched",status="404"}'
    openapi_before = metric_value(client, openapi)
    unknown_before = metric_value(client, unknown)

    client.get("/openapi.json")
    client.get("/no-such-page")
    client.get("/another-missing-page")

    # Служебный адрес FastAPI записан своим шаблоном. Несуществующие адреса —
    # под одной меткой unmatched: случайные пути не плодят новых строк метрики.
    assert metric_value(client, openapi) == openapi_before + 1
    assert metric_value(client, unknown) == unknown_before + 2


def test_metrics_show_the_model_version(client):
    text = client.get("/metrics").text

    # Версия фальшивой модели из фикстуры — "7".
    assert 'api_model_info{name="moex-vol",version="7"} 1.0' in text


def test_metrics_requests_are_not_counted(client):
    client.get("/metrics")
    client.get("/metrics")

    # Служебные обращения к /metrics в счётчик запросов не попадают.
    assert 'path="/metrics"' not in client.get("/metrics").text


def test_online_forecasts_are_recorded(client):
    before = metric_value(client, "api_forecast_pk_count")

    client.post("/predict", json=FEATURES)

    # _count — сколько значений попало в гистограмму прогнозов.
    assert metric_value(client, "api_forecast_pk_count") == before + 1
