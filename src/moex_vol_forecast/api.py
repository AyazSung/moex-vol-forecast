"""HTTP API: next-day volatility forecasts from the champion model."""

# Веб-сервис на FastAPI. Запуск на Mac (из корня проекта):
#   uv run uvicorn moex_vol_forecast.api:app --reload
# Документация и «пульт» для ручных запросов: http://localhost:8000/docs
#
# Адреса (endpoints):
#   GET  /health             — сервис жив;
#   GET  /model              — какая версия модели загружена;
#   POST /predict            — прогноз по признакам из запроса (онлайн-прогноз);
#   GET  /forecasts/{ticker} — последний прогноз тикера из базы (пакетный прогноз);
#   GET  /metrics            — метрики сервиса для Prometheus (этап 6).
#
# Модель-чемпион загружается один раз при старте сервиса. Если в реестре
# перевесили псевдоним champion, сервис нужно перезапустить.

import logging
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from importlib.metadata import version as package_version
from pathlib import Path

import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.routing import Match

from moex_vol_forecast.config import load_settings
from moex_vol_forecast.db import latest_forecast
from moex_vol_forecast.logging_setup import setup_logging
from moex_vol_forecast.tracking import (
    CHAMPION,
    MODEL_NAME,
    load_champion,
    setup_tracking,
)

logger = logging.getLogger(__name__)

# Файлы страницы документации Swagger UI (swagger-ui-dist 5.33.3). По умолчанию
# FastAPI берёт их из интернета (CDN), и без сети /docs была бы пустой.
# Поэтому они лежат в проекте, в папке static рядом с этим модулем.
STATIC_DIR = Path(__file__).parent / "static"


# ---------- форма запросов и ответов ----------
# Класс-наследник BaseModel описывает JSON: какие поля, какого типа, в каких
# пределах. FastAPI сам проверит запрос по этому описанию и при ошибке вернёт
# код 422 с объяснением, какое поле не так. Field(ge=0, le=1) — «от 0 до 1».
# examples — пример значения для страницы /docs: признаки SBER за 15.10.2026.
class Features(BaseModel):
    """Features of day t, all known at the close of that day."""

    pk_1: float = Field(
        ge=0, le=1, examples=[0.008192], description="Parkinson volatility of day t"
    )
    pk_5: float = Field(
        ge=0, le=1, examples=[0.008301], description="mean volatility over 5 days"
    )
    pk_22: float = Field(
        ge=0, le=1, examples=[0.010081], description="mean volatility over 22 days"
    )
    abs_ret_1: float = Field(
        ge=0, le=1, examples=[0.007241], description="absolute log return of day t"
    )
    volume_ratio: float = Field(
        ge=0, examples=[0.518488], description="volume of day t / 22-day mean"
    )


class Prediction(BaseModel):
    """Forecast of the next day's volatility."""

    forecast_pk: float
    model_version: str


class StoredForecast(BaseModel):
    """A forecast written to the database by `predict-vol`."""

    ticker: str
    as_of: date
    model_version: int
    forecast_pk: float
    created_at: datetime


# ---------- метрики для Prometheus (этап 6) ----------
# Prometheus — система сбора метрик. Каждые 15 секунд он заходит на адрес
# /metrics сервиса, забирает текущие значения и хранит их историю, а Grafana
# рисует по этой истории графики. Сам сервис хранит только текущие числа.
#
# Виды метрик:
#   Counter   — счётчик, который только растёт: сколько запросов пришло с
#               момента запуска. «Запросов в секунду» Prometheus считает сам —
#               по тому, как быстро растёт счётчик;
#   Histogram — раскладывает значения по корзинам: сколько запросов ответили
#               быстрее 5 мс, быстрее 10 мс и так далее. По корзинам Prometheus
#               считает, например, время, быстрее которого ответили 95% запросов;
#   Info      — постоянные сведения о сервисе: здесь — какая модель загружена.
#
# Метки (labels) — разрезы метрики: метод, адрес, код ответа. Адрес берём
# шаблоном (/forecasts/{ticker}), а не настоящим путём: иначе на каждый тикер
# появлялась бы своя строка метрики, и их число росло бы без предела.
REQUESTS = Counter(
    "api_requests_total",
    "HTTP requests by method, path template and status code",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "api_request_duration_seconds",
    "Time to answer an HTTP request, seconds",
    ["method", "path"],
)
MODEL = Info("api_model", "Model the API serves")
# Онлайн-прогнозы (POST /predict): какие значения отдаёт модель. Корзины — в
# единицах волатильности: в обычный день она около 0.01–0.02.
FORECASTS = Histogram(
    "api_forecast_pk",
    "Online forecasts of next-day volatility",
    buckets=[0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1],
)


# ---------- запуск сервиса ----------
# lifespan — код вокруг жизни сервиса: что до yield, выполняется один раз при
# старте (до первого запроса), что после — при остановке. Модель грузим здесь,
# а не в каждом запросе: загрузка занимает секунды, прогноз — миллисекунды.
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    setup_tracking()
    # app.state — «карман» приложения: положенное сюда видно во всех запросах.
    app.state.champion = load_champion()
    logger.info("loaded %s version %s", MODEL_NAME, app.state.champion.version)
    # Метрика api_model_info{name="moex-vol",version="6"} 1 — на дашборде видно,
    # какая версия сейчас в API и когда она сменилась.
    MODEL.info({"name": MODEL_NAME, "version": app.state.champion.version})
    yield


# docs_url=None, redoc_url=None — выключаем страницы документации со ссылками
# на CDN; свою /docs, на локальных файлах, объявляем ниже.
# version — версия нашего пакета из pyproject.toml: package_version читает её из
# установленного пакета, и после `uv version --bump` она обновится сама.
app = FastAPI(
    title="moex-vol",
    version=package_version("moex-vol-forecast"),
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
# Раздавать файлы из папки static по адресам /static/<имя файла>.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def path_template(request: Request) -> str:
    """Path template of the route that handles the request, e.g. /forecasts/{ticker}."""
    # Каждый адрес приложения умеет ответить, подходит ли ему запрос:
    #   FULL    — подходят и путь, и метод;
    #   PARTIAL — путь подходит, а метод нет (ответ будет 405);
    #   NONE    — не подходит.
    # Выбираем так же, как сам FastAPI: первый FULL, а если его нет — первый
    # PARTIAL. route.path — шаблон адреса: /forecasts/{ticker}, /openapi.json,
    # /static для всех файлов страницы документации.
    # FastAPI и сам отмечает в запросе, какой адрес его обработал, но только для
    # адресов, объявленных через @app.get и подобные. Служебный /openapi.json и
    # файлы /static он не отмечает, поэтому ищем сами.
    partial = None
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return route.path
        if match == Match.PARTIAL and partial is None:
            partial = route.path
    # Несуществующие адреса (ответ 404) — под одной меткой unmatched: иначе
    # каждый случайный путь давал бы новую строку метрики.
    return partial or "unmatched"


# Middleware — код, через который проходит каждый запрос до функции адреса и
# каждый ответ после неё. Здесь он засекает время и считает запросы.
@app.middleware("http")
async def measure_requests(request: Request, call_next):
    """Count every request and measure how long the answer took."""
    path = path_template(request)
    # Обращения самого Prometheus к /metrics не считаем: это служебный трафик.
    if path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    # call_next — передать запрос дальше, адресу, и дождаться ответа.
    response = await call_next(request)
    LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    return response


# include_in_schema=False — сама страница документации в документацию не попадает.
@app.get("/docs", include_in_schema=False)
def docs():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="moex-vol — docs",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon-32x32.png",
    )


# ---------- адреса ----------
# Декоратор @app.get("/путь") связывает адрес и метод HTTP с функцией.
# Что функция вернёт, FastAPI превратит в JSON.
@app.get("/health")
def health() -> dict:
    """Liveness check: the service is up."""
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict:
    """Which model version serves the forecasts."""
    champion = app.state.champion
    return {
        "name": MODEL_NAME,
        "alias": CHAMPION,
        "version": champion.version,
        "features": champion.columns,
    }


@app.post("/predict")
def predict(features: Features) -> Prediction:
    """Next-day volatility for the features in the request body."""
    champion = app.state.champion
    # Одна строка таблицы из полей запроса; модель берёт из неё свои столбцы.
    row = pd.DataFrame([features.model_dump()])
    value = float(champion.predict(row)[0])
    # Прогноз — ещё и в метрику: на дашборде видно, какие значения отдаёт модель.
    FORECASTS.observe(value)
    return Prediction(forecast_pk=value, model_version=champion.version)


@app.get("/forecasts/{ticker}")
def get_forecast(ticker: str) -> StoredForecast:
    """The latest forecast for a ticker saved by the batch job."""
    # {ticker} в пути — параметр: /forecasts/sber -> ticker="sber".
    database_url = load_settings().database_url
    if not database_url:
        # 503 — «сервис временно не может ответить»: база не настроена.
        raise HTTPException(status_code=503, detail="DATABASE_URL is not set")
    try:
        row = latest_forecast(ticker.upper(), database_url)
    except psycopg.Error as err:
        # База не отвечает или отказала — тоже 503, а не 500 с трассировкой:
        # клиенту понятно, что дело во временной недоступности, а не в его запросе.
        logger.error("cannot read forecasts: %s", err)
        raise HTTPException(status_code=503, detail="database is unavailable") from err
    if row is None:
        # 404 — «не найдено».
        raise HTTPException(status_code=404, detail=f"no forecast for {ticker.upper()}")
    return StoredForecast(**row)


# include_in_schema=False — служебный адрес, в документацию API не попадает.
@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Current metric values in the Prometheus text format."""
    # generate_latest() — все метрики текстом, по строке на значение, например:
    #   api_requests_total{method="GET",path="/health",status="200"} 3.0
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
