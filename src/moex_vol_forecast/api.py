"""HTTP API: next-day volatility forecasts from the champion model."""

# Веб-сервис на FastAPI. Запуск на Mac (из корня проекта):
#   uv run uvicorn moex_vol_forecast.api:app --reload
# Документация и «пульт» для ручных запросов: http://localhost:8000/docs
#
# Адреса (endpoints):
#   GET  /health             — сервис жив;
#   GET  /model              — какая версия модели загружена;
#   POST /predict            — прогноз по признакам из запроса (онлайн-прогноз);
#   GET  /forecasts/{ticker} — последний прогноз тикера из базы (пакетный прогноз).
#
# Модель-чемпион загружается один раз при старте сервиса. Если в реестре
# перевесили псевдоним champion, сервис нужно перезапустить.

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from importlib.metadata import version as package_version
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
    return Prediction(forecast_pk=value, model_version=champion.version)


@app.get("/forecasts/{ticker}")
def get_forecast(ticker: str) -> StoredForecast:
    """The latest forecast for a ticker saved by the batch job."""
    # {ticker} в пути — параметр: /forecasts/sber -> ticker="sber".
    database_url = load_settings().database_url
    if not database_url:
        # 503 — «сервис временно не может ответить»: база не настроена.
        raise HTTPException(status_code=503, detail="DATABASE_URL is not set")
    row = latest_forecast(ticker.upper(), database_url)
    if row is None:
        # 404 — «не найдено».
        raise HTTPException(status_code=404, detail=f"no forecast for {ticker.upper()}")
    return StoredForecast(**row)
