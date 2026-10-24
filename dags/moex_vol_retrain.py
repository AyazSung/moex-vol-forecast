"""Weekly retraining: train a candidate, promote it if better, refresh the API."""

# Еженедельное переобучение модели:
#
#   train_candidate → promote → api_is_stale → restart_api → check_api
#     → backfill_forecasts
#
# Кандидат учится на всех данных, кроме последних 13 недель, и сравнивается с
# чемпионом на этих неделях (команда promote-model, часть 4). Если чемпион
# сменился, API перезапускается: модель он загружает только при старте (этап 3).
# В конце в таблицу forecasts дописываются прогнозы нового чемпиона за эти
# 13 недель — чтобы проверка качества сразу могла о нём судить (этап 6).

import json
import time
import urllib.request
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param, dag, task

# Папка проекта: здесь лежит compose.yaml, из неё задачи запускают docker compose.
PROJECT_DIR = str(Path(__file__).resolve().parents[1])
# Адреса с Mac: API проекта и сервер MLflow (5001: порт 5000 на Mac занят AirPlay).
API_URL = "http://localhost:8000"
MLFLOW_URL = "http://localhost:5001"


def get_json(url: str) -> dict:
    """GET the URL and parse the JSON answer."""
    # urllib — из стандартной библиотеки Python. Задачи-функции выполняет Python
    # самого Airflow, а библиотек проекта (requests и других) в нём нет.
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def served_version() -> str:
    """Model version the API serves now."""
    return get_json(f"{API_URL}/model")["version"]


def champion_version() -> str:
    """Model version with the champion alias in the MLflow registry."""
    # REST API сервера MLflow: какая версия модели moex-vol под псевдонимом champion.
    url = (
        f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/alias"
        "?name=moex-vol&alias=champion"
    )
    return get_json(url)["model_version"]["version"]


@dag(
    # По субботам в 02:00 по Москве: торги пятницы закрыты, а ежедневный DAG в
    # 00:15 уже загрузил свечи.
    schedule="0 2 * * 6",
    start_date=pendulum.datetime(2026, 10, 1, tz="Europe/Moscow"),
    catchup=False,
    max_active_runs=1,
    # При сбое — ещё одна попытка через две минуты.
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    # Параметры запуска: какую модель учить и какой нужен запас, чтобы сменить
    # чемпиона. В форме запуска model — список, min_improvement — число.
    params={
        "model": Param("har", enum=["har", "gbm"], description="model to train"),
        "min_improvement": Param(
            0.01, type="number", minimum=0, description="0.01 = at least 1% better"
        ),
    },
    tags=["moex"],
)
def moex_vol_retrain():
    # Кандидат учится на данных до «срок запуска минус 13 недель», проверяется
    # на последних 13 неделях и попадает в реестр новой версией (--register).
    train_candidate = BashOperator(
        task_id="train_candidate",
        cwd=PROJECT_DIR,
        bash_command=(
            "docker compose run --rm trainer train-model "
            "--model {{ params.model }} --register "
            "--test-start {{ (dag_run.run_after - macros.timedelta(weeks=13)) | ds }}"
        ),
    )

    # Сравнение с чемпионом на тех же 13 неделях. Если кандидат лучше хотя бы на
    # min_improvement, псевдоним champion переходит на него (часть 4).
    promote = BashOperator(
        task_id="promote",
        cwd=PROJECT_DIR,
        bash_command=(
            "docker compose run --rm trainer promote-model "
            "--min-improvement {{ params.min_improvement }}"
        ),
    )

    # Развилка: вернула True — идём дальше, False — все задачи после неё
    # пропускаются (skipped). Перезапуск нужен, только если API отдаёт не ту
    # версию, что сейчас чемпион в реестре. Так подхватится и чемпион, которого
    # перевесили руками в интерфейсе MLflow.
    @task.short_circuit
    def api_is_stale() -> bool:
        served, champion = served_version(), champion_version()
        print(f"API serves version {served}, champion is version {champion}")
        return served != champion

    # API загружает модель-чемпиона только при старте — перезапускаем контейнер.
    restart_api = BashOperator(
        task_id="restart_api",
        cwd=PROJECT_DIR,
        bash_command="docker compose restart api",
    )

    # Проверка: API снова отвечает и отдаёт чемпиона. Пока API запускается, он
    # отклоняет соединения (OSError) — ждём до минуты.
    @task
    def check_api() -> None:
        champion = champion_version()
        for _ in range(30):
            try:
                served = served_version()
            except OSError:
                time.sleep(2)
                continue
            print(f"API serves version {served}, champion is version {champion}")
            if served != champion:
                raise RuntimeError("the API does not serve the champion")
            return
        raise RuntimeError("the API did not start in 60 seconds")

    # Дописать прогнозы нового чемпиона (этап 6). Проверке качества в ежедневном
    # DAG нужны прогнозы версии за 20 дней, а у нового чемпиона их ещё нет.
    # Команда backfill-forecasts считает его прогнозы на те 13 недель, на
    # которых он не учился, — и проверка может судить о нём сразу, а не через
    # месяц. Задача стоит после api_is_stale, поэтому, как и перезапуск API,
    # выполняется только когда чемпион сменился; иначе её пропускают (skipped).
    backfill_forecasts = BashOperator(
        task_id="backfill_forecasts",
        cwd=PROJECT_DIR,
        bash_command="docker compose run --rm predictor backfill-forecasts",
    )

    (
        train_candidate
        >> promote
        >> api_is_stale()
        >> restart_api
        >> check_api()
        >> backfill_forecasts
    )


moex_vol_retrain()
