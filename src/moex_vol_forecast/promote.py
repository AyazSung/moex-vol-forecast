"""Make a newly trained model version the champion if it beats the current one."""

# Команда `promote-model`: решает, станет ли кандидат — новая версия модели в
# реестре MLflow — чемпионом, то есть моделью, которой строятся прогнозы.
#
# Сравнение честное: кандидат и чемпион считают прогноз на одном и том же
# отрезке — последних неделях, которые кандидат при обучении не видел (его
# параметр test_start). Метрика та же, что при обучении: rmse_to_naive, меньше —
# лучше. Кандидат становится чемпионом, только если он лучше хотя бы на
# --min-improvement (по умолчанию 1%): меньшая разница может оказаться случайной,
# и менять модель ради неё не стоит.
#
# Решение — promoted (стал чемпионом) или kept (чемпион прежний) — видно в логе и
# в тегах версии в реестре MLflow.
#
# Примеры (в Docker, как остальные задачи проекта):
#   docker compose run --rm trainer promote-model                    # последняя версия
#   docker compose run --rm trainer promote-model --candidate 3      # версия 3
#   docker compose run --rm trainer promote-model --min-improvement 0  # любое улучшение

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import root_mean_squared_error

from moex_vol_forecast.config import load_settings
from moex_vol_forecast.features import TARGET, build_dataset
from moex_vol_forecast.logging_setup import setup_logging
from moex_vol_forecast.tracking import (
    CHAMPION,
    MODEL_NAME,
    Champion,
    load_version,
    setup_tracking,
)

logger = logging.getLogger(__name__)


def score(model: Champion, window: pd.DataFrame) -> float:
    """rmse_to_naive of the model on the window; below 1 means better than naive."""
    # Наивный прогноз «завтра как сегодня» — это pk_1 (train.py, этап 2).
    naive = root_mean_squared_error(window[TARGET], window["pk_1"])
    return root_mean_squared_error(window[TARGET], model.predict(window)) / naive


def should_promote(
    candidate: float, champion: float | None, min_improvement: float
) -> bool:
    """True if the candidate's metric is lower than the champion's by the margin."""
    # Чемпиона ещё нет — кандидат становится им без сравнения.
    if champion is None:
        return True
    # min_improvement=0.01 — метрика кандидата должна быть меньше 99% метрики
    # чемпиона, то есть кандидат лучше хотя бы на 1%.
    return candidate < champion * (1 - min_improvement)


def first_test_day(client: MlflowClient, version: str) -> str:
    """Start of the test period of the run that trained the version."""
    # Обучение записывает test_start в параметры запуска (train.py), например
    # "2026-07-23"; до этого дня модель училась, с него — проверялась.
    run_id = client.get_model_version(MODEL_NAME, version).run_id
    return client.get_run(run_id).data.params["test_start"]


def decide(client: MlflowClient, raw_dir: Path, args: argparse.Namespace) -> None:
    """Compare the candidate with the champion; move the alias if it wins."""
    # aliases — словарь «псевдоним → номер версии», например {"champion": "1"}.
    champion = client.get_registered_model(MODEL_NAME).aliases.get(CHAMPION)
    if args.candidate is None:
        # Кандидат не указан — берём последнюю версию: с наибольшим номером.
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        candidate = str(max(int(v.version) for v in versions))
    else:
        candidate = str(args.candidate)
    if candidate == champion:
        logger.info("version %s is already the champion", candidate)
        return

    # Отрезок сравнения — тестовый период кандидата: эти дни он при обучении не
    # видел. Чемпион тоже не должен был их видеть: если он учился на данных
    # позже начала отрезка, на части дней он «знает ответ», и сравнение нечестное.
    window_start = first_test_day(client, candidate)
    if champion is not None:
        champion_start = first_test_day(client, champion)
        if pd.Timestamp(champion_start) > pd.Timestamp(window_start):
            raise ValueError(
                f"champion version {champion} was trained on data up to "
                f"{champion_start}, after the window start {window_start}"
            )

    dataset = build_dataset(raw_dir, today=date.today())
    window = dataset[dataset["date"] >= pd.Timestamp(window_start)]
    candidate_score = score(load_version(candidate), window)
    champion_score = score(load_version(champion), window) if champion else None
    promote = should_promote(candidate_score, champion_score, args.min_improvement)
    decision = "promoted" if promote else "kept"
    logger.info(
        "window from %s: %d rows; candidate version %s rmse_to_naive=%.4f; "
        "champion version %s rmse_to_naive=%s",
        window_start,
        len(window),
        candidate,
        candidate_score,
        champion,
        f"{champion_score:.4f}" if champion else "-",
    )

    # Пометки (теги) на версии кандидата: в реестре MLflow будет видно, на каком
    # отрезке её сравнивали, с какими числами и что решили.
    tags = {
        "window_start": window_start,
        "rmse_to_naive": f"{candidate_score:.4f}",
        "decision": decision,
    }
    if champion is not None:
        tags["champion_version"] = champion
        tags["champion_rmse_to_naive"] = f"{champion_score:.4f}"
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, candidate, key, value)

    if promote:
        # Перевесить псевдоним: прогнозы пойдут от кандидата.
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, candidate)
    logger.info("%s version %s", decision, candidate)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Promote a model version.")
    # type=int — номер версии; не указан (None) — последняя версия.
    parser.add_argument("--candidate", type=int)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    args = parser.parse_args()

    settings = load_settings()
    setup_tracking()
    try:
        decide(MlflowClient(), settings.data_dir / "raw", args)
    except (MlflowException, OSError, KeyError, ValueError) as err:
        logger.error("cannot compare models: %s", err)
        sys.exit(1)
