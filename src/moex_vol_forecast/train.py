"""Train a next-day volatility model and log it to MLflow."""

# Команда `train-model`: обучает модель прогноза волатильности на завтра и
# записывает в MLflow всё, что нужно, чтобы потом сравнить запуски:
# параметры, метрики качества и саму модель.
#
# Примеры:
#   uv run train-model --model naive
#   uv run train-model --model har
#   uv run train-model --model gbm --learning-rate 0.05 --max-iter 300
#   uv run train-model --model gbm --register    # ещё и добавить в реестр моделей

import argparse
import logging
import sys
from datetime import date

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from moex_vol_forecast.config import load_settings
from moex_vol_forecast.features import FEATURES, TARGET, build_dataset
from moex_vol_forecast.logging_setup import setup_logging
from moex_vol_forecast.tracking import EXPERIMENT, MODEL_NAME, setup_tracking

logger = logging.getLogger(__name__)

# Признаки модели HAR: дневная, недельная и месячная волатильность.
HAR_FEATURES = ["pk_1", "pk_5", "pk_22"]

# MLflow сохраняет модели sklearn в безопасном формате skops: при загрузке он
# проверяет, какие типы объектов лежат в файле, и чужой код не выполнит.
# Внутренний тип деревьев HistGradientBoosting skops сам не пропускает —
# модель мы обучили сами, поэтому явно разрешаем этот тип.
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def parse_args() -> argparse.Namespace:
    """Read command line options like --model gbm --max-iter 300."""
    # argparse — стандартный модуль Python для разбора аргументов командной строки.
    # Каждый add_argument описывает один параметр: имя, тип, значение по умолчанию.
    parser = argparse.ArgumentParser(description="Train a next-day volatility model.")
    parser.add_argument("--model", choices=["naive", "har", "gbm"], default="har")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    # action="store_true" — флаг без значения: указан -> True, не указан -> False.
    parser.add_argument("--register", action="store_true")
    return parser.parse_args()


def make_model(args: argparse.Namespace):
    """Return (sklearn model or None, list of feature columns)."""
    if args.model == "har":
        # Линейная регрессия на трёх компонентах HAR.
        return LinearRegression(), HAR_FEATURES
    if args.model == "gbm":
        # Градиентный бустинг: много небольших деревьев, каждое исправляет ошибки
        # предыдущих. random_state — фиксируем случайность для воспроизводимости.
        model = HistGradientBoostingRegressor(
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            random_state=42,
        )
        return model, FEATURES
    # naive: «завтра будет как сегодня» — прогноз равен pk_1, обучать нечего.
    # Это точка отсчёта: модель имеет смысл, только если она лучше naive.
    return None, ["pk_1"]


def main() -> None:
    setup_logging()
    args = parse_args()
    settings = load_settings()
    try:
        dataset = build_dataset(settings.data_dir / "raw", today=date.today())
        test_start = pd.Timestamp(args.test_start)
    except (FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        sys.exit(1)

    # Делим по времени, а не случайно: учимся на прошлом, проверяем на будущем,
    # как модель и будет работать в жизни. Случайное деление «подсмотрело» бы будущее.
    train = dataset[dataset["date"] < test_start]
    test = dataset[dataset["date"] >= test_start]

    model, columns = make_model(args)
    if model is None:
        predictions = test["pk_1"].to_numpy()
    else:
        model.fit(train[columns], train[TARGET])
        predictions = model.predict(test[columns])

    # Метрики на тестовом периоде (чем меньше, тем лучше):
    #   rmse — корень из средней квадратичной ошибки, сильно штрафует большие промахи;
    #   mae  — средняя абсолютная ошибка;
    #   rmse_to_naive — rmse модели / rmse наивного прогноза; < 1 — модель лучше naive.
    rmse = root_mean_squared_error(test[TARGET], predictions)
    mae = mean_absolute_error(test[TARGET], predictions)
    naive_rmse = root_mean_squared_error(test[TARGET], test["pk_1"])

    setup_tracking()
    mlflow.set_experiment(EXPERIMENT)
    # start_run — начать запуск: всё, что записано внутри блока, попадёт в один run.
    with mlflow.start_run(run_name=args.model) as run:
        # Параметры — настройки запуска (что пробовали).
        mlflow.log_params(
            {
                "model": args.model,
                "features": ",".join(columns),
                "test_start": args.test_start,
                "train_rows": len(train),
                "test_rows": len(test),
            }
        )
        if args.model == "gbm":
            mlflow.log_params(
                {
                    "learning_rate": args.learning_rate,
                    "max_iter": args.max_iter,
                    "max_depth": args.max_depth,
                }
            )
        # Метрики — результат (что получилось).
        mlflow.log_metrics(
            {"rmse": rmse, "mae": mae, "rmse_to_naive": rmse / naive_rmse}
        )
        if model is not None:
            # Сохранить саму модель как артефакт запуска. input_example — пример
            # входных данных: по нему MLflow запомнит, какие столбцы ждёт модель.
            # registered_model_name — заодно добавить новую версию в реестр моделей.
            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=test[columns].head(5),
                registered_model_name=MODEL_NAME if args.register else None,
                skops_trusted_types=TRUSTED_TYPES,
            )

    logger.info(
        "%s: rmse=%.5f mae=%.5f rmse_to_naive=%.3f run_id=%s",
        args.model,
        rmse,
        mae,
        rmse / naive_rmse,
        run.info.run_id,
    )
