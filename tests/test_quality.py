# Тесты проверки качества (quality.py). База и MLflow им не нужны: каждый шаг
# проверки — обычная функция, которая получает таблицу и возвращает таблицу.
# Таблицы здесь маленькие и с придуманными числами, чтобы правильный ответ
# можно было посчитать в уме.

import pandas as pd
import pytest

from moex_vol_forecast.quality import (
    WINDOW,
    add_20_day_ratio,
    add_actuals,
    check_quality,
    current_version,
    quality_by_day,
)


def make_checked(misses, version=1):
    """Two tickers, one row per weekday: forecasts with their actual values.

    misses — список пар (промах модели, промах наивного прогноза), одна пара
    на день. Факт каждый день 0.02, а прогнозы отличаются от него ровно на
    заданные величины.
    """
    days = pd.bdate_range("2026-09-01", periods=len(misses))  # только будни
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "as_of": day,
                "model_version": version,
                "forecast_pk": 0.02 + model_miss,
                "naive_pk": 0.02 + naive_miss,
                "actual_pk": 0.02,
            }
            for day, (model_miss, naive_miss) in zip(days, misses, strict=True)
            for ticker in ["AAA", "BBB"]
        ]
    )


def evaluate(checked):
    """Steps 2 and 3 of the command: errors per day, then the 20-day window."""
    return add_20_day_ratio(quality_by_day(checked))


def test_only_forecasts_with_known_outcome_get_actuals():
    # Три прогноза: за 1 октября — от версий 1 и 2, за 2 октября — от версии 1.
    forecasts = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "as_of": pd.to_datetime(["2026-10-01", "2026-10-01", "2026-10-02"]),
            "model_version": [1, 2, 1],
            "forecast_pk": [0.01, 0.02, 0.03],
        }
    )
    # В таблице для обучения есть только 1 октября: 2 октября — последний день,
    # его следующий день ещё не закрыт, факт неизвестен.
    dataset = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": pd.to_datetime(["2026-10-01"]),
            "pk_1": [0.015],
            "target": [0.025],
        }
    )

    checked = add_actuals(forecasts, dataset)

    # Прогноз за 2 октября отпал. Оба прогноза за 1 октября получили один и тот
    # же факт (target) и наивный прогноз (pk_1) — под понятными именами.
    assert checked["model_version"].tolist() == [1, 2]
    assert checked["actual_pk"].tolist() == [0.025, 0.025]
    assert checked["naive_pk"].tolist() == [0.015, 0.015]


def test_metrics_of_one_day():
    # Модель каждый день промахивается на 0.01, наивный прогноз — на 0.02.
    days = quality_by_day(make_checked([(0.01, 0.02)] * 3))

    # Одна строка на день, в каждой два тикера. RMSE модели 0.01, наивного
    # 0.02 — модель вдвое точнее, отношение 0.5.
    assert days["tickers"].tolist() == [2, 2, 2]
    assert days["rmse"].tolist() == pytest.approx([0.01] * 3)
    assert days["naive_rmse"].tolist() == pytest.approx([0.02] * 3)
    assert days["rmse_to_naive"].tolist() == pytest.approx([0.5] * 3)


def test_window_needs_20_days():
    days = evaluate(make_checked([(0.01, 0.02)] * WINDOW))

    # Первые 19 дней — пусто: окно ещё не набралось. На 20-й день — отношение
    # за все 20 дней.
    assert days["rmse_to_naive_20d"].iloc[:-1].isna().all()
    assert days["rmse_to_naive_20d"].iloc[-1] == pytest.approx(0.5)


def test_one_bad_day_does_not_decide():
    # 20 хороших дней, затем один день, когда модель втрое хуже наивного.
    days = evaluate(make_checked([(0.01, 0.02)] * WINDOW + [(0.03, 0.01)]))

    # За сам плохой день отношение 3.0.
    assert days["rmse_to_naive"].iloc[-1] == pytest.approx(3.0)
    # В окно последнего дня попали 19 хороших дней и плохой. Суммы квадратов
    # промахов на одном тикере (у второго такие же, они сокращаются):
    #   модель:   19 * 0.01² + 0.03² = 0.0019 + 0.0009 = 0.0028
    #   наивный:  19 * 0.02² + 0.01² = 0.0076 + 0.0001 = 0.0077
    # Отношение за 20 дней — sqrt(0.0028 / 0.0077) ≈ 0.603: модель всё ещё лучше.
    assert days["rmse_to_naive_20d"].iloc[-1] == pytest.approx((0.0028 / 0.0077) ** 0.5)


def test_window_is_counted_per_version():
    # У версии 1 — 20 дней, у версии 2 — только пять. Дни версии 1 не
    # засчитываются версии 2: у неё окно ещё не набралось.
    checked = pd.concat(
        [
            make_checked([(0.01, 0.02)] * WINDOW, version=1),
            make_checked([(0.01, 0.02)] * 5, version=2),
        ]
    )

    days = evaluate(checked)

    assert days.loc[days["model_version"] == 1, "rmse_to_naive_20d"].notna().sum() == 1
    assert days.loc[days["model_version"] == 2, "rmse_to_naive_20d"].isna().all()


def test_current_version_is_the_version_of_the_newest_day():
    forecasts = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2026-10-21", "2026-10-22", "2026-10-21"]),
            "model_version": [1, 6, 7],
            # Прогноз версии 7 записан позже всех, но за прошлый день — так
            # выглядит дозаполнение. В работе — версия 6: её прогноз за самый
            # поздний день.
            "created_at": pd.to_datetime(
                ["2026-10-22 00:20", "2026-10-23 00:20", "2026-10-23 12:00"]
            ),
        }
    )

    assert current_version(forecasts) == 6


def test_check_quality():
    good = evaluate(make_checked([(0.01, 0.02)] * WINDOW))
    bad = evaluate(make_checked([(0.03, 0.02)] * WINDOW))
    young = evaluate(make_checked([(0.03, 0.02)] * 5))

    # Модель вдвое точнее наивного прогноза — проверка пройдена.
    assert check_quality(good, 1, max_ratio=1.0)[0]
    # Модель в полтора раза хуже наивного — не пройдена, число в сообщении.
    ok, message = check_quality(bad, 1, max_ratio=1.0)
    assert not ok
    assert "rmse_to_naive 1.5000" in message
    # Пять дней — судить рано. Это не ошибка: проверка пройдена и ждёт.
    ok, message = check_quality(young, 1, max_ratio=1.0)
    assert ok
    assert "not judged yet" in message
