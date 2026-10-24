# Тесты проверки дрейфа (drift.py). База и MLflow не нужны: PSI, корзины, окна
# и выход за границы — обычные функции над маленькими таблицами с придуманными
# числами.

import numpy as np
import pandas as pd
import pytest

from moex_vol_forecast.drift import (
    WINDOW,
    bin_edges,
    daily_counts,
    drift_table,
    psi,
    window_psi,
)


def make_rows(values_by_day, start="2026-01-05"):
    """One feature x, several tickers per weekday: values_by_day[i] — day i."""
    days = pd.bdate_range(start, periods=len(values_by_day))
    return pd.DataFrame(
        [
            {"ticker": f"T{k}", "date": day, "x": value}
            for day, values in zip(days, values_by_day, strict=True)
            for k, value in enumerate(values)
        ]
    )


def test_psi_is_zero_for_the_same_distribution():
    shares = np.full(10, 0.1)
    assert psi(shares, shares) == pytest.approx(0.0)


def test_psi_by_hand():
    # Две корзины. В обучении — пополам, сейчас — 90% и 10%:
    #   (0.9 − 0.5) × ln(0.9 / 0.5) + (0.1 − 0.5) × ln(0.1 / 0.5)
    # = 0.4 × 0.588 + (−0.4) × (−1.609) = 0.235 + 0.644 ≈ 0.879.
    # Оба слагаемых положительны: расхождение в любую сторону увеличивает PSI.
    value = psi(np.array([0.9, 0.1]), np.array([0.5, 0.5]))
    assert value == pytest.approx(0.4 * np.log(1.8) + 0.4 * np.log(5))


def test_bins_hold_ten_percent_of_the_reference_each():
    rows = make_rows([np.arange(1, 101)])  # один день, значения 1..100

    counts = daily_counts(rows, "x", bin_edges(rows["x"]))

    # Границы корзин — децили, поэтому в каждую корзину попало по 10 значений.
    assert counts.iloc[0].tolist() == [10] * 10


def test_window_psi_matches_a_direct_calculation():
    rng = np.random.default_rng(0)  # фиксированная случайность — тест повторяем
    rows = make_rows([rng.normal(size=8) for _ in range(30)])
    reference = rows[rows["date"] < rows["date"].unique()[10]]
    edges = bin_edges(reference["x"])
    reference_shares = np.histogram(reference["x"], edges)[0] / len(reference)

    by_window = window_psi(daily_counts(rows, "x", edges), reference_shares)

    # То же для последнего окна напрямую: 20 последних дней, доли по корзинам.
    last_days = rows["date"].unique()[-WINDOW:]
    window = rows[rows["date"].isin(last_days)]["x"]
    shares = np.histogram(window, edges)[0] / len(window)
    assert by_window.iloc[-1] == pytest.approx(psi(shares, reference_shares))
    # Окно набирается с 20-го дня: из 30 дней получается 11 окон.
    assert len(by_window) == 30 - WINDOW + 1


def test_stable_data_pass_and_shifted_data_fail():
    rng = np.random.default_rng(1)
    # 200 дней обучения и 40 тестовых дней из того же распределения, затем
    # 40 дней, где все значения сдвинуты далеко вверх.
    stable = [rng.uniform(0, 1, size=10) for _ in range(240)]
    shifted = [rng.uniform(5, 6, size=10) for _ in range(40)]
    rows = make_rows(stable + shifted)
    test_start = rows["date"].unique()[200]

    table = drift_table(rows, ["x"], test_start)
    before_shift = table[table["as_of"] < rows["date"].unique()[240]]
    after_shift = table[table["as_of"] >= rows["date"].unique()[259]]

    # Пока данные те же, что в обучении: PSI ниже порога, за границы никто не вышел.
    assert (before_shift["psi"] < before_shift["psi_limit"]).all()
    assert (before_shift["outside"] == 0).all()
    # Через 20 дней после сдвига окно целиком из новых значений: PSI выше порога,
    # и все 10 значений дня — выше максимума обучения.
    assert (after_shift["psi"] > after_shift["psi_limit"]).all()
    assert (after_shift["outside"] == 10).all()


def test_one_value_outside_the_training_range_is_counted():
    # Обучение — значения от 1 до 2. В тестовый день у одного тикера 3.
    rows = make_rows([[1.0, 1.5, 2.0]] * 30 + [[1.0, 1.5, 3.0]])
    test_start = rows["date"].unique()[30]

    table = drift_table(rows, ["x"], test_start)

    assert table["outside"].tolist() == [1]
