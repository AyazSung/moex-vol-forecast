# Тесты выбора чемпиона (promote.py) без MLflow: правило решения и метрика —
# обычные функции, им хватает чисел и маленькой таблицы.

import numpy as np
import pandas as pd
import pytest

from moex_vol_forecast.promote import score, should_promote, summary


def test_first_model_becomes_champion():
    # Чемпиона ещё нет — сравнивать не с чем.
    assert should_promote(0.9, None, min_improvement=0.01)


def test_clearly_better_candidate_is_promoted():
    # 0.78 против 0.80 — лучше на 2.5%, порог 1% пройден.
    assert should_promote(0.78, 0.80, min_improvement=0.01)


def test_tiny_improvement_is_not_enough():
    # Числа с наших данных: 0.7895 против 0.7907 — лучше всего на 0.15%.
    assert not should_promote(0.7895, 0.7907, min_improvement=0.01)
    # Без порога такое улучшение засчиталось бы.
    assert should_promote(0.7895, 0.7907, min_improvement=0)


def test_worse_or_equal_candidate_is_kept():
    assert not should_promote(0.81, 0.80, min_improvement=0)
    assert not should_promote(0.80, 0.80, min_improvement=0)


class HalfModel:
    """Fake model: the forecast is half of pk_1."""

    def predict(self, df):
        return np.asarray(df["pk_1"]) / 2


def test_score_is_error_relative_to_naive_forecast():
    window = pd.DataFrame({"pk_1": [0.02, 0.04], "target": [0.02, 0.02]})

    # Наивный прогноз (pk_1) ошибается на 0 и 0.02, его rmse = 0.02 / √2.
    # Модель (pk_1 / 2) — на 0.01 и 0, её rmse = 0.01 / √2: вдвое меньше.
    assert score(HalfModel(), window) == pytest.approx(0.5)


def test_summary_names_the_champion():
    # Решение — о кандидате, а в строке сразу видно, кто чемпион. Раньше было
    # «kept version 5», и это читалось как «оставили версию 5».
    assert summary(False, "5", "1") == "rejected version 5: champion stays version 1"
    assert summary(True, "6", "1") == "promoted version 6: new champion, was version 1"
    assert summary(True, "1", None) == "promoted version 1: the first champion"
