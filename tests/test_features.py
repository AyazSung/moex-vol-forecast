# Тесты признаков (features.py). Главное, что они проверяют, — отсутствие
# «заглядывания в будущее»: признаки дня t считаются только по данным до дня t
# включительно, а будущее используется только для цели. Ошибка здесь сделала бы
# модель отличной на истории и бесполезной в жизни.

from datetime import date

import numpy as np
import pandas as pd

from moex_vol_forecast.features import FEATURES, TARGET, add_features, clean_candles


def make_candles(ticker, days=30, start="2026-01-05"):
    """Fake weekday candles with slowly rising prices."""
    dates = pd.bdate_range(start, periods=days)  # только будние дни
    rng = np.random.default_rng(0)  # фиксированная случайность — тест повторяем
    close = 100 + np.cumsum(rng.normal(0, 1, days))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": rng.integers(1_000, 2_000, days),
        }
    )


def test_target_is_next_day_volatility():
    df = add_features(make_candles("AAA"))

    # Цель строки t равна волатильности pk строки t+1 того же тикера.
    assert df[TARGET].iloc[:-1].tolist() == df["pk"].iloc[1:].tolist()
    # У последнего дня следующего дня нет — цель неизвестна.
    assert np.isnan(df[TARGET].iloc[-1])


def test_features_do_not_look_into_the_future():
    candles = make_candles("AAA")
    before = add_features(candles)

    # Портим цены последних пяти дней — как будто будущее оказалось другим.
    changed = candles.copy()
    changed.loc[changed.index[-5:], ["high", "low", "close"]] *= [3.0, 0.3, 2.0]
    after = add_features(changed)

    # Признаки более ранних дней не должны измениться ни на волос.
    pd.testing.assert_frame_equal(before[FEATURES].iloc[:-5], after[FEATURES].iloc[:-5])


def test_tickers_do_not_mix():
    candles = pd.concat([make_candles("AAA"), make_candles("BBB")], ignore_index=True)
    df = add_features(candles)

    # Последний день каждого тикера — без цели: сдвиг не взял день другого тикера.
    last_rows = df.groupby("ticker").tail(1)
    assert last_rows[TARGET].isna().all()


def test_clean_drops_weekends_and_today():
    candles = make_candles("AAA", days=5, start="2026-10-12")  # пн..пт
    saturday = candles.iloc[[0]].assign(date=pd.Timestamp("2026-10-17"))
    candles = pd.concat([candles, saturday], ignore_index=True)

    cleaned = clean_candles(candles, today=date(2026, 10, 16))

    # Остались пн..чт: суббота (торги выходного дня) и «сегодня» (пятница) убраны.
    assert cleaned["date"].dt.dayofweek.max() < 5
    assert cleaned["date"].max() == pd.Timestamp("2026-10-15")
