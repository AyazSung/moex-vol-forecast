from datetime import date

import pytest

from moex_vol_forecast.config import load_settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("TICKER", raising=False)
    monkeypatch.delenv("FROM_DATE", raising=False)

    settings = load_settings()

    assert settings.ticker == "SBER"
    assert settings.from_date == date(2026, 10, 1)


def test_reads_env_and_uppercases_ticker(monkeypatch):
    monkeypatch.setenv("TICKER", "gazp")
    monkeypatch.setenv("FROM_DATE", "2020-01-01")

    settings = load_settings()

    assert settings.ticker == "GAZP"
    assert settings.from_date == date(2020, 1, 1)


def test_bad_date_fails_fast(monkeypatch):
    monkeypatch.setenv("FROM_DATE", "2026-13-01")

    with pytest.raises(ValueError):
        load_settings()
