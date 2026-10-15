# Тесты настроек (config.py). Запуск: uv run pytest
# monkeypatch — помощник pytest: временно меняет переменные окружения на время
# одного теста и сам всё возвращает после него, поэтому тесты не влияют друг на друга.

from datetime import date

import pytest

from moex_vol_forecast.config import PROJECT_DIR, load_settings


def test_defaults(monkeypatch):
    # Убираем переменные, если они заданы в терминале: тест должен давать
    # одинаковый результат на любом компьютере. raising=False — не ругаться,
    # если переменной и так нет.
    monkeypatch.delenv("TICKER", raising=False)
    monkeypatch.delenv("FROM_DATE", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = load_settings()

    # Без переменных берутся значения по умолчанию; папка данных — data в проекте.
    assert settings.ticker == "SBER"
    assert settings.from_date == date(2026, 10, 1)
    assert settings.data_dir == PROJECT_DIR / "data"
    # Адрес базы не задан — в базу не пишем.
    assert settings.database_url is None


def test_reads_env_and_uppercases_ticker(monkeypatch):
    monkeypatch.setenv("TICKER", "gazp")
    monkeypatch.setenv("FROM_DATE", "2020-01-01")

    settings = load_settings()

    # Значения взяты из окружения, тикер приведён к верхнему регистру.
    assert settings.ticker == "GAZP"
    assert settings.from_date == date(2020, 1, 1)


def test_data_dir_from_env(monkeypatch, tmp_path):
    # tmp_path — временная папка, которую pytest создаёт для теста.
    # str(...) — переменные окружения бывают только строками.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    settings = load_settings()

    # Если DATA_DIR задана (как в Docker), используется именно она.
    assert settings.data_dir == tmp_path


def test_bad_date_fails_fast(monkeypatch):
    monkeypatch.setenv("FROM_DATE", "2026-13-01")

    # Тест проходит, только если внутри блока возникла ValueError:
    # неправильная дата должна ловиться сразу, до запросов к бирже.
    with pytest.raises(ValueError):
        load_settings()
