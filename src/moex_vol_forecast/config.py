"""Loader settings, read from environment variables."""

import os
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Settings:
    ticker: str
    from_date: date


def load_settings() -> Settings:
    """Build settings from environment variables, falling back to defaults."""
    ticker = os.environ.get("TICKER", "SBER").upper()
    from_date = date.fromisoformat(os.environ.get("FROM_DATE", "2026-10-01"))
    return Settings(ticker=ticker, from_date=from_date)
