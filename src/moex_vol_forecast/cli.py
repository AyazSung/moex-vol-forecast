"""Command line entry point: fetch candles and save them to data/raw."""

import logging
import os
import signal
import sys
from pathlib import Path

import requests

from moex_vol_forecast.config import load_settings
from moex_vol_forecast.moex import fetch_candles
from moex_vol_forecast.storage import write_csv_atomic

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"


def exit_on_sigterm(signum: int, frame: object) -> None:
    """Turn SIGTERM into a normal exit so that finally blocks still run."""
    raise SystemExit(128 + signum)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    signal.signal(signal.SIGTERM, exit_on_sigterm)

    try:
        settings = load_settings()
        candles = fetch_candles(settings.ticker, settings.from_date)
        if candles.empty:
            raise ValueError(
                f"no candles for {settings.ticker} from {settings.from_date}"
            )
        out_path = RAW_DIR / f"{settings.ticker.lower()}.csv"
        write_csv_atomic(candles, out_path)
    except (ValueError, RuntimeError, OSError, requests.RequestException) as err:
        logger.error("%s", err)
        sys.exit(1)

    logger.info(
        "%s: %d candles from %s", settings.ticker, len(candles), settings.from_date
    )
