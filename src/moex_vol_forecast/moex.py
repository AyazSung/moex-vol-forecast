"""MOEX ISS client: daily candles, downloaded page by page."""

import logging
import time
from datetime import date

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

URL = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
PAGE_SIZE = 500
MAX_PAGES = 50
PAUSE_SECONDS = 0.5
TIMEOUT_SECONDS = 30


def make_session() -> requests.Session:
    """HTTP session that retries network errors and 429/5xx answers."""
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_candles(ticker: str, from_date: date, session: requests.Session | None = None) -> pd.DataFrame:
    """Download all daily candles for ticker since from_date."""
    session = session or make_session()
    url = URL.format(ticker=ticker)
    frames = []
    start = 0
    page_num = 0
    while True:
        page_num += 1
        if page_num > MAX_PAGES:
            raise RuntimeError(f"more than {MAX_PAGES} pages for {ticker}, stopping")

        params = {"interval": 24, "from": from_date.isoformat(), "start": start, "iss.meta": "off"}
        response = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        block = response.json()["candles"]
        rows = block["data"]
        logger.info("page %d: %d rows (start=%d)", page_num, len(rows), start)

        if rows:
            frames.append(pd.DataFrame(rows, columns=block["columns"]))
        if len(rows) < PAGE_SIZE:
            break
        start += PAGE_SIZE
        time.sleep(PAUSE_SECONDS)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)