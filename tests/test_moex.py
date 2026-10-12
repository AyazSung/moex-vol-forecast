from datetime import date

import pytest

from moex_vol_forecast import moex

COLUMNS = ["open", "close", "high", "low", "value", "volume", "begin", "end"]


def make_rows(count, first=0):
    """Fake candles; row number goes into begin/end so every row is unique."""
    return [
        [1.0, 2.0, 3.0, 0.5, 100.0, 10, f"row-{first + i}", f"row-{first + i}"]
        for i in range(count)
    ]


class FakeResponse:
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        pass

    def json(self):
        return {"candles": {"columns": COLUMNS, "data": self.rows}}


class FakeSession:
    """Pretends to be MOEX: serves `total` candles page by page."""

    def __init__(self, total):
        self.total = total
        self.starts = []

    def get(self, url, params, timeout):
        start = params["start"]
        self.starts.append(start)
        count = max(0, min(moex.PAGE_SIZE, self.total - start))
        return FakeResponse(make_rows(count, first=start))


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(moex, "PAUSE_SECONDS", 0)


def test_single_page():
    session = FakeSession(total=12)

    df = moex.fetch_candles("SBER", date(2026, 10, 1), session=session)

    assert len(df) == 12
    assert list(df.columns) == COLUMNS
    assert session.starts == [0]


def test_pages_are_glued_without_gaps():
    session = FakeSession(total=1234)

    df = moex.fetch_candles("SBER", date(2015, 1, 1), session=session)

    assert len(df) == 1234
    assert session.starts == [0, 500, 1000]
    assert df["begin"].is_unique


def test_exact_multiple_of_page_size():
    session = FakeSession(total=1000)

    df = moex.fetch_candles("SBER", date(2015, 1, 1), session=session)

    assert len(df) == 1000
    assert session.starts == [0, 500, 1000]


def test_no_data_returns_empty_frame():
    df = moex.fetch_candles("NOSUCH", date(2026, 10, 1), session=FakeSession(total=0))

    assert df.empty


def test_stops_if_api_ignores_start(monkeypatch):
    monkeypatch.setattr(moex, "MAX_PAGES", 3)

    class BrokenSession(FakeSession):
        def get(self, url, params, timeout):
            return FakeResponse(make_rows(moex.PAGE_SIZE))

    with pytest.raises(RuntimeError, match="more than 3 pages"):
        moex.fetch_candles("SBER", date(2015, 1, 1), session=BrokenSession(total=0))
