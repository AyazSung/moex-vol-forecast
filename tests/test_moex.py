# Тесты загрузки с биржи (moex.py). Запуск: uv run pytest
# Настоящий MOEX здесь не используется: вместо сети — поддельная сессия FakeSession,
# которая отвечает заготовленными данными. Так тесты быстрые, работают без интернета
# и могут проверить редкие случаи (ровно 1000 свечей, сломанный API).

from datetime import date

import pytest

from moex_vol_forecast import moex

# Столбцы, которые отдаёт MOEX.
COLUMNS = ["open", "close", "high", "low", "value", "volume", "begin", "end"]


def make_rows(count, first=0):
    """Fake candles; row number goes into begin/end so every row is unique."""
    # Номер строки записан в даты: так видно, что при склейке страниц
    # ничего не потерялось и не задвоилось.
    return [
        [1.0, 2.0, 3.0, 0.5, 100.0, 10, f"row-{first + i}", f"row-{first + i}"]
        for i in range(count)
    ]


# Поддельный ответ сервера: умеет то же, чем пользуется fetch_candles —
# raise_for_status() и json().
class FakeResponse:
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        pass  # поддельный ответ всегда «успешный»

    def json(self):
        return {"candles": {"columns": COLUMNS, "data": self.rows}}


class FakeSession:
    """Pretends to be MOEX: serves `total` candles page by page."""

    def __init__(self, total):
        self.total = total  # сколько всего свечей «есть на бирже»
        self.starts = []  # запоминаем все запросы, чтобы тест мог их проверить

    def get(self, url, params, timeout):
        # Как настоящий MOEX: по смещению start отдаём следующий кусок, до 500 строк.
        start = params["start"]
        self.starts.append(start)
        count = max(0, min(moex.PAGE_SIZE, self.total - start))
        return FakeResponse(make_rows(count, first=start))


# Фикстура — помощник, который pytest применяет к тестам. autouse=True — ко всем
# тестам этого файла: убираем паузу между страницами, чтобы тесты не ждали.
@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(moex, "PAUSE_SECONDS", 0)


def test_single_page():
    session = FakeSession(total=12)

    df = moex.fetch_candles("SBER", date(2026, 10, 1), session=session)

    # 12 свечей — одна неполная страница — ровно один запрос.
    assert len(df) == 12
    assert list(df.columns) == COLUMNS
    assert session.starts == [0]


def test_pages_are_glued_without_gaps():
    session = FakeSession(total=1234)

    df = moex.fetch_candles("SBER", date(2015, 1, 1), session=session)

    # 1234 свечи — три страницы (500 + 500 + 234), склеены без потерь и повторов.
    assert len(df) == 1234
    assert session.starts == [0, 500, 1000]
    assert df["begin"].is_unique


def test_exact_multiple_of_page_size():
    session = FakeSession(total=1000)

    df = moex.fetch_candles("SBER", date(2015, 1, 1), session=session)

    # Ровно 1000 свечей: две полные страницы, третий запрос приходит пустым
    # и завершает цикл.
    assert len(df) == 1000
    assert session.starts == [0, 500, 1000]


def test_no_data_returns_empty_frame():
    df = moex.fetch_candles("NOSUCH", date(2026, 10, 1), session=FakeSession(total=0))

    # Несуществующий тикер — пустая таблица (ошибку из этого делает cli.py).
    assert df.empty


def test_stops_if_api_ignores_start(monkeypatch):
    # Уменьшаем предохранитель до 3 страниц, чтобы тест не делал 50 запросов.
    monkeypatch.setattr(moex, "MAX_PAGES", 3)

    # «Сломанный MOEX»: игнорирует start и всегда отдаёт полную страницу.
    class BrokenSession(FakeSession):
        def get(self, url, params, timeout):
            return FakeResponse(make_rows(moex.PAGE_SIZE))

    # Ждём ошибку предохранителя, а не бесконечный цикл.
    with pytest.raises(RuntimeError, match="more than 3 pages"):
        moex.fetch_candles("SBER", date(2015, 1, 1), session=BrokenSession(total=0))
