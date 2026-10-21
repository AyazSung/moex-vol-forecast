# Тесты атомарной записи и дозаписи истории (storage.py). Запуск: uv run pytest
# tmp_path — временная папка pytest: тесты пишут файлы туда, а не в твою data/.

from pathlib import Path

import pandas as pd
import pytest

from moex_vol_forecast.storage import merge_candles, read_candles, write_csv_atomic


def test_writes_csv_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "raw" / "sber.csv"
    df = pd.DataFrame({"close": [1.5, 2.5], "begin": ["2026-10-01", "2026-10-02"]})

    write_csv_atomic(df, path)

    # Файл записан ровно в нужном формате (разделитель ";")...
    assert path.read_text() == "close;begin\n1.5;2026-10-01\n2.5;2026-10-02\n"
    # ...и в папке нет ничего, кроме него: временный .tmp убран.
    assert list(path.parent.iterdir()) == [path]


def test_failed_write_keeps_old_file(tmp_path, monkeypatch):
    # Уже есть старый файл с данными.
    path = tmp_path / "sber.csv"
    path.write_text("old content\n")

    # Имитируем сбой посреди записи: «сломанный» to_csv успевает записать
    # во временный файл кусок данных и падает, как при переполненном диске.
    def broken_to_csv(self, path_or_buf, *args, **kwargs):
        Path(path_or_buf).write_text("half-writ")
        raise OSError("disk full")

    # На время теста подменяем настоящий DataFrame.to_csv сломанным.
    monkeypatch.setattr(pd.DataFrame, "to_csv", broken_to_csv)

    with pytest.raises(OSError):
        write_csv_atomic(pd.DataFrame({"close": [1.0]}), path)

    # Старый файл не тронут (атомарность), недописанного .tmp нет (сработал finally).
    assert path.read_text() == "old content\n"
    assert list(tmp_path.iterdir()) == [path]


def test_merge_keeps_history_and_updates_overlap():
    # В файле — три дня. Загрузка за последние дни принесла 2-е и 3-е октября
    # (3-е — с уточнённой ценой: вчера свеча была незакрытой) и новое 4-е.
    old = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0], "begin": ["2026-10-01", "2026-10-02", "2026-10-03"]}
    )
    new = pd.DataFrame(
        {"close": [2.0, 3.5, 4.0], "begin": ["2026-10-02", "2026-10-03", "2026-10-04"]}
    )

    merged = merge_candles(old, new)

    # История с 1-го октября на месте, 3-е обновилось, 4-е добавилось, дублей нет.
    assert merged["begin"].tolist() == [
        "2026-10-01",
        "2026-10-02",
        "2026-10-03",
        "2026-10-04",
    ]
    assert merged["close"].tolist() == [1.0, 2.0, 3.5, 4.0]


def test_read_returns_numbers_exactly_as_written(tmp_path):
    # Настоящий оборот SBER за 28.11.2018. Обычный разбор CSV в pandas прочитал бы
    # его как 11541818590.300003 — и при каждой перезаписи файла значение менялось бы.
    path = tmp_path / "sber.csv"
    df = pd.DataFrame({"value": [11541818590.300005], "begin": ["2018-11-28"]})

    write_csv_atomic(df, path)

    assert read_candles(path)["value"].tolist() == [11541818590.300005]
