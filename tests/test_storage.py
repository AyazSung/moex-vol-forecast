from pathlib import Path

import pandas as pd
import pytest

from moex_vol_forecast.storage import write_csv_atomic


def test_writes_csv_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "raw" / "sber.csv"
    df = pd.DataFrame({"close": [1.5, 2.5], "begin": ["2026-10-01", "2026-10-02"]})

    write_csv_atomic(df, path)

    assert path.read_text() == "close;begin\n1.5;2026-10-01\n2.5;2026-10-02\n"
    assert list(path.parent.iterdir()) == [path]


def test_failed_write_keeps_old_file(tmp_path, monkeypatch):
    path = tmp_path / "sber.csv"
    path.write_text("old content\n")

    def broken_to_csv(self, path_or_buf, *args, **kwargs):
        Path(path_or_buf).write_text("half-writ")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", broken_to_csv)

    with pytest.raises(OSError):
        write_csv_atomic(pd.DataFrame({"close": [1.0]}), path)

    assert path.read_text() == "old content\n"
    assert list(tmp_path.iterdir()) == [path]
