"""Atomic file writing: readers never see a half-written file."""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write df to a temporary file next to path, then swap it in one step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        df.to_csv(tmp_path, sep=";", index=False)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info("saved %d rows to %s", len(df), path)