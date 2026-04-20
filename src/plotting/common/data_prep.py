from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cache import load_csv_cached, load_json_cached, load_npy_cached


def load_table(path: str | Path) -> pd.DataFrame:
    return load_csv_cached(path)


def load_json_object(path: str | Path):
    return load_json_cached(path)


def load_numpy_array(path: str | Path):
    return load_npy_cached(path)

