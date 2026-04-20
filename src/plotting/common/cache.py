from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_CSV_CACHE: dict[str, pd.DataFrame] = {}
_JSON_CACHE: dict[str, object] = {}
_NPY_CACHE: dict[str, np.ndarray] = {}


def load_csv_cached(path: str | Path) -> pd.DataFrame:
    key = str(Path(path).resolve())
    if key not in _CSV_CACHE:
        _CSV_CACHE[key] = pd.read_csv(key)
    return _CSV_CACHE[key].copy()


def load_json_cached(path: str | Path):
    key = str(Path(path).resolve())
    if key not in _JSON_CACHE:
        with Path(key).open("r", encoding="utf-8") as handle:
            _JSON_CACHE[key] = json.load(handle)
    return _JSON_CACHE[key]


def load_npy_cached(path: str | Path) -> np.ndarray:
    key = str(Path(path).resolve())
    if key not in _NPY_CACHE:
        _NPY_CACHE[key] = np.load(key, allow_pickle=False)
    return np.array(_NPY_CACHE[key], copy=True)

