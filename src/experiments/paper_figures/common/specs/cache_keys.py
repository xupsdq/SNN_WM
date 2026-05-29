from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.paper_figures.common.specs.schemas import SCHEMA_NAME, SCHEMA_VERSION


def build_spec_cache_key(
    *,
    spec_family: str,
    producer_namespace: str,
    network_seed: int,
    dataset_root: str | Path,
    dataset_split: str,
    sampling_config: Mapping[str, Any],
    schema_version: int = SCHEMA_VERSION,
    model_fingerprint: Mapping[str, Any] | None = None,
    parent_spec_digests: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": int(schema_version),
        "spec_family": str(spec_family),
        "producer_namespace": str(producer_namespace),
        "network_seed": int(network_seed),
        "dataset_root": str(Path(dataset_root).resolve()),
        "dataset_split": str(dataset_split),
        "sampling_config": _json_safe(dict(sampling_config)),
        "model_fingerprint": _json_safe(dict(model_fingerprint or {})),
        "parent_spec_digests": _json_safe(dict(parent_spec_digests or {})),
    }


def spec_digest(payload: Mapping[str, Any] | pd.DataFrame) -> str:
    if isinstance(payload, pd.DataFrame):
        return _dataframe_hash(payload)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def table_digest(tables: Mapping[str, pd.DataFrame]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(tables):
        df = tables[name]
        hasher.update(str(name).encode("utf-8"))
        hasher.update(b"\n")
        hasher.update(",".join(str(col) for col in df.columns).encode("utf-8"))
        hasher.update(b"\n")
        hasher.update(_dataframe_hash(df).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def cache_key_digest(cache_key: Mapping[str, Any]) -> str:
    return spec_digest(cache_key)


def rng_namespace_seed(
    network_seed: int,
    spec_family: str,
    producer_namespace: str,
    schema_version: int = SCHEMA_VERSION,
) -> int:
    text = _canonical_json(
        {
            "network_seed": int(network_seed),
            "spec_family": str(spec_family),
            "producer_namespace": str(producer_namespace),
            "schema_version": int(schema_version),
        }
    )
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % (2**32 - 1)


def _dataframe_hash(df: pd.DataFrame) -> str:
    csv_text = df.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "build_spec_cache_key",
    "cache_key_digest",
    "rng_namespace_seed",
    "spec_digest",
    "table_digest",
]
