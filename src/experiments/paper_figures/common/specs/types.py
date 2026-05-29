from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SpecProvenance:
    payload: dict[str, Any]


@dataclass(frozen=True)
class SpecArtifact:
    root: Path
    tables: dict[str, pd.DataFrame]
    manifest: pd.DataFrame
    provenance: SpecProvenance
    cache_key: dict[str, Any]
    cache_key_digest: str
    table_digest: str
    artifact_digest: str


@dataclass(frozen=True)
class SpecViewLink:
    root: Path
    payload: dict[str, Any]


__all__ = ["SpecArtifact", "SpecProvenance", "SpecViewLink"]
