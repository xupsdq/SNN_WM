from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SharedSequenceSpecBundle:
    root: Path
    sequence_trials: pd.DataFrame
    singleton_reference_trials: pd.DataFrame
    partial_cue_trials: pd.DataFrame
    manifest: pd.DataFrame
    digest: str
    spec_artifact: Any | None = None


@dataclass(frozen=True)
class SharedSequenceRootBank:
    root: Path
    specs: SharedSequenceSpecBundle
    fig3_state_bank_dir: Path
    fig6_sequence_bank_dir: Path
    manifest: pd.DataFrame
    digest: str


__all__ = ["SharedSequenceRootBank", "SharedSequenceSpecBundle"]
