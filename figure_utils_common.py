from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_DYNAMIC = "#E69F00"
COLOR_STATIC = "#56B4E9"
COLOR_PING = "#009E73"
COLOR_NOISE = "#CC79A7"
COLOR_SAMPLE_ALIGNED = "#4C566A"

PUBLICATION_ANNOTATION_FONT_SIZE = 6
PUBLICATION_ERRORBAR_CAPSIZE = 2
PUBLICATION_LINE_WIDTH = 1.6
PUBLICATION_MARKER_SIZE = 4
PUBLICATION_TWO_COLUMN_FIGSIZE = (7.2, 4.8)


def get_paper_color_map() -> dict[str, str]:
    return {
        "dynamic": COLOR_DYNAMIC,
        "static": COLOR_STATIC,
        "ping": COLOR_PING,
        "noise": COLOR_NOISE,
        "sample_aligned": COLOR_SAMPLE_ALIGNED,
    }


def save_figure_all_formats(fig: plt.Figure, stem: str | Path) -> dict[str, str]:
    base = Path(stem)
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf = base.with_suffix(".pdf")
    svg = base.with_suffix(".svg")
    png = base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=450, bbox_inches="tight")
    return {"pdf": str(pdf), "svg": str(svg), "png": str(png)}


def save_run_config(payload: Mapping[str, object], output_dir: str | Path) -> str:
    out = Path(output_dir) / "run_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def save_tidy_csv(df: pd.DataFrame, path: str | Path, sort_by: Iterable[str] | None = None) -> str:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if sort_by:
        existing = [column for column in sort_by if column in out.columns]
        if existing:
            out = out.sort_values(existing, kind="stable")
    out.to_csv(csv_path, index=False)
    return str(csv_path)


def select_representative_trial(df: pd.DataFrame, condition_col: str, correct_col: str, silent_col: str, first_fire_col: str) -> int:
    ordered = df.sort_values([condition_col, correct_col, silent_col, first_fire_col, "trial_id"], ascending=[True, False, True, True, True], kind="stable")
    return int(ordered.iloc[0]["trial_id"])


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


__all__ = [
    "COLOR_DYNAMIC",
    "COLOR_NOISE",
    "COLOR_PING",
    "COLOR_SAMPLE_ALIGNED",
    "COLOR_STATIC",
    "PUBLICATION_ANNOTATION_FONT_SIZE",
    "PUBLICATION_ERRORBAR_CAPSIZE",
    "PUBLICATION_LINE_WIDTH",
    "PUBLICATION_MARKER_SIZE",
    "PUBLICATION_TWO_COLUMN_FIGSIZE",
    "get_paper_color_map",
    "save_figure_all_formats",
    "save_run_config",
    "save_tidy_csv",
    "select_representative_trial",
    "validate_required_columns",
]
