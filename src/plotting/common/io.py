import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.plotting.common.colors import (
    get_paper_color_map,
    get_plot_cmap,
    get_plot_color,
    infer_plot_cmap_kind,
    resolve_plot_color,
)
from src.plotting.common.style import (
    ERRORBAR_CAPSIZE,
    LINE_WIDTH,
    MARKER_SIZE,
    apply_paper_style,
)

COLOR_DYNAMIC = get_plot_color("dynamic")
COLOR_STATIC = get_plot_color("static_frozen")
COLOR_DONOR_SHIFT = get_plot_color("donor_trace")
COLOR_DISTRACTOR = get_plot_color("trial_shuffled_ux")
COLOR_PING = get_plot_color("probe_only_region")
COLOR_SAMPLE_ALIGNED = get_plot_color("original_sample_trace")
COLOR_NOISE = get_plot_color("other_residual")
COLOR_NONOVERLAP = get_plot_color("non_overlap_control")
COLOR_EARLIER_ITEM = get_plot_color("first_item_reference")
COLOR_BACKGROUND_SHADE = "#F2F2F2"
COLOR_SAMPLE_WINDOW = "#FFF2B2"
COLOR_PROBE_WINDOW = "#DDEEFF"
COLOR_PING_WINDOW = "#E8DDF5"

_NON_SCALAR_SEQUENCE_TYPES = (list, tuple, set, frozenset, np.ndarray, pd.Series, pd.Index)

__all__ = [
    "COLOR_DYNAMIC",
    "COLOR_STATIC",
    "COLOR_DONOR_SHIFT",
    "COLOR_DISTRACTOR",
    "COLOR_PING",
    "COLOR_SAMPLE_ALIGNED",
    "COLOR_NOISE",
    "COLOR_NONOVERLAP",
    "COLOR_EARLIER_ITEM",
    "COLOR_BACKGROUND_SHADE",
    "COLOR_SAMPLE_WINDOW",
    "COLOR_PROBE_WINDOW",
    "COLOR_PING_WINDOW",
    "get_plot_color",
    "resolve_plot_color",
    "get_plot_cmap",
    "infer_plot_cmap_kind",
    "PUBLICATION_FIGURE_TITLE_FONT_SIZE",
    "PUBLICATION_TITLE_FONT_SIZE",
    "PUBLICATION_AXIS_LABEL_FONT_SIZE",
    "PUBLICATION_TICK_LABEL_FONT_SIZE",
    "PUBLICATION_LEGEND_FONT_SIZE",
    "PUBLICATION_ANNOTATION_FONT_SIZE",
    "PUBLICATION_LINE_WIDTH",
    "PUBLICATION_MARKER_SIZE",
    "PUBLICATION_ERRORBAR_CAPSIZE",
    "PUBLICATION_SINGLE_COLUMN_FIGSIZE",
    "PUBLICATION_TWO_COLUMN_FIGSIZE",
    "apply_publication_style",
    "apply_paper_theme",
    "get_paper_color_map",
    "save_figure_all_formats",
    "save_run_config",
    "save_tidy_csv",
    "select_representative_trial",
    "validate_required_columns",
]

PUBLICATION_FIGURE_TITLE_FONT_SIZE = 14
PUBLICATION_TITLE_FONT_SIZE = 14
PUBLICATION_AXIS_LABEL_FONT_SIZE = 13
PUBLICATION_TICK_LABEL_FONT_SIZE = 12
PUBLICATION_LEGEND_FONT_SIZE = 11
PUBLICATION_ANNOTATION_FONT_SIZE = 11
PUBLICATION_LINE_WIDTH = LINE_WIDTH
PUBLICATION_MARKER_SIZE = MARKER_SIZE
PUBLICATION_ERRORBAR_CAPSIZE = ERRORBAR_CAPSIZE
PUBLICATION_SINGLE_COLUMN_FIGSIZE = (6.0, 5.0)
PUBLICATION_TWO_COLUMN_FIGSIZE = (12.0, 5.0)


def apply_publication_style() -> None:
    apply_paper_style()
    mpl.rcParams["errorbar.capsize"] = PUBLICATION_ERRORBAR_CAPSIZE


def apply_paper_theme() -> None:
    apply_paper_style()


def prepare_figure_for_publication(fig: Figure) -> None:
    fig.set_dpi(300)
    for ax in fig.axes:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=PUBLICATION_TICK_LABEL_FONT_SIZE,
            width=1.5,
        )
        ax.xaxis.label.set_size(PUBLICATION_AXIS_LABEL_FONT_SIZE)
        ax.yaxis.label.set_size(PUBLICATION_AXIS_LABEL_FONT_SIZE)
        ax.title.set_fontsize(PUBLICATION_TITLE_FONT_SIZE)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for text in legend.get_texts():
                text.set_fontsize(PUBLICATION_LEGEND_FONT_SIZE)
            legend_title = legend.get_title()
            if legend_title is not None:
                legend_title.set_fontsize(PUBLICATION_LEGEND_FONT_SIZE)

    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)


def save_figure_all_formats(fig: Figure, base_path: Union[str, Path]) -> Dict[str, str]:
    base = _normalize_base_path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    prepare_figure_for_publication(fig)

    out_paths: Dict[str, str] = {}
    for ext in ("png", "pdf", "svg"):
        out_path = base.with_suffix(f".{ext}")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=300)
        out_paths[ext] = str(out_path)
    return out_paths


def save_run_config(config_dict: Mapping[str, Any], save_dir: Union[str, Path]) -> str:
    if not isinstance(config_dict, Mapping):
        raise TypeError("config_dict must be a mapping")

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    save_path = save_dir_path / "run_config.json"

    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_json_safe(config_dict), handle, ensure_ascii=False, indent=2, sort_keys=True)
    return str(save_path)


def save_tidy_csv(
    df: pd.DataFrame,
    save_path: Union[str, Path],
    sort_by: Optional[Union[str, Sequence[str]]] = None,
) -> str:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    out_df = df.copy()
    if sort_by is not None:
        sort_columns = [sort_by] if isinstance(sort_by, str) else list(sort_by)
        if not sort_columns:
            raise ValueError("sort_by must not be empty")
        validate_required_columns(out_df, sort_columns)
        out_df = out_df.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)

    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(save_path_obj, index=False, encoding="utf-8")
    return str(save_path_obj)


def validate_required_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required = list(required_cols)
    if not required:
        raise ValueError("required_cols must not be empty")

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def select_representative_trial(
    df_trials: pd.DataFrame,
    condition_col: str,
    correct_col: str,
    silent_col: str,
    first_fire_col: str,
    pair_id_col: Optional[str] = None,
    extra_filters: Optional[Mapping[str, Any]] = None,
) -> int:
    required_columns = ["trial_id", condition_col, correct_col, silent_col, first_fire_col]
    if pair_id_col is not None:
        required_columns.append(pair_id_col)
    validate_required_columns(df_trials, required_columns)

    if extra_filters is not None and not isinstance(extra_filters, Mapping):
        raise TypeError("extra_filters must be a mapping from column name to scalar or sequence")

    df_filtered = df_trials.copy()
    if extra_filters:
        for column, raw_value in extra_filters.items():
            validate_required_columns(df_filtered, [column])
            if _is_multi_value_filter(raw_value):
                values = list(raw_value)
                if not values:
                    raise ValueError(f"extra_filters[{column!r}] must not be empty")
                df_filtered = df_filtered[df_filtered[column].isin(values)]
            else:
                df_filtered = df_filtered[df_filtered[column] == raw_value]

    if df_filtered.empty:
        raise ValueError("No trials remain after applying extra_filters")

    unique_conditions = pd.unique(df_filtered[condition_col])
    if len(unique_conditions) != 1:
        raise ValueError(
            f"select_representative_trial expects a single condition after filtering; got {len(unique_conditions)}"
        )

    condition_df = df_filtered.copy()
    eligible_df = condition_df[(condition_df[correct_col] == 1) & (condition_df[silent_col] == 0)].copy()
    if eligible_df.empty:
        raise ValueError("No eligible trials remain after enforcing correct == 1 and silent == 0")

    selection_df = _prefer_paired_trials(eligible_df, pair_id_col=pair_id_col)

    first_fire_numeric = pd.to_numeric(selection_df[first_fire_col], errors="coerce")
    if first_fire_numeric.isna().any():
        raise ValueError(f"{first_fire_col} must be numeric for representative trial selection")
    selection_df = selection_df.copy()
    selection_df["_first_fire_numeric"] = first_fire_numeric

    median_first_fire = float(selection_df["_first_fire_numeric"].median())
    selection_df["_median_distance"] = (selection_df["_first_fire_numeric"] - median_first_fire).abs()

    min_distance = float(selection_df["_median_distance"].min())
    tied_df = selection_df[selection_df["_median_distance"] == min_distance].copy()

    tied_df["_label_frequency_score"] = _compute_label_frequency_scores(tied_df, reference_df=condition_df)
    tied_df["_trial_id_numeric"] = pd.to_numeric(tied_df["trial_id"], errors="coerce")
    if tied_df["_trial_id_numeric"].isna().any():
        raise ValueError("trial_id must be numeric")

    chosen_row = tied_df.sort_values(
        by=["_label_frequency_score", "_trial_id_numeric"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    return int(chosen_row["trial_id"])


def _normalize_base_path(base_path: Union[str, Path]) -> Path:
    path = Path(base_path)
    if path.suffix.lower() in {".png", ".pdf", ".svg"}:
        path = path.with_suffix("")
    return path


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_to_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def _is_multi_value_filter(value: Any) -> bool:
    return isinstance(value, _NON_SCALAR_SEQUENCE_TYPES) and not isinstance(value, (str, bytes))


def _prefer_paired_trials(df: pd.DataFrame, pair_id_col: Optional[str]) -> pd.DataFrame:
    if pair_id_col is None:
        return df.copy()

    pair_ids = df[pair_id_col]
    valid_mask = pair_ids.notna()
    if not valid_mask.any():
        return df.copy()

    valid_pair_ids = pair_ids[valid_mask].value_counts()
    paired_ids = valid_pair_ids[valid_pair_ids >= 2].index
    if len(paired_ids) == 0:
        return df.copy()

    paired_df = df[df[pair_id_col].isin(paired_ids)].copy()
    return paired_df if not paired_df.empty else df.copy()


def _compute_label_frequency_scores(selection_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.Series:
    if {"sample_label", "probe_label"}.issubset(reference_df.columns):
        ref_key = _make_joint_label_key(reference_df, "sample_label", "probe_label")
        sel_key = _make_joint_label_key(selection_df, "sample_label", "probe_label")
        counts = ref_key.value_counts(dropna=False)
        return sel_key.map(counts).fillna(0).astype(int)

    if "sample_label" in reference_df.columns:
        counts = reference_df["sample_label"].value_counts(dropna=False)
        return selection_df["sample_label"].map(counts).fillna(0).astype(int)

    if "probe_label" in reference_df.columns:
        counts = reference_df["probe_label"].value_counts(dropna=False)
        return selection_df["probe_label"].map(counts).fillna(0).astype(int)

    return pd.Series(np.ones(len(selection_df), dtype=np.int64), index=selection_df.index)


def _make_joint_label_key(df: pd.DataFrame, left_col: str, right_col: str) -> pd.Series:
    left = df[left_col].astype("string").fillna("<NA>")
    right = df[right_col].astype("string").fillna("<NA>")
    return left + "||" + right
