from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.plotting.paper_fig.utils import write_json


CANONICAL_COLUMNS = (
    "figure_id",
    "panel_id",
    "metric",
    "condition",
    "layer",
    "network_id",
    "seed_id",
    "value",
    "unit",
    "source_file",
)


@dataclass(frozen=True)
class AdapterResult:
    """Result emitted by a paper-figure data adapter."""

    panel_data_path: Path
    stats_manifest_path: Path
    source_manifest_path: Path
    source_manifest: dict[str, Any]
    warnings: list[str]


def panel_stem(figure_id: str, panel_id: str) -> str:
    """Return the canonical output stem for a panel."""
    if figure_id.lower() in {"fig1_supp", "fig1_supp_s2", "fig2", "fig2_supp", "fig3", "fig4", "fig4_supp", "fig5", "fig5_supp", "fig6", "fig6_supp"}:
        return f"{figure_id.lower()}{panel_id.upper()}"
    return f"{figure_id.lower()}{panel_id.lower()}"


def panel_output_paths(output_dir: Path, figure_id: str, panel_id: str) -> dict[str, Path]:
    """Return canonical output paths for one panel."""
    stem = panel_stem(figure_id, panel_id)
    return {
        "panel_data": output_dir / "panel_data" / f"{stem}_panel_data.csv",
        "stats": output_dir / "stats" / f"{stem}_stats.json",
        "sources": output_dir / "source_manifests" / f"{stem}_sources.json",
    }


def resolve_repo_path(repo_root: Path, path_value: str | Path | None) -> Path | None:
    """Resolve a spec path relative to the repository root."""
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def first_existing_path(repo_root: Path, candidates: Iterable[str | Path]) -> tuple[Path | None, list[str]]:
    """Return the first existing path from candidates and all checked paths."""
    checked: list[str] = []
    for candidate in candidates:
        path = resolve_repo_path(repo_root, candidate)
        if path is None:
            continue
        checked.append(str(path))
        if path.exists():
            return path, checked
    return None, checked


def write_adapter_outputs(
    output_dir: Path,
    figure_id: str,
    panel_id: str,
    panel_df: pd.DataFrame,
    stats: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    warnings: list[str] | None = None,
) -> AdapterResult:
    """Write canonical CSV, stats JSON, and source manifest JSON."""
    paths = panel_output_paths(output_dir, figure_id, panel_id)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    out_df = panel_df.copy()
    for col in CANONICAL_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = ""
    out_df = out_df[list(CANONICAL_COLUMNS) + [c for c in out_df.columns if c not in CANONICAL_COLUMNS]]
    out_df.to_csv(paths["panel_data"], index=False, encoding="utf-8")

    warning_list = list(warnings or [])
    stats_payload = dict(stats)
    stats_payload.setdefault("warnings", warning_list)
    write_json(stats_payload, paths["stats"])

    manifest_payload = dict(source_manifest)
    manifest_payload.setdefault("warnings", warning_list)
    write_json(manifest_payload, paths["sources"])
    return AdapterResult(
        panel_data_path=paths["panel_data"],
        stats_manifest_path=paths["stats"],
        source_manifest_path=paths["sources"],
        source_manifest=manifest_payload,
        warnings=warning_list,
    )


def missing_adapter_result(
    spec: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    reason: str,
) -> AdapterResult:
    """Create placeholder adapter outputs for missing data."""
    figure_id = str(spec.get("figure_id", "unknown"))
    panel_id = str(spec.get("panel_id", "unknown"))
    df = pd.DataFrame(
        [
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "metric": "missing_source",
                "condition": "missing",
                "layer": "",
                "network_id": "",
                "seed_id": "",
                "value": float("nan"),
                "unit": "",
                "source_file": "",
                "placeholder_reason": reason,
            }
        ]
    )
    stats = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "missing_source",
        "values_used_for_plotting": [],
    }
    source_manifest = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "status": "missing_source",
        "repo_root": str(repo_root),
        "sources": [],
    }
    return write_adapter_outputs(output_dir, figure_id, panel_id, df, stats, source_manifest, [reason])


def summarize_values(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    """Summarize values as mean/SEM/n by group."""
    if df.empty or "value" not in df.columns:
        return []
    summaries: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, part in grouped:
        values = pd.to_numeric(part["value"], errors="coerce").dropna()
        if values.empty:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(
            {
                "mean": float(values.mean()),
                "sem": float(values.sem()) if len(values) > 1 else 0.0,
                "n": int(values.count()),
                "values_used_for_plotting": [float(v) for v in values.tolist()],
            }
        )
        summaries.append(row)
    return summaries
