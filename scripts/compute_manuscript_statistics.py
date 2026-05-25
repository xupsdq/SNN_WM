from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is expected in torch_env.
    stats = None  # type: ignore[assignment]

from src.plotting.paper_fig.statistics_tasks import MANUSCRIPT_STAT_TASKS


OUTPUT_FIELDS = [
    "status",
    "task_id",
    "task_type",
    "figure_id",
    "panel_id",
    "claim",
    "metric",
    "replicate_unit",
    "n_networks",
    "n_observations",
    "group",
    "condition_a",
    "condition_b",
    "reference",
    "mean",
    "sd",
    "sem",
    "ci95_low",
    "ci95_high",
    "mean_a",
    "mean_b",
    "effect",
    "effect_ci95_low",
    "effect_ci95_high",
    "effect_size",
    "test_name",
    "statistic",
    "p_value",
    "p_value_fdr",
    "source_file",
    "warnings",
]


@dataclass(frozen=True)
class PanelContext:
    figure_id: str
    panel_id: str
    panel_data_path: Path
    stats_path: Path | None
    warnings: list[str]
    n_networks: int
    status: str


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _warnings_from_stats(path: Path | None) -> list[str]:
    if path is None:
        return []
    payload = _read_json(path)
    warnings = payload.get("warnings", [])
    if isinstance(warnings, str):
        return [warnings]
    if isinstance(warnings, list):
        return [str(item) for item in warnings if str(item)]
    return []


def _panel_stats_path(panel_data_path: Path) -> Path | None:
    stats_dir = panel_data_path.parent.parent / "stats"
    stem = panel_data_path.name.replace("_panel_data.csv", "_stats.json")
    candidate = stats_dir / stem
    return candidate if candidate.is_file() else None


def _panel_context(panel_data_path: Path, *, min_networks: int) -> PanelContext:
    df_head = pd.read_csv(panel_data_path, nrows=1)
    figure_id = str(df_head.get("figure_id", pd.Series([panel_data_path.parent.parent.name])).iloc[0])
    panel_id = str(df_head.get("panel_id", pd.Series([panel_data_path.stem])).iloc[0])
    stats_path = _panel_stats_path(panel_data_path)
    warnings = _warnings_from_stats(stats_path)

    df_ids = pd.read_csv(panel_data_path, usecols=lambda col: col in {"seed_id", "network_id"})
    seed_col = _seed_col(df_ids)
    n_networks = int(df_ids[seed_col].nunique()) if seed_col else 0
    single_network = n_networks < min_networks or any("Single-network result" in item for item in warnings)
    status = "descriptive_only" if single_network else "manuscript_ready"
    return PanelContext(
        figure_id=figure_id,
        panel_id=panel_id,
        panel_data_path=panel_data_path,
        stats_path=stats_path,
        warnings=warnings,
        n_networks=n_networks,
        status=status,
    )


def _seed_col(df: pd.DataFrame) -> str | None:
    for col in ("seed_id", "network_id", "network_seed", "seed"):
        if col in df.columns:
            return col
    return None


def _present_cols(df: pd.DataFrame, cols: Sequence[str] | None) -> list[str]:
    return [str(col) for col in (cols or []) if str(col) in df.columns]


def _default_group_cols(df: pd.DataFrame) -> list[str]:
    candidates = ["condition", "layer", "metric", "category", "model", "target_item", "probe_condition"]
    return [col for col in candidates if col in df.columns]


def _to_numeric_series(values: Iterable[Any]) -> pd.Series:
    return pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def _group_label(group_values: Mapping[str, Any]) -> str:
    return ";".join(f"{key}={value}" for key, value in group_values.items() if str(value) != "")


def _split_group_key(key: Any, group_cols: Sequence[str]) -> dict[str, Any]:
    if not group_cols:
        return {}
    if len(group_cols) == 1:
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        return {group_cols[0]: key}
    if not isinstance(key, tuple):
        key = (key,)
    return {col: key[index] for index, col in enumerate(group_cols)}


def _mean_ci(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = float(sd / math.sqrt(n)) if n > 1 else 0.0
    if n > 1 and stats is not None and sem > 0:
        half_width = float(stats.t.ppf(0.975, n - 1) * sem)
    else:
        half_width = 0.0
    return (mean, sd, sem, mean - half_width, mean + half_width)


def _one_sample(values: Sequence[float], reference: float) -> tuple[str, float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3 or stats is None:
        return ("", math.nan, math.nan, math.nan)
    res = stats.ttest_1samp(arr, popmean=float(reference), nan_policy="omit")
    sd = float(np.std(arr, ddof=1))
    effect_size = float((np.mean(arr) - reference) / sd) if sd > 0 else math.nan
    return ("one_sample_t", float(res.statistic), float(res.pvalue), effect_size)


def _paired(values_a: Sequence[float], values_b: Sequence[float]) -> tuple[str, float, float, float]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]
    if a.size < 3 or stats is None:
        return ("", math.nan, math.nan, math.nan)
    diff = a - b
    res = stats.ttest_rel(a, b, nan_policy="omit")
    sd = float(np.std(diff, ddof=1))
    effect_size = float(np.mean(diff) / sd) if sd > 0 else math.nan
    return ("paired_t", float(res.statistic), float(res.pvalue), effect_size)


def _base_row(ctx: PanelContext, task: Mapping[str, Any], task_type: str) -> dict[str, str]:
    return {
        "status": ctx.status,
        "task_id": str(task.get("task_id", "")),
        "task_type": task_type,
        "figure_id": ctx.figure_id,
        "panel_id": ctx.panel_id,
        "claim": str(task.get("claim", "")),
        "replicate_unit": "network_seed",
        "n_networks": str(ctx.n_networks),
        "source_file": str(ctx.panel_data_path),
        "warnings": "; ".join(ctx.warnings),
    }


def _skip_row(ctx: PanelContext, task: Mapping[str, Any], reason: str) -> dict[str, str]:
    row = _base_row(ctx, task, str(task.get("task_type", "unknown")))
    row["status"] = "skipped"
    row["warnings"] = "; ".join([item for item in [row.get("warnings", ""), reason] if item])
    return row


def _seed_aggregate(df: pd.DataFrame, *, seed_col: str, value_col: str, group_cols: Sequence[str]) -> pd.DataFrame:
    use_cols = [seed_col, *group_cols, value_col]
    work = df.loc[:, [col for col in use_cols if col in df.columns]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    if not group_cols:
        return work.groupby(seed_col, dropna=False, as_index=False)[value_col].mean()
    return work.groupby([seed_col, *group_cols], dropna=False, as_index=False)[value_col].mean()


def _describe_rows(ctx: PanelContext, task: Mapping[str, Any], df: pd.DataFrame) -> list[dict[str, str]]:
    value_col = str(task.get("value_col", "value"))
    seed_col = _seed_col(df)
    if seed_col is None or value_col not in df.columns:
        return [_skip_row(ctx, task, f"Missing seed column or value column {value_col}")]
    group_cols = _present_cols(df, task.get("group_cols")) or _default_group_cols(df)
    agg = _seed_aggregate(df, seed_col=seed_col, value_col=value_col, group_cols=group_cols)
    grouped = [((), agg)] if not group_cols else agg.groupby(group_cols, dropna=False)
    rows: list[dict[str, str]] = []
    for key, sub in grouped:
        group_values = _split_group_key(key, group_cols)
        values = _to_numeric_series(sub[value_col])
        mean, sd, sem, low, high = _mean_ci(values.to_list())
        row = _base_row(ctx, task, "describe")
        row.update(
            {
                "metric": str(sub.get("metric", pd.Series([value_col])).iloc[0]) if "metric" in sub.columns else value_col,
                "n_observations": str(int(values.size)),
                "group": _group_label(group_values),
                "mean": _fmt(mean),
                "sd": _fmt(sd),
                "sem": _fmt(sem),
                "ci95_low": _fmt(low),
                "ci95_high": _fmt(high),
            }
        )
        rows.append(row)
    return rows


def _one_sample_rows(ctx: PanelContext, task: Mapping[str, Any], df: pd.DataFrame) -> list[dict[str, str]]:
    value_col = str(task.get("value_col", "value"))
    seed_col = _seed_col(df)
    if seed_col is None or value_col not in df.columns:
        return [_skip_row(ctx, task, f"Missing seed column or value column {value_col}")]
    group_cols = _present_cols(df, task.get("group_cols"))
    reference = float(task.get("reference", 0.0))
    agg = _seed_aggregate(df, seed_col=seed_col, value_col=value_col, group_cols=group_cols)
    grouped = [((), agg)] if not group_cols else agg.groupby(group_cols, dropna=False)
    rows: list[dict[str, str]] = []
    for key, sub in grouped:
        group_values = _split_group_key(key, group_cols)
        values = _to_numeric_series(sub[value_col]).to_list()
        mean, sd, sem, low, high = _mean_ci(values)
        test_name, statistic, p_value, effect_size = _one_sample(values, reference)
        row = _base_row(ctx, task, "one_sample")
        row.update(
            {
                "metric": str(sub.get("metric", pd.Series([value_col])).iloc[0]) if "metric" in sub.columns else value_col,
                "n_observations": str(len(values)),
                "group": _group_label(group_values),
                "reference": _fmt(reference),
                "mean": _fmt(mean),
                "sd": _fmt(sd),
                "sem": _fmt(sem),
                "ci95_low": _fmt(low),
                "ci95_high": _fmt(high),
                "effect": _fmt(mean - reference if math.isfinite(mean) else math.nan),
                "effect_size": _fmt(effect_size),
                "test_name": test_name,
                "statistic": _fmt(statistic),
                "p_value": _fmt(p_value),
            }
        )
        rows.append(row)
    return rows


def _paired_rows(ctx: PanelContext, task: Mapping[str, Any], df: pd.DataFrame) -> list[dict[str, str]]:
    value_col = str(task.get("value_col", "value"))
    factor_col = str(task.get("factor_col", "condition"))
    seed_col = _seed_col(df)
    if seed_col is None or value_col not in df.columns or factor_col not in df.columns:
        return [_skip_row(ctx, task, f"Missing seed/value/factor column for paired task: {factor_col}")]
    group_cols = _present_cols(df, task.get("group_cols"))
    agg = _seed_aggregate(df, seed_col=seed_col, value_col=value_col, group_cols=[*group_cols, factor_col])
    rows: list[dict[str, str]] = []
    pairs = [(str(a), str(b)) for a, b in task.get("pairs", [])]
    available = {str(item) for item in agg[factor_col].dropna().unique()}
    for condition_a, condition_b in pairs:
        if condition_a not in available or condition_b not in available:
            rows.append(_skip_row(ctx, task, f"Missing paired levels {condition_a}/{condition_b}; available={sorted(available)}"))
            continue
        index_cols = [seed_col, *group_cols]
        pivot = agg.pivot_table(index=index_cols, columns=factor_col, values=value_col, aggfunc="mean")
        if condition_a not in pivot.columns or condition_b not in pivot.columns:
            rows.append(_skip_row(ctx, task, f"Cannot pivot paired levels {condition_a}/{condition_b}"))
            continue
        compare = pivot[[condition_a, condition_b]].dropna()
        if group_cols:
            group_level_names: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols
            for key, sub in compare.groupby(level=group_level_names, dropna=False):
                group_values = _split_group_key(key, group_cols)
                rows.append(_paired_summary_row(ctx, task, sub, condition_a, condition_b, group_values))
        else:
            rows.append(_paired_summary_row(ctx, task, compare, condition_a, condition_b, {}))
    return rows


def _paired_summary_row(
    ctx: PanelContext,
    task: Mapping[str, Any],
    compare: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    group_values: Mapping[str, Any],
) -> dict[str, str]:
    values_a = _to_numeric_series(compare[condition_a]).to_list()
    values_b = _to_numeric_series(compare[condition_b]).to_list()
    diff = [a - b for a, b in zip(values_a, values_b)]
    effect, _sd, _sem, low, high = _mean_ci(diff)
    mean_a = float(np.mean(values_a)) if values_a else math.nan
    mean_b = float(np.mean(values_b)) if values_b else math.nan
    test_name, statistic, p_value, effect_size = _paired(values_a, values_b)
    row = _base_row(ctx, task, "paired")
    row.update(
        {
            "metric": str(task.get("metric", "value")),
            "n_observations": str(len(diff)),
            "group": _group_label(group_values),
            "condition_a": condition_a,
            "condition_b": condition_b,
            "mean_a": _fmt(mean_a),
            "mean_b": _fmt(mean_b),
            "effect": _fmt(effect),
            "effect_ci95_low": _fmt(low),
            "effect_ci95_high": _fmt(high),
            "effect_size": _fmt(effect_size),
            "test_name": test_name,
            "statistic": _fmt(statistic),
            "p_value": _fmt(p_value),
        }
    )
    return row


def _curve_auc_rows(ctx: PanelContext, task: Mapping[str, Any], df: pd.DataFrame) -> list[dict[str, str]]:
    value_col = str(task.get("value_col", "value"))
    x_col = str(task.get("x_col", "x_value"))
    factor_col = str(task.get("factor_col", "condition"))
    seed_col = _seed_col(df)
    missing = [col for col in [seed_col, value_col, x_col, factor_col] if col is None or col not in df.columns]
    if missing:
        return [_skip_row(ctx, task, f"Missing curve columns: {missing}")]
    group_cols = _present_cols(df, task.get("group_cols"))
    agg = _seed_aggregate(df, seed_col=seed_col, value_col=value_col, group_cols=[*group_cols, factor_col, x_col])  # type: ignore[arg-type]
    auc_rows: list[dict[str, Any]] = []
    for key, sub in agg.groupby([seed_col, *group_cols, factor_col], dropna=False):  # type: ignore[list-item]
        key_tuple = key if isinstance(key, tuple) else (key,)
        seed_value = key_tuple[0]
        group_values = {col: key_tuple[index + 1] for index, col in enumerate(group_cols)}
        condition = key_tuple[len(group_cols) + 1]
        sorted_sub = sub.sort_values(x_col)
        x = pd.to_numeric(sorted_sub[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sorted_sub[value_col], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() < 2:
            continue
        auc_rows.append({seed_col: seed_value, **group_values, factor_col: condition, "auc": float(np.trapezoid(y[keep], x[keep]))})
    auc_df = pd.DataFrame(auc_rows)
    if auc_df.empty:
        return [_skip_row(ctx, task, "No valid AUC rows")]

    describe_task = dict(task)
    describe_task["value_col"] = "auc"
    describe_task["group_cols"] = [*group_cols, factor_col]
    rows = _describe_rows(ctx, describe_task, auc_df.assign(metric="auc"))

    paired_task = dict(task)
    paired_task["value_col"] = "auc"
    paired_task["metric"] = "auc"
    paired_task["group_cols"] = group_cols
    rows.extend(_paired_rows(ctx, paired_task, auc_df))
    for row in rows:
        row["task_type"] = "curve_auc" if row.get("task_type") in {"describe", "paired"} else row.get("task_type", "curve_auc")
        row["metric"] = row.get("metric") or "auc"
    return rows


def _trend_slope_rows(ctx: PanelContext, task: Mapping[str, Any], df: pd.DataFrame) -> list[dict[str, str]]:
    value_col = str(task.get("value_col", "value"))
    x_col = str(task.get("x_col", "x_value"))
    seed_col = _seed_col(df)
    if seed_col is None or value_col not in df.columns or x_col not in df.columns:
        return [_skip_row(ctx, task, f"Missing trend columns: {x_col}/{value_col}")]
    group_cols = _present_cols(df, task.get("group_cols"))
    agg = _seed_aggregate(df, seed_col=seed_col, value_col=value_col, group_cols=[*group_cols, x_col])
    slope_rows: list[dict[str, Any]] = []
    for key, sub in agg.groupby([seed_col, *group_cols], dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        seed_value = key_tuple[0]
        group_values = {col: key_tuple[index + 1] for index, col in enumerate(group_cols)}
        x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() < 2:
            continue
        slope = float(np.polyfit(x[keep], y[keep], deg=1)[0])
        slope_rows.append({seed_col: seed_value, **group_values, "slope": slope})
    slope_df = pd.DataFrame(slope_rows)
    if slope_df.empty:
        return [_skip_row(ctx, task, "No valid slope rows")]
    slope_task = dict(task)
    slope_task["task_type"] = "one_sample"
    slope_task["value_col"] = "slope"
    slope_task["reference"] = task.get("reference", 0.0)
    slope_input = slope_df.assign(metric="slope")
    rows = _one_sample_rows(ctx, slope_task, slope_input)
    for row in rows:
        row["task_type"] = "trend_slope"
        row["metric"] = "slope"
    return rows


def _run_task(ctx: PanelContext, task: Mapping[str, Any]) -> list[dict[str, str]]:
    df = pd.read_csv(ctx.panel_data_path)
    task_type = str(task.get("task_type", "describe"))
    if task_type == "auto_describe" or task_type == "describe":
        return _describe_rows(ctx, task, df)
    if task_type == "one_sample":
        return _one_sample_rows(ctx, task, df)
    if task_type == "paired":
        return _paired_rows(ctx, task, df)
    if task_type == "curve_auc":
        return _curve_auc_rows(ctx, task, df)
    if task_type == "trend_slope":
        return _trend_slope_rows(ctx, task, df)
    return [_skip_row(ctx, task, f"Unknown task type: {task_type}")]


def _discover_panel_data(root: Path, fig_ids: Sequence[str]) -> dict[tuple[str, str], Path]:
    discovered: dict[tuple[str, str], Path] = {}
    for fig_dir in sorted(root.iterdir()):
        if not fig_dir.is_dir() or (fig_ids and fig_dir.name not in fig_ids):
            continue
        panel_dir = fig_dir / "panel_data"
        if not panel_dir.is_dir():
            continue
        for path in sorted(panel_dir.glob("*_panel_data.csv")):
            head = pd.read_csv(path, nrows=1)
            panel_id = str(head.get("panel_id", pd.Series([path.stem])).iloc[0])
            discovered[(fig_dir.name, panel_id.upper())] = path
    return discovered


def _tasks_for_figures(fig_ids: Sequence[str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for task in MANUSCRIPT_STAT_TASKS:
        fig_id = str(task.get("figure_id", ""))
        if fig_ids and fig_id not in fig_ids:
            continue
        if str(task.get("panel_id", "")) == "*":
            continue
        tasks.append(dict(task))
    return tasks


def _task_panel_path(root: Path, task: Mapping[str, Any]) -> Path:
    return root / str(task["figure_id"]) / "panel_data" / str(task["panel_data"])


def _auto_tasks(discovered: Mapping[tuple[str, str], Path], explicit_paths: set[Path]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for (figure_id, panel_id), path in discovered.items():
        if path.resolve() in explicit_paths:
            continue
        tasks.append(
            {
                "task_id": f"auto_describe_{figure_id}_{panel_id}",
                "task_type": "auto_describe",
                "figure_id": figure_id,
                "panel_id": panel_id,
                "panel_data": path.name,
                "claim": "Automatic descriptive statistics for manuscript-statistics audit coverage.",
            }
        )
    return tasks


def _apply_fdr(rows: list[dict[str, str]]) -> None:
    indexed: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        try:
            p_value = float(row.get("p_value", ""))
        except ValueError:
            continue
        if math.isfinite(p_value):
            indexed.append((index, p_value))
    if not indexed:
        return
    ordered = sorted(indexed, key=lambda item: item[1])
    m = len(ordered)
    adjusted = [1.0] * m
    running = 1.0
    for rank_from_end, (original_index, p_value) in enumerate(reversed(ordered), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        adjusted[rank - 1] = running
    for (original_index, _p_value), p_adj in zip(ordered, adjusted):
        rows[original_index]["p_value_fdr"] = _fmt(min(p_adj, 1.0))


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def _write_json(path: Path, rows: Sequence[Mapping[str, str]], *, root: Path, fig_ids: Sequence[str]) -> None:
    by_status: dict[str, int] = defaultdict(int)
    by_figure: dict[str, int] = defaultdict(int)
    for row in rows:
        by_status[row.get("status", "")] += 1
        by_figure[row.get("figure_id", "")] += 1
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_fig_root": str(root),
        "figures": list(fig_ids),
        "row_count": len(rows),
        "status_counts": dict(sorted(by_status.items())),
        "figure_counts": dict(sorted(by_figure.items())),
        "rows": list(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    status_counts: dict[str, int] = defaultdict(int)
    warnings: list[str] = []
    for row in rows:
        status_counts[row.get("status", "")] += 1
        if row.get("warnings"):
            warnings.append(f"- {row.get('figure_id')}{row.get('panel_id')} {row.get('task_id')}: {row.get('warnings')}")
    lines = [
        "# Manuscript Statistics Report",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Manuscript-Ready Test Rows", ""])
    ready = [row for row in rows if row.get("status") == "manuscript_ready" and row.get("test_name")]
    if not ready:
        lines.append("- None")
    else:
        lines.append("| Figure | Panel | Task | Group | Contrast | Effect | 95% CI | Test | p | FDR p |")
        lines.append("| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |")
        for row in ready:
            contrast = row.get("condition_a") or "mean"
            if row.get("condition_b"):
                contrast = f"{row.get('condition_a')} - {row.get('condition_b')}"
            if row.get("reference"):
                contrast = f"mean - {row.get('reference')}"
            ci = ""
            if row.get("effect_ci95_low") or row.get("effect_ci95_high"):
                ci = f"[{row.get('effect_ci95_low')}, {row.get('effect_ci95_high')}]"
            elif row.get("ci95_low") or row.get("ci95_high"):
                ci = f"[{row.get('ci95_low')}, {row.get('ci95_high')}]"
            lines.append(
                "| {fig} | {panel} | {task} | {group} | {contrast} | {effect} | {ci} | {test} | {p} | {pfdr} |".format(
                    fig=row.get("figure_id", ""),
                    panel=row.get("panel_id", ""),
                    task=row.get("task_id", ""),
                    group=row.get("group", ""),
                    contrast=contrast,
                    effect=row.get("effect", "") or row.get("mean", ""),
                    ci=ci,
                    test=row.get("test_name", ""),
                    p=row.get("p_value", ""),
                    pfdr=row.get("p_value_fdr", ""),
                )
            )
    if warnings:
        lines.extend(["", "## Warnings", "", *warnings[:200]])
        if len(warnings) > 200:
            lines.append(f"- ... {len(warnings) - 200} additional warning rows omitted")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute unified manuscript statistics from paper-figure panel_data bundles.")
    parser.add_argument("--paper-fig-root", default="results/paper_figures/outputs", help="Root containing per-figure output folders.")
    parser.add_argument("--output-dir", default="results/paper_figures/statistics", help="Output directory for CSV/JSON/Markdown statistics.")
    parser.add_argument("--fig", action="append", default=[], help="Figure id to include. Repeatable. Defaults to all discovered figures.")
    parser.add_argument("--min-networks", type=int, default=3, help="Minimum network seeds required for manuscript-ready inferential rows.")
    parser.add_argument("--no-auto-describe", action="store_true", help="Disable automatic descriptive rows for panels without explicit tasks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(args.paper_fig_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    fig_ids = [str(fig).lower() for fig in args.fig]
    if not root.is_dir():
        raise FileNotFoundError(f"Paper-figure output root does not exist: {root}")

    discovered = _discover_panel_data(root, fig_ids)
    tasks = _tasks_for_figures(fig_ids)
    explicit_paths = {_task_panel_path(root, task).resolve() for task in tasks if task.get("panel_data")}
    if not args.no_auto_describe:
        tasks.extend(_auto_tasks(discovered, explicit_paths))

    rows: list[dict[str, str]] = []
    for task in tasks:
        panel_path = _task_panel_path(root, task)
        if not panel_path.is_file():
            dummy_ctx = PanelContext(
                figure_id=str(task.get("figure_id", "")),
                panel_id=str(task.get("panel_id", "")),
                panel_data_path=panel_path,
                stats_path=None,
                warnings=[],
                n_networks=0,
                status="skipped",
            )
            rows.append(_skip_row(dummy_ctx, task, f"Missing panel_data file: {panel_path}"))
            continue
        ctx = _panel_context(panel_path, min_networks=int(args.min_networks))
        rows.extend(_run_task(ctx, task))

    _apply_fdr(rows)
    _write_csv(output_dir / "manuscript_stats_long.csv", rows)
    _write_json(output_dir / "manuscript_stats_summary.json", rows, root=root, fig_ids=fig_ids)
    _write_markdown(output_dir / "manuscript_stats_report.md", rows)
    print(f"Wrote {output_dir / 'manuscript_stats_long.csv'}")
    print(f"Wrote {output_dir / 'manuscript_stats_summary.json'}")
    print(f"Wrote {output_dir / 'manuscript_stats_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
