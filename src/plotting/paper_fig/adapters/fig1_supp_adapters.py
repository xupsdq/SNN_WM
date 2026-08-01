from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.plotting.paper_fig.data_resolver import (
    AdapterResult,
    missing_adapter_result,
    panel_output_paths,
    write_adapter_outputs,
)
from src.plotting.paper_fig.adapters.fig1_adapters import (
    LAYER_ORDER,
    _canonical_row,
    _display_path,
    _fraction,
    _manifest,
    _run_mode,
    _run_mode_warnings,
    _seed_dirs,
    _seed_id,
    _source_entry,
    _stats_payload,
    _to_percent,
    _unique,
    _unique_numeric,
)


CONDITION_ORDER = ("dynamic_intact", "spike_state_shuffle", "membrane_state_shuffle", "ux_trial_shuffle", "static_frozen")
PHASE_ORDER = ("stimulus", "sample", "early_delay", "late_delay", "delay", "probe")


def validate_frozen_panel_input(spec: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Validate one hash-pinned S1 panel bundle without recomputing scientific values."""
    frozen = spec.get("persisted_input")
    if not isinstance(frozen, Mapping):
        raise ValueError("persisted_input is required for every active S1 panel")

    figure_id, panel_id = _ids(spec)
    panel_key = str(frozen.get("panel_key", ""))
    if figure_id != "supp_fig_s1" or panel_key != f"S1{panel_id}":
        raise ValueError(f"Frozen identity mismatch: figure={figure_id}, panel={panel_id}, panel_key={panel_key}")

    resolved: dict[str, Path] = {}
    for role in ("panel_data", "stats", "source_manifest", "parent_hash_manifest"):
        entry = frozen.get(role)
        if not isinstance(entry, Mapping):
            raise ValueError(f"persisted_input.{role} must be a mapping")
        path = _frozen_repo_path(repo_root, entry.get("path"))
        expected_sha = str(entry.get("sha256", "")).lower()
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen {role}: {path}")
        observed_sha = _sha256_file(path)
        if observed_sha != expected_sha:
            raise ValueError(f"Frozen {role} hash mismatch: expected {expected_sha}, observed {observed_sha}")
        resolved[role] = path

    panel_df = pd.read_csv(resolved["panel_data"])
    panel_entry = frozen["panel_data"]
    expected_columns = [str(value) for value in panel_entry.get("columns", [])]
    if list(panel_df.columns) != expected_columns:
        raise ValueError(f"Frozen panel_data columns changed for {panel_key}")
    if len(panel_df) != int(panel_entry.get("row_count", -1)):
        raise ValueError(f"Frozen panel_data row count changed for {panel_key}")
    if panel_df["figure_id"].astype(str).drop_duplicates().tolist() != [figure_id]:
        raise ValueError(f"Frozen figure identity changed for {panel_key}")
    if panel_df["panel_id"].astype(str).drop_duplicates().tolist() != [panel_id]:
        raise ValueError(f"Frozen panel identity changed for {panel_key}")
    if panel_df["metric"].astype(str).drop_duplicates().tolist() != [str(panel_entry.get("metric"))]:
        raise ValueError(f"Frozen metric identity changed for {panel_key}")
    observed_networks = panel_df["network_id"].dropna().astype(str).drop_duplicates().tolist()
    if len(observed_networks) != int(panel_entry.get("network_count", -1)):
        raise ValueError(f"Frozen network identity count changed for {panel_key}")

    stats = _read_json_object(resolved["stats"])
    if str(stats.get("figure_id")) != figure_id or str(stats.get("panel_id")) != panel_id:
        raise ValueError(f"Frozen stats identity changed for {panel_key}")
    summaries = stats.get("summaries")
    if not isinstance(summaries, list) or len(summaries) != int(frozen["stats"].get("summary_count", -1)):
        raise ValueError(f"Frozen summary count changed for {panel_key}")

    source_manifest = _read_json_object(resolved["source_manifest"])
    if str(source_manifest.get("status")) != "ok":
        raise ValueError(f"Frozen source manifest is not ok for {panel_key}")
    if str(source_manifest.get("figure_id")) != figure_id or str(source_manifest.get("panel_id")) != panel_id:
        raise ValueError(f"Frozen source-manifest identity changed for {panel_key}")

    parent_rows = _read_json_array(resolved["parent_hash_manifest"])
    parent_index = {
        (str(row.get("panel_key")), str(row.get("source_path"))): row
        for row in parent_rows
        if isinstance(row, Mapping)
    }
    source_paths = sorted(set(_manifest_source_paths(source_manifest.get("sources", []))))
    expected_source_count = int(frozen["source_manifest"].get("source_path_count", -1))
    if len(source_paths) != expected_source_count:
        raise ValueError(
            f"Frozen source path count changed for {panel_key}: expected {expected_source_count}, observed {len(source_paths)}"
        )
    for source_path in source_paths:
        parent = parent_index.get((panel_key, source_path))
        if parent is None:
            raise ValueError(f"No frozen parent-hash row for {panel_key}: {source_path}")
        source = _frozen_repo_path(repo_root, source_path)
        if not source.is_file() or not bool(parent.get("exists")):
            raise FileNotFoundError(f"Frozen parent source is missing for {panel_key}: {source_path}")
        if source.stat().st_size != int(parent.get("size_bytes", -1)):
            raise ValueError(f"Frozen parent size changed for {panel_key}: {source_path}")
        expected_parent_sha = str(parent.get("recorded_sha256", "")).lower()
        if _sha256_file(source) != expected_parent_sha:
            raise ValueError(f"Frozen parent hash changed for {panel_key}: {source_path}")

    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "panel_key": panel_key,
        "paths": {role: str(path) for role, path in resolved.items()},
        "row_count": int(len(panel_df)),
        "network_count": int(len(observed_networks)),
        "summary_count": int(len(summaries)),
        "source_path_count": int(len(source_paths)),
        "hashes_verified": 4 + len(source_paths),
        "status": "pass",
    }


def _load_frozen_panel_output(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    """Copy a validated persisted panel bundle byte-for-byte into the plot-only output root."""
    validation = validate_frozen_panel_input(spec, repo_root)
    figure_id, panel_id = _ids(spec)
    frozen = spec["persisted_input"]
    source_paths = {
        role: _frozen_repo_path(repo_root, frozen[role]["path"])
        for role in ("panel_data", "stats", "source_manifest")
    }
    destination_paths = panel_output_paths(output_dir, figure_id, panel_id)
    for destination in destination_paths.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_paths["panel_data"], destination_paths["panel_data"])
    shutil.copyfile(source_paths["stats"], destination_paths["stats"])
    shutil.copyfile(source_paths["source_manifest"], destination_paths["sources"])
    source_manifest = _read_json_object(destination_paths["sources"])
    warnings = [str(value) for value in source_manifest.get("warnings", [])]
    source_manifest["f1_frozen_validation"] = validation
    return AdapterResult(
        panel_data_path=destination_paths["panel_data"],
        stats_manifest_path=destination_paths["stats"],
        source_manifest_path=destination_paths["sources"],
        source_manifest=source_manifest,
        warnings=warnings,
    )


def _frozen_repo_path(repo_root: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Frozen input escapes repository root: {raw_path}") from exc
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_json_array(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array: {path}")
    return payload


def _manifest_source_paths(payload: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(payload, Mapping):
        raw_path = payload.get("path")
        if raw_path not in (None, "") and "exists" in payload:
            if not bool(payload.get("exists")):
                raise FileNotFoundError(f"Frozen source manifest contains a missing path: {raw_path}")
            paths.append(str(raw_path).replace("\\", "/"))
        for value in payload.values():
            paths.extend(_manifest_source_paths(value))
    elif isinstance(payload, list):
        for value in payload:
            paths.extend(_manifest_source_paths(value))
    return paths


def build_s1_class_recall_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 supplement experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "supp_class_recall_by_digit.csv"
        if not path.exists():
            warnings.append(f"Missing class recall source: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        class_col = _first_col(df, ["class", "digit", "label", "true_label"])
        recall_col = _first_col(df, ["recall", "class_recall", "overall_recall", "acc"])
        if class_col is None or recall_col is None:
            warnings.append(f"Class recall source missing class/recall columns: {_display_path(path, repo_root)}")
            continue
        for _, row in df.iterrows():
            digit = row.get(class_col)
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="class_recall",
                    condition=str(digit),
                    value=_to_percent(row.get(recall_col, 0.0)),
                    unit="percent",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    digit_class=str(digit),
                    n_trials=row.get("n_trials", ""),
                    n_correct=row.get("n_correct", ""),
                    run_mode=_run_mode(seeds),
                )
            )
    panel_df = pd.DataFrame(rows)
    if not panel_df.empty:
        panel_df["_digit_order"] = pd.to_numeric(panel_df["digit_class"], errors="coerce").fillna(99)
        panel_df = panel_df.sort_values(["_digit_order", "seed_id"], kind="stable").drop(columns=["_digit_order"]).reset_index(drop=True)
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="class_recall", run_mode=_run_mode(seeds), group_cols=["digit_class"])
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_s1_confusion_matrix_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 supplement experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    labels: set[str] = set()
    total_count = 0.0
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "supp_confusion_matrix_long.csv"
        if not path.exists():
            warnings.append(f"Missing confusion matrix source: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        true_col = _first_col(df, ["true_label", "label", "class"])
        pred_col = _first_col(df, ["pred_label", "prediction", "predicted_label"])
        count_col = _first_col(df, ["count", "n", "value"])
        rate_col = _first_col(df, ["rate", "proportion", "percent"])
        if true_col is None or pred_col is None or (count_col is None and rate_col is None):
            warnings.append(f"Confusion matrix source missing required columns: {_display_path(path, repo_root)}")
            continue
        work = df.copy()
        work["_true_label"] = work[true_col].map(_format_label)
        work["_pred_label"] = work[pred_col].map(_format_label)
        if count_col is not None:
            work["_raw_count"] = pd.to_numeric(work[count_col], errors="coerce").fillna(0.0)
            work["_row_total"] = work.groupby(["network_seed", "_true_label"], dropna=False)["_raw_count"].transform("sum") if "network_seed" in work.columns else work.groupby("_true_label", dropna=False)["_raw_count"].transform("sum")
            work["_value"] = work["_raw_count"] / work["_row_total"].replace(0, pd.NA) * 100.0
            total_count += float(work["_raw_count"].sum())
        else:
            work["_raw_count"] = ""
            work["_value"] = pd.to_numeric(work[rate_col], errors="coerce").map(_to_percent)
        for _, row in work.iterrows():
            true_label = str(row["_true_label"])
            pred_label = str(row["_pred_label"])
            labels.update([true_label, pred_label])
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="confusion_matrix_row_percent",
                    condition=true_label,
                    value=float(row["_value"]) if pd.notna(row["_value"]) else float("nan"),
                    unit="percent",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    true_label=true_label,
                    pred_label=pred_label,
                    raw_count=row.get("_raw_count", ""),
                    run_mode=_run_mode(seeds),
                )
            )
    panel_df = pd.DataFrame(rows)
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="confusion_matrix_row_percent", run_mode=_run_mode(seeds), group_cols=["true_label", "pred_label"] if not panel_df.empty else [])
    stats["matrix_shape"] = [len(labels), len(labels)] if labels else [0, 0]
    stats["labels"] = _sorted_labels(labels)
    stats["total_count"] = total_count
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    manifest["labels"] = stats["labels"]
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_s2_phase_firing_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_frozen_panel_output(spec, repo_root, output_dir)


def build_s2_delay_decode_curve_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_frozen_panel_output(spec, repo_root, output_dir)


def build_s2_dms_delay_accuracy_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 supplement experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    conditions = [str(v) for v in spec.get("conditions", ["dynamic_intact", "static_frozen"])]
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "supp_dms_delay_sweep_metrics.csv"
        if not path.exists():
            warnings.append(f"Missing DMS delay sweep source: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        for _, row in df[df["condition"].astype(str).isin(conditions)].iterrows():
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="probe_accuracy",
                    condition=str(row.get("condition", "")),
                    value=_to_percent(row.get("acc_probe", 0.0)),
                    unit="percent",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    delay_ms=row.get("delay_ms", ""),
                    sample_attribution_rate=_to_percent(row.get("sample_attribution_rate", 0.0)),
                    silent_rate=_to_percent(row.get("silent_rate", 0.0)),
                    n_trials=row.get("n_trials", ""),
                    run_mode=_run_mode(seeds),
                )
            )
    if not rows:
        return missing_adapter_result(spec, repo_root, output_dir, "Missing supp_dms_delay_sweep_metrics.csv for Fig.1 supplement S2C.")
    panel_df = _sort_delay_condition(pd.DataFrame(rows), conditions)
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="probe_accuracy", run_mode=_run_mode(seeds), group_cols=["condition", "delay_ms"])
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok")
    manifest["conditions"] = _unique(panel_df, "condition")
    manifest["delay_ms_values"] = _unique_numeric(panel_df, "delay_ms")
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_s2_dms_delay_contrast_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_frozen_panel_output(spec, repo_root, output_dir)


def build_s2_substrate_specificity_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _load_frozen_panel_output(spec, repo_root, output_dir)


def _build_delay_decode_curve(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 supplement experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    for seed_dir in seeds:
        metrics_dir = seed_dir / "data" / "metrics"
        path = _first_existing(metrics_dir, ["supp_delay_decode_curve.csv", "panel_c_delay_decode_metrics.csv"])
        if path is None:
            warnings.append(f"Missing delay decode curve source under {_display_path(seed_dir, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        for _, row in df.iterrows():
            layer = str(row.get("layer", ""))
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="delay_decode_accuracy",
                    condition=layer,
                    layer=layer,
                    value=_to_percent(row.get("acc", 0.0)),
                    unit="percent",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    delay_ms=row.get("delay_ms", ""),
                    feature_type=row.get("feature_type", ""),
                    classifier=row.get("classifier", ""),
                    chance=_to_percent(row.get("chance", 0.1)),
                    n_train=row.get("n_train", ""),
                    n_test=row.get("n_test", ""),
                    run_mode=_run_mode(seeds),
                )
            )
    if not rows:
        return missing_adapter_result(spec, repo_root, output_dir, "Missing delay decode curve source for Fig.1 supplement S2B.")
    panel_df = _sort_layer_delay(pd.DataFrame(rows))
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="delay_decode_accuracy", run_mode=_run_mode(seeds), group_cols=["layer", "delay_ms"])
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok")
    manifest["layers"] = _unique(panel_df, "layer")
    manifest["delay_ms_values"] = _unique_numeric(panel_df, "delay_ms")
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def _ids(spec: Mapping[str, Any]) -> tuple[str, str]:
    return str(spec.get("figure_id", "fig1_supp")), str(spec.get("panel_id", "")).upper()


def _first_col(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    lower = {str(col).lower(): str(col) for col in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _format_label(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric) < 0:
        return "silent"
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    text = str(value)
    return "silent" if text.strip().lower() in {"-1", "silent", "no_response", "none"} else text


def _sorted_labels(labels: set[str]) -> list[str]:
    def key(label: str) -> tuple[int, Any]:
        if label == "silent":
            return (1, 99)
        numeric = pd.to_numeric(pd.Series([label]), errors="coerce").iloc[0]
        return (0, int(numeric)) if pd.notna(numeric) else (1, label)

    return sorted(labels, key=key)


def _first_existing(base: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def _source_entry_with_raw_rows(path: Path, repo_root: Path, raw_rows: int, *, role: str) -> dict[str, Any]:
    entry = _source_entry(path, repo_root)
    entry["raw_rows_read"] = int(raw_rows)
    entry["role"] = role
    return entry


def _sort_layer_phase(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_layer_order"] = df["layer"].map({name: i for i, name in enumerate(LAYER_ORDER)}).fillna(99)
    df["_phase_order"] = df["phase"].map({name: i for i, name in enumerate(PHASE_ORDER)}).fillna(99)
    return df.sort_values(["_layer_order", "_phase_order", "seed_id"], kind="stable").drop(columns=["_layer_order", "_phase_order"]).reset_index(drop=True)


def _sort_layer_delay(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_layer_order"] = df["layer"].map({name: i for i, name in enumerate(LAYER_ORDER)}).fillna(99)
    df["_delay"] = pd.to_numeric(df["delay_ms"], errors="coerce")
    return df.sort_values(["_layer_order", "_delay", "seed_id"], kind="stable").drop(columns=["_layer_order", "_delay"]).reset_index(drop=True)


def _sort_delay_condition(df: pd.DataFrame, conditions: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_condition_order"] = df["condition"].map({name: i for i, name in enumerate(conditions)}).fillna(99)
    df["_delay"] = pd.to_numeric(df["delay_ms"], errors="coerce")
    return df.sort_values(["_condition_order", "_delay", "seed_id"], kind="stable").drop(columns=["_condition_order", "_delay"]).reset_index(drop=True)


def _sort_conditions(df: pd.DataFrame, conditions: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_condition_order"] = df["condition"].map({name: i for i, name in enumerate(conditions)}).fillna(99)
    return df.sort_values(["_condition_order", "seed_id"], kind="stable").drop(columns=["_condition_order"]).reset_index(drop=True)
