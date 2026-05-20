from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.plotting.paper_fig.data_resolver import AdapterResult, missing_adapter_result, write_adapter_outputs
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
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 supplement experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    for seed_dir in seeds:
        path = seed_dir / "data" / "metrics" / "supp_phase_firing_rates.csv"
        if not path.exists():
            warnings.append(f"Missing phase firing source: {_display_path(path, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        if not {"layer", "phase", "spike_rate_hz"}.issubset(df.columns):
            warnings.append(f"Phase firing source missing layer/phase/spike_rate_hz columns: {_display_path(path, repo_root)}")
            continue
        grouped = df.groupby(["network_seed", "layer", "phase"], dropna=False)["spike_rate_hz"].mean().reset_index()
        for _, row in grouped.iterrows():
            rows.append(
                _canonical_row(
                    figure_id,
                    panel_id,
                    metric="spike_rate_hz",
                    condition=str(row["phase"]),
                    layer=str(row["layer"]),
                    value=float(row["spike_rate_hz"]),
                    unit="Hz",
                    seed_id=row.get("network_seed", _seed_id(seed_dir)),
                    source_file=_display_path(path, repo_root),
                    phase=str(row["phase"]),
                    run_mode=_run_mode(seeds),
                )
            )
    panel_df = _sort_layer_phase(pd.DataFrame(rows))
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="spike_rate_hz", run_mode=_run_mode(seeds), group_cols=["layer", "phase"])
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok" if rows else "missing_source")
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_s2_delay_decode_curve_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    return _build_delay_decode_curve(spec, repo_root, output_dir)


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
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 supplement experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    for seed_dir in seeds:
        contrast_path = seed_dir / "data" / "metrics" / "supp_dms_delay_sweep_contrast.csv"
        metrics_path = seed_dir / "data" / "metrics" / "supp_dms_delay_sweep_metrics.csv"
        if contrast_path.exists():
            df = pd.read_csv(contrast_path)
            sources.append(_source_entry(contrast_path, repo_root))
            for _, row in df.iterrows():
                value = row.get("stsp_interference", row.get("contrast", row.get("static_minus_dynamic", 0.0)))
                rows.append(
                    _canonical_row(
                        figure_id,
                        panel_id,
                        metric="static_minus_dynamic_accuracy",
                        condition="static_minus_dynamic",
                        value=_to_percent(value),
                        unit="percent",
                        seed_id=row.get("network_seed", _seed_id(seed_dir)),
                        source_file=_display_path(contrast_path, repo_root),
                        delay_ms=row.get("delay_ms", ""),
                        run_mode=_run_mode(seeds),
                    )
                )
            continue
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            sources.append(_source_entry(metrics_path, repo_root))
            pivot = df.pivot_table(index=["network_seed", "delay_ms"], columns="condition", values="acc_probe", aggfunc="mean").reset_index()
            for _, row in pivot.iterrows():
                if "static_frozen" not in row or "dynamic_intact" not in row:
                    continue
                rows.append(
                    _canonical_row(
                        figure_id,
                        panel_id,
                        metric="static_minus_dynamic_accuracy",
                        condition="static_minus_dynamic",
                        value=_to_percent(row["static_frozen"] - row["dynamic_intact"]),
                        unit="percent",
                        seed_id=row.get("network_seed", _seed_id(seed_dir)),
                        source_file=_display_path(metrics_path, repo_root),
                        delay_ms=row.get("delay_ms", ""),
                        run_mode=_run_mode(seeds),
                    )
                )
        else:
            warnings.append(f"Missing DMS delay contrast sources under {_display_path(seed_dir, repo_root)}")
    if not rows:
        return missing_adapter_result(spec, repo_root, output_dir, "Missing DMS delay contrast sources for Fig.1 supplement S2D.")
    panel_df = pd.DataFrame(rows).sort_values(["delay_ms", "seed_id"], kind="stable").reset_index(drop=True)
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="static_minus_dynamic_accuracy", run_mode=_run_mode(seeds), group_cols=["delay_ms"])
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok")
    manifest["delay_ms_values"] = _unique_numeric(panel_df, "delay_ms")
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


def build_s2_substrate_specificity_adapter(spec: Mapping[str, Any], repo_root: Path, output_dir: Path) -> AdapterResult:
    figure_id, panel_id = _ids(spec)
    seeds = _seed_dirs(spec, repo_root)
    if not seeds:
        return missing_adapter_result(spec, repo_root, output_dir, "Fig.1 supplement experiment seed directory not found.")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    supplement_files: list[dict[str, Any]] = []
    warnings = _run_mode_warnings(seeds)
    conditions = [str(v) for v in spec.get("conditions", CONDITION_ORDER)]
    primary_names = [
        "supp_substrate_shuffle_metrics.csv",
        "compat_metrics_condition_summary.csv",
        "compat_metrics_error_bias.csv",
        "compat_metrics_collapse_summary.csv",
    ]
    companion_names = [
        ("data/raw", "supp_state_intervention_manifest.csv"),
        ("data/metrics", "compat_metrics_condition_summary.csv"),
        ("data/metrics", "compat_metrics_error_bias.csv"),
        ("data/metrics", "compat_metrics_collapse_summary.csv"),
        ("data/metrics", "compat_metrics_bootstrap_tests.csv"),
    ]
    for seed_dir in seeds:
        for folder, name in companion_names:
            companion = seed_dir / folder / name
            supplement_files.append({"path": _display_path(companion, repo_root), "exists": companion.exists()})
        path = _first_existing(seed_dir / "data" / "metrics", primary_names)
        if path is None:
            warnings.append(f"Missing substrate specificity sources under {_display_path(seed_dir, repo_root)}")
            continue
        df = pd.read_csv(path)
        sources.append(_source_entry(path, repo_root))
        if "condition" not in df.columns or "donor_attribution_rate" not in df.columns:
            warnings.append(f"Substrate source lacks condition/donor_attribution_rate columns: {_display_path(path, repo_root)}")
            continue
        work = df[df["condition"].astype(str).isin(conditions)].copy()
        if work.empty:
            continue
        seed_col = "network_seed" if "network_seed" in work.columns else None
        if seed_col is None:
            work["network_seed"] = _seed_id(seed_dir)
            seed_col = "network_seed"
        for seed_value, seed_part in work.groupby(seed_col, dropna=False):
            dyn = seed_part[seed_part["condition"].astype(str).eq("dynamic_intact")]
            dynamic_donor = _fraction(dyn.iloc[0].get("donor_attribution_rate", 0.0)) if not dyn.empty else 0.0
            dynamic_original = _fraction(dyn.iloc[0].get("sample_attribution_rate", 0.0)) if not dyn.empty and "sample_attribution_rate" in dyn.columns else 0.0
            for _, row in seed_part.iterrows():
                donor_rate = _fraction(row.get("donor_attribution_rate", 0.0))
                sample_rate = _fraction(row.get("sample_attribution_rate", 0.0))
                rows.append(
                    _canonical_row(
                        figure_id,
                        panel_id,
                        metric="donor_gain_vs_dynamic",
                        condition=str(row.get("condition", "")),
                        value=(donor_rate - dynamic_donor) * 100.0,
                        unit="percent",
                        seed_id=seed_value,
                        source_file=_display_path(path, repo_root),
                        substrate=row.get("substrate", ""),
                        sample_attribution_rate=sample_rate * 100.0,
                        donor_attribution_rate=donor_rate * 100.0,
                        original_drop_vs_dynamic=(dynamic_original - sample_rate) * 100.0,
                        acc_probe=_to_percent(row.get("acc_probe", 0.0)),
                        error_rate=_to_percent(row.get("error_rate", 0.0)),
                        silent_rate=_to_percent(row.get("silent_rate", 0.0)),
                        n_trials=row.get("n_trials", ""),
                        run_mode=_run_mode(seeds),
                    )
                )
    if not rows:
        return missing_adapter_result(spec, repo_root, output_dir, "Missing substrate specificity sources for Fig.1 supplement S2E.")
    panel_df = _sort_conditions(pd.DataFrame(rows), conditions)
    stats = _stats_payload(figure_id, panel_id, panel_df, metric="donor_gain_vs_dynamic", run_mode=_run_mode(seeds), group_cols=["condition"])
    stats["conditions"] = _unique(panel_df, "condition")
    stats["supplement_files"] = supplement_files
    manifest = _manifest(figure_id, panel_id, sources, seeds, status="ok")
    manifest["conditions"] = stats["conditions"]
    manifest["supplement_files"] = supplement_files
    return write_adapter_outputs(output_dir, figure_id, panel_id, panel_df, stats, manifest, warnings)


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
