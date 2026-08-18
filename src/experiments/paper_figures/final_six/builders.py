from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .schema import (
    EXPECTED_SEEDS,
    SOURCE_MANIFEST_COLUMNS,
    STATISTICS_COLUMNS,
    LoadedSource,
    SourceDescriptor,
    build_statistics,
    finalize_source_records,
    load_source,
    make_plot_data,
    record_file_source,
    resolve_source_paths,
    schematic_statistics,
    validate_plot_data,
)


BUILDER_MODULE = "src.experiments.paper_figures.final_six.builders"
LAYER_ORDER = ("layer1", "layer2", "layer3")
GROUP_ORDER = (
    "overlap_dominant",
    "probe_only_dominant",
    "balanced",
    "random_matched",
)


@dataclass(frozen=True)
class BuilderContext:
    repo_root: Path
    output_root: Path
    figure_id: str
    builder_version: str


@dataclass
class PanelResult:
    panel_id: str
    panel_type: str
    plot_data: Optional[pd.DataFrame]
    statistics: pd.DataFrame
    source_manifest: pd.DataFrame
    unique_key: Sequence[str] = ()
    extra_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    extra_metrics: dict[str, pd.DataFrame] = field(default_factory=dict)
    panel_meta: dict[str, Any] = field(default_factory=dict)
    cohort_record: Optional[dict[str, Any]] = None


def _load(ctx: BuilderContext, panel_id: str, descriptor: SourceDescriptor) -> LoadedSource:
    return load_source(
        repo_root=ctx.repo_root,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        descriptor=descriptor,
    )


def _manifest(
    ctx: BuilderContext,
    panel_id: str,
    records: Sequence[dict[str, Any]],
    *,
    output_rows: int,
    input_rows: int,
    output_csvs: Sequence[str],
    exclusion_reason: str = "",
) -> pd.DataFrame:
    excluded_rows = max(0, int(input_rows) - int(output_rows))
    return finalize_source_records(
        records,
        output_rows=output_rows,
        excluded_rows=excluded_rows,
        exclusion_reason=exclusion_reason,
        output_csv=";".join(output_csvs),
        builder_module=BUILDER_MODULE,
        builder_version=ctx.builder_version,
    )


def _selected_source_manifest(
    ctx: BuilderContext,
    records: Sequence[dict[str, Any]],
    selected_rows: Sequence[int],
    *,
    output_csvs: Sequence[str],
) -> pd.DataFrame:
    if len(records) != len(selected_rows):
        raise ValueError("source records and selected-row counts must align")
    frames: list[pd.DataFrame] = []
    output_csv = ";".join(output_csvs)
    for record, selected in zip(records, selected_rows):
        input_rows = int(record["input_rows"])
        selected_count = int(selected)
        if selected_count < 0 or selected_count > input_rows:
            raise ValueError(
                f"invalid selected-row count {selected_count} for {input_rows} inputs"
            )
        excluded_rows = input_rows - selected_count
        frames.append(
            finalize_source_records(
                [record],
                output_rows=selected_count,
                excluded_rows=excluded_rows,
                exclusion_reason=(
                    "source units outside the frozen schematic selection were excluded"
                    if excluded_rows
                    else ""
                ),
                output_csv=output_csv,
                builder_module=BUILDER_MODULE,
                builder_version=ctx.builder_version,
            )
        )
    return pd.concat(frames, ignore_index=True)[list(SOURCE_MANIFEST_COLUMNS)]


def _statistics_values(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    status: str,
    null_by_endpoint: Optional[Mapping[str, float]] = None,
    contrast: str = "",
    p_adjust_family: str = "",
) -> pd.DataFrame:
    null_by_endpoint = dict(null_by_endpoint or {})
    required = {"network_seed", "endpoint", "value", "unit", *group_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"statistics source missing columns {missing}")
    output = frame.loc[:, ["network_seed", "endpoint", "value", "unit", *group_columns]].copy()
    output["contrast"] = contrast
    output["group"] = output.apply(
        lambda row: "|".join(
            [str(row["endpoint"]), *[str(row[column]) for column in group_columns]]
        ),
        axis=1,
    )
    output["null_value"] = output["endpoint"].map(null_by_endpoint)
    output["statistics_status"] = status
    output["p_adjust_family"] = p_adjust_family
    return output


def _contrast_statistics_values(
    frame: pd.DataFrame,
    *,
    endpoint: str,
    contrast: str,
    unit: str,
    status: str = "predeclared_recomputed",
    null_value: float = 0.0,
    p_adjust_family: str = "",
) -> pd.DataFrame:
    output = frame.loc[:, ["network_seed", "value"]].copy()
    output["endpoint"] = endpoint
    output["contrast"] = contrast
    output["group"] = contrast
    output["null_value"] = float(null_value)
    output["unit"] = unit
    output["statistics_status"] = status
    output["p_adjust_family"] = p_adjust_family
    return output


def _finalize_quantitative_panel(
    ctx: BuilderContext,
    panel_id: str,
    plot_data: pd.DataFrame,
    statistics: pd.DataFrame,
    source_records: Sequence[dict[str, Any]],
    *,
    input_rows: int,
    unique_key: Sequence[str],
    extra_data: Optional[dict[str, pd.DataFrame]] = None,
    extra_metrics: Optional[dict[str, pd.DataFrame]] = None,
    exclusion_reason: str = "",
) -> PanelResult:
    cohort = validate_plot_data(
        plot_data,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        unique_key=unique_key,
    )
    output_csvs = [f"{ctx.figure_id}/data/panel_{panel_id}_plot_data.csv"]
    output_csvs.extend(
        f"{ctx.figure_id}/data/{name}" for name in sorted((extra_data or {}).keys())
    )
    manifest = _manifest(
        ctx,
        panel_id,
        source_records,
        output_rows=len(plot_data),
        input_rows=input_rows,
        output_csvs=output_csvs,
        exclusion_reason=exclusion_reason,
    )
    return PanelResult(
        panel_id=panel_id,
        panel_type="quantitative",
        plot_data=plot_data,
        statistics=statistics,
        source_manifest=manifest,
        unique_key=tuple(unique_key),
        extra_data=dict(extra_data or {}),
        extra_metrics=dict(extra_metrics or {}),
        cohort_record=cohort,
    )


def _asset_panel(
    ctx: BuilderContext,
    panel_id: str,
    *,
    relative_path: str,
    role: str,
    semantics: str,
    allowed_cleanup: str,
) -> PanelResult:
    path = ctx.repo_root / relative_path
    descriptor = SourceDescriptor(
        key=f"{ctx.figure_id}.{panel_id}.asset",
        pattern=relative_path,
        source_level="manual_asset",
        producer_task="registered manual SVG asset",
        filters="asset_sha256_verified",
        held_fixed=semantics,
        aggregation_path="manual SVG asset -> contained vector panel",
        seeded=False,
    )
    source_record = record_file_source(
        repo_root=ctx.repo_root,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        descriptor=descriptor,
        path=path,
    )
    text = path.read_text(encoding="utf-8", errors="strict")
    viewbox_match = re.search(r'viewBox\s*=\s*["\']([^"\']+)["\']', text)
    if not viewbox_match:
        raise ValueError(f"{ctx.figure_id}{panel_id}: SVG has no viewBox: {path}")
    viewbox = viewbox_match.group(1).strip()
    asset_manifest = pd.DataFrame(
        [
            {
                "figure_id": ctx.figure_id,
                "panel_id": panel_id,
                "asset_path": source_record["source_path"],
                "asset_sha256": source_record["source_sha256"],
                "viewBox": viewbox,
                "asset_role": role,
                "semantics": semantics,
                "allowed_cleanup": allowed_cleanup,
                "statistics_status": "not_applicable",
            }
        ]
    )
    manifest = _manifest(
        ctx,
        panel_id,
        [source_record],
        output_rows=1,
        input_rows=1,
        output_csvs=[f"{ctx.figure_id}/meta/panel_{panel_id}_asset_manifest.csv"],
    )
    return PanelResult(
        panel_id=panel_id,
        panel_type="schematic",
        plot_data=None,
        statistics=schematic_statistics(ctx.figure_id, panel_id),
        source_manifest=manifest,
        panel_meta={"asset_manifest": asset_manifest},
    )


def build_fig1(ctx: BuilderContext) -> list[PanelResult]:
    return [
        _asset_panel(
            ctx,
            "a",
            relative_path="results/paper_figures/outputs/structure-enhanced.svg",
            role="fixed STSP-SNN architecture",
            semantics=(
                "input image -> DoG encoder -> spiking layers -> decision/readout; "
                "STSP feedforward connections and local inhibition"
            ),
            allowed_cleanup=(
                "font normalization; whitespace normalization; remove unintended dark "
                "decision/readout background; preserve editable vectors and source aspect"
            ),
        ),
        _build_fig1b(ctx),
        _build_fig1c(ctx),
        _build_fig1d(ctx),
        _build_fig1e(ctx),
    ]


def _build_fig1b(ctx: BuilderContext) -> PanelResult:
    panel_id = "b"
    source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig1.baseline",
            pattern=(
                "results/paper_figure_multi_seed/fig1_functional_stsp_substrate/"
                "fig1_functional_stsp_substrate/seed_*/data/metrics/"
                "panel_b_baseline_metrics_by_network.csv"
            ),
            source_level="network_metric",
            producer_task="fig1 baseline evaluation",
            filters="metric=overall_recall",
            held_fixed="chance=10%; seeds=1000-1019",
            aggregation_path="trial recall -> network overall recall",
            required_columns=("network_seed", "overall_recall"),
        ),
    )
    work = source.frame.loc[:, ["network_seed", "overall_recall"]].copy()
    work["overall_recall_percent"] = pd.to_numeric(work["overall_recall"], errors="coerce") * 100.0
    plot = make_plot_data(
        work,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_metric",
        endpoint="overall_recall",
        condition="network",
        value="overall_recall_percent",
        unit="percent",
    ).sort_values(["network_seed"], kind="mergesort")
    values = _statistics_values(
        plot,
        group_columns=("condition",),
        status="descriptive_only",
        null_by_endpoint={"overall_recall": 10.0},
    )
    statistics = build_statistics(values, figure_id=ctx.figure_id, panel_id=panel_id)
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=("figure_id", "panel_id", "network_seed", "endpoint"),
    )


def _build_fig1c(ctx: BuilderContext) -> PanelResult:
    panel_id = "c"
    source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig1.time_binned_phase_rates",
            pattern=(
                "results/multi_seed_rollout/fig1_time_binned_firing/"
                "seed_*/data/metrics/supp_time_binned_firing_rates.csv"
            ),
            source_level="trial",
            producer_task="fig1 time_binned_firing_rate_control",
            filters="bin_width=50 ms; time=0-600 ms; stimulus plus 400 ms delay",
            held_fixed=(
                "dynamic STSP; original DMS trials; layers=layer1,layer2,layer3; "
                "stimulus=0-200 ms; delay=200-600 ms"
            ),
            aggregation_path=(
                "per-step spikes -> 50 ms trial bins -> network x layer x time-bin mean"
            ),
            independent_unit="network_seed",
            required_columns=(
                "network_seed",
                "trial_id",
                "layer",
                "phase",
                "bin_start_ms",
                "bin_end_ms",
                "time_ms",
                "time_window_ms",
                "stimulus_start_ms",
                "stimulus_end_ms",
                "spike_count",
                "spike_rate_hz",
                "dms_trial_specs_digest",
            ),
        ),
    )
    phase_source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig1.phase_rate_validation",
            pattern=(
                "results/paper_figure_multi_seed/fig1_functional_stsp_substrate/"
                "fig1_functional_stsp_substrate/seed_*/data/metrics/"
                "supp_phase_firing_rates.csv"
            ),
            source_level="trial",
            producer_task="fig1 firing-rate control",
            filters="phase in stimulus,early_delay,late_delay; probe excluded",
            held_fixed="original DMS trials and 200 ms validation windows",
            aggregation_path="trial phase spike counts -> phase-average population rate",
            independent_unit="network_seed",
            required_columns=(
                "network_seed",
                "trial_id",
                "layer",
                "phase",
                "time_window_ms",
                "spike_rate_hz",
            ),
        ),
    )
    selected = source.frame.loc[
        source.frame["layer"].isin(LAYER_ORDER)
        & source.frame["phase"].isin(("stimulus", "early_delay", "late_delay"))
    ].copy()
    numeric_columns = (
        "bin_start_ms",
        "bin_end_ms",
        "time_ms",
        "time_window_ms",
        "stimulus_start_ms",
        "stimulus_end_ms",
        "spike_rate_hz",
    )
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    observed_bin_widths = set(selected["time_window_ms"].astype(float))
    if observed_bin_widths != {50.0}:
        raise ValueError(
            f"fig1c requires exact 50 ms bins, observed={sorted(observed_bin_widths)}"
        )
    observed_starts = sorted(selected["bin_start_ms"].unique().tolist())
    expected_starts = [float(value) for value in range(0, 600, 50)]
    if observed_starts != expected_starts:
        raise ValueError(
            f"fig1c time-bin coverage mismatch: expected={expected_starts}, "
            f"observed={observed_starts}"
        )
    stimulus_starts = set(selected["stimulus_start_ms"].astype(float))
    stimulus_ends = set(selected["stimulus_end_ms"].astype(float))
    if stimulus_starts != {0.0} or stimulus_ends != {200.0}:
        raise ValueError(
            "fig1c stimulus boundaries must come from the original 0-200 ms "
            f"protocol; starts={stimulus_starts}, ends={stimulus_ends}"
        )
    per_trial_counts = selected.groupby(
        ["network_seed", "trial_id", "layer"], as_index=False
    ).size()
    if not per_trial_counts["size"].eq(12).all():
        raise ValueError("fig1c requires 12 complete 50 ms bins per trial and layer")

    reconstructed = (
        selected.groupby(
            ["network_seed", "trial_id", "layer", "phase"],
            as_index=False,
        )["spike_rate_hz"]
        .mean()
        .rename(columns={"spike_rate_hz": "time_binned_rate_hz"})
    )
    supplied = phase_source.frame.loc[
        phase_source.frame["layer"].isin(LAYER_ORDER)
        & phase_source.frame["phase"].isin(("stimulus", "early_delay", "late_delay"))
    ].copy()
    supplied["phase_rate_hz"] = pd.to_numeric(
        supplied["spike_rate_hz"], errors="raise"
    )
    validation = reconstructed.merge(
        supplied.loc[
            :,
            [
                "network_seed",
                "trial_id",
                "layer",
                "phase",
                "phase_rate_hz",
            ],
        ],
        on=["network_seed", "trial_id", "layer", "phase"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not validation["_merge"].eq("both").all():
        raise ValueError("fig1c 50 ms replay and persisted phase summary keys differ")
    validation["difference_hz"] = (
        validation["time_binned_rate_hz"] - validation["phase_rate_hz"]
    )
    network_validation = (
        validation.groupby(["network_seed", "layer", "phase"], as_index=False)[
            ["time_binned_rate_hz", "phase_rate_hz", "difference_hz"]
        ]
        .mean()
        .sort_values(["network_seed", "layer", "phase"], kind="mergesort")
    )
    network_validation["relative_difference"] = (
        network_validation["difference_hz"].abs()
        / network_validation["phase_rate_hz"].abs().clip(lower=1.0)
    )
    delay_validation = network_validation.loc[
        network_validation["phase"].isin(("early_delay", "late_delay"))
    ]
    if float(delay_validation["time_binned_rate_hz"].abs().max()) > 1e-12:
        raise ValueError("fig1c replay produced nonzero delay-period firing")
    if float(network_validation["relative_difference"].max()) > 0.01:
        raise ValueError(
            "fig1c replay differs from persisted phase rates by more than 1% "
            "at the network x layer x phase level"
        )

    network = (
        selected.groupby(
            [
                "network_seed",
                "layer",
                "phase",
                "bin_start_ms",
                "bin_end_ms",
                "time_ms",
                "time_window_ms",
                "stimulus_start_ms",
                "stimulus_end_ms",
            ],
            as_index=False,
        )["spike_rate_hz"]
        .mean()
        .rename(columns={"spike_rate_hz": "network_mean_spike_rate_hz"})
    )
    plot = make_plot_data(
        network,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_metric",
        endpoint="population_spike_rate",
        condition="dynamic_intact",
        value="network_mean_spike_rate_hz",
        unit="Hz",
        dimensions=(
            "layer",
            "phase",
            "bin_start_ms",
            "bin_end_ms",
            "time_ms",
            "time_window_ms",
            "stimulus_start_ms",
            "stimulus_end_ms",
        ),
    )
    plot["layer"] = pd.Categorical(plot["layer"], LAYER_ORDER, ordered=True)
    plot = plot.sort_values(
        ["network_seed", "layer", "time_ms"], kind="mergesort"
    )
    plot["layer"] = plot["layer"].astype(str)
    values = _statistics_values(
        plot,
        group_columns=("layer", "time_ms"),
        status="descriptive_only",
    )
    statistics = build_statistics(values, figure_id=ctx.figure_id, panel_id=panel_id)
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        [*source.records, *phase_source.records],
        input_rows=len(source.frame) + len(phase_source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "layer",
            "time_ms",
        ),
        extra_metrics={
            "panel_c_time_bin_validation.csv": network_validation.drop(
                columns=[],
                errors="ignore",
            )
        },
        exclusion_reason=(
            "probe excluded; plot covers the original 200 ms stimulus and "
            "400 ms delay using authorized 50 ms replay bins"
        ),
    )


def _build_fig1d(ctx: BuilderContext) -> PanelResult:
    panel_id = "d"
    source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig1.delay_decode",
            pattern=(
                "results/paper_figure_multi_seed/fig1_functional_stsp_substrate/"
                "fig1_functional_stsp_substrate/seed_*/data/metrics/"
                "panel_c_delay_decode_metrics.csv"
            ),
            source_level="network_metric",
            producer_task="fig1 delay decoder",
            filters="feature_type=ux_concat; delay_ms in 100,200,400,800,1200",
            held_fixed="layers=layer1,layer2,layer3; chance=10%",
            aggregation_path="classifier trial predictions -> network decoding accuracy",
            required_columns=(
                "network_seed",
                "layer",
                "delay_ms",
                "feature_type",
                "acc",
            ),
        ),
    )
    delays = (100, 200, 400, 800, 1200)
    selected = source.frame.loc[
        source.frame["feature_type"].eq("ux_concat")
        & source.frame["delay_ms"].isin(delays)
        & source.frame["layer"].isin(LAYER_ORDER)
    ].copy()
    selected["accuracy_percent"] = pd.to_numeric(selected["acc"], errors="coerce") * 100.0
    plot = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_metric",
        endpoint="ux_delay_decoding_accuracy",
        condition="ux_concat",
        value="accuracy_percent",
        unit="percent",
        dimensions=("layer", "delay_ms"),
    ).sort_values(["network_seed", "layer", "delay_ms"], kind="mergesort")
    values = _statistics_values(
        plot,
        group_columns=("layer", "delay_ms"),
        status="descriptive_only",
        null_by_endpoint={"ux_delay_decoding_accuracy": 10.0},
    )
    statistics = build_statistics(values, figure_id=ctx.figure_id, panel_id=panel_id)
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "layer",
            "delay_ms",
        ),
    )


def _build_fig1e(ctx: BuilderContext) -> PanelResult:
    panel_id = "e"
    source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig1.error_composition",
            pattern=(
                "results/paper_figure_multi_seed/fig1_functional_stsp_substrate/"
                "fig1_functional_stsp_substrate/seed_*/data/raw/"
                "panel_d_dms_condition_trial_readout.csv"
            ),
            source_level="trial",
            producer_task="fig1 DMS u/x trial readout",
            filters=(
                "condition in dynamic_intact,ux_trial_shuffle; "
                "is_correct_probe=0"
            ),
            held_fixed=(
                "joint u/x permutation; intact dynamic reference; "
                "Original/Donor/Other mutually exclusive attribution"
            ),
            aggregation_path=(
                "error-trial attribution indicators -> within-network error-pool "
                "composition"
            ),
            independent_unit="network_seed",
            required_columns=(
                "network_seed",
                "trial_id",
                "condition",
                "is_correct_probe",
                "pred_is_original_sample",
                "pred_is_donor_sample",
                "pred_is_other",
            ),
        ),
    )
    condition_order = ("dynamic_intact", "ux_trial_shuffle")
    selected = source.frame.loc[
        source.frame["condition"].isin(condition_order)
    ].copy()
    selected["is_correct_probe"] = pd.to_numeric(
        selected["is_correct_probe"], errors="raise"
    ).astype(int)
    indicator_columns = (
        "pred_is_original_sample",
        "pred_is_donor_sample",
        "pred_is_other",
    )
    for column in indicator_columns:
        selected[column] = pd.to_numeric(selected[column], errors="raise").astype(int)
    errors = selected.loc[selected["is_correct_probe"].eq(0)].copy()
    if errors.empty:
        raise ValueError("fig1e error pool is empty")
    indicator_sum = errors.loc[:, indicator_columns].sum(axis=1)
    if not indicator_sum.eq(1).all():
        raise ValueError(
            "fig1e Original/Donor/Other indicators are not mutually exclusive "
            "and exhaustive within the error pool"
        )
    network = (
        errors.groupby(["network_seed", "condition"], as_index=False)[
            list(indicator_columns)
        ]
        .mean()
    )
    error_counts = (
        errors.groupby(["network_seed", "condition"], as_index=False)
        .size()
        .rename(columns={"size": "error_trial_count"})
    )
    network = network.merge(
        error_counts,
        on=["network_seed", "condition"],
        how="left",
        validate="one_to_one",
    )
    long = network.melt(
        id_vars=("network_seed", "condition", "error_trial_count"),
        value_vars=indicator_columns,
        var_name="category_source",
        value_name="composition_fraction",
    )
    category_map = {
        "pred_is_original_sample": "Original",
        "pred_is_donor_sample": "Donor",
        "pred_is_other": "Other",
    }
    long["category"] = long["category_source"].map(category_map)
    long["composition_percent"] = (
        pd.to_numeric(long["composition_fraction"], errors="raise") * 100.0
    )
    plot = make_plot_data(
        long,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_composition",
        endpoint="error_composition",
        condition="error_pool",
        value="composition_percent",
        unit="percent",
        dimensions=("category", "error_trial_count"),
    )
    plot["condition"] = long["condition"].to_numpy()
    plot["condition"] = pd.Categorical(
        plot["condition"], condition_order, ordered=True
    )
    plot["category"] = pd.Categorical(
        plot["category"], ("Original", "Donor", "Other"), ordered=True
    )
    plot = plot.sort_values(
        ["network_seed", "condition", "category"], kind="mergesort"
    )
    plot["condition"] = plot["condition"].astype(str)
    plot["category"] = plot["category"].astype(str)
    audit = (
        plot.groupby(["network_seed", "condition"], as_index=False)
        .agg(
            composition_sum_percent=("value", "sum"),
            error_trial_count=("error_trial_count", "first"),
        )
        .sort_values(["network_seed", "condition"], kind="mergesort")
    )
    if not np.allclose(
        audit["composition_sum_percent"].to_numpy(dtype=float),
        100.0,
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError("fig1e network error compositions do not sum to 100%")
    values = _statistics_values(
        plot,
        group_columns=("condition", "category"),
        status="descriptive_only",
    )
    statistics = build_statistics(values, figure_id=ctx.figure_id, panel_id=panel_id)
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "condition",
            "category",
        ),
        extra_metrics={"panel_e_composition_audit.csv": audit},
        exclusion_reason=(
            "correct trials and non-primary conditions excluded; the denominator "
            "is the remaining error pool within each network and condition"
        ),
    )


def _load_mnist_test_images(path: Path, *, panel_label: str) -> np.ndarray:
    raw_bytes = path.read_bytes()
    header = np.frombuffer(raw_bytes, dtype=">u4", count=4)
    expected = (2051, 10000, 28, 28)
    if tuple(int(value) for value in header) != expected:
        raise ValueError(
            f"{panel_label} unexpected MNIST test-image header: {header.tolist()}"
        )
    return np.frombuffer(raw_bytes, dtype=np.uint8, offset=16).reshape(
        int(header[1]),
        int(header[2]),
        int(header[3]),
    )


def _stimulus_pixel_frame(
    *,
    figure_id: str,
    panel_id: str,
    raw_images: np.ndarray,
    exemplars: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    pixel_y, pixel_x = np.indices((28, 28), dtype=np.int16)
    tables: list[pd.DataFrame] = []
    for exemplar in exemplars:
        image_id = int(exemplar["image_id"])
        if image_id < 0 or image_id >= len(raw_images):
            raise ValueError(
                f"{figure_id}{panel_id} image_id is outside the MNIST test set: "
                f"{image_id}"
            )
        image = np.asarray(raw_images[image_id], dtype=np.uint8)
        table = pd.DataFrame(
            {
                "pixel_x": pixel_x.ravel(),
                "pixel_y": pixel_y.ravel(),
                "pixel_value": image.ravel(),
                "normalized_intensity": image.astype(np.float64).ravel() / 255.0,
            }
        )
        identifiers = [
            ("figure_id", figure_id),
            ("panel_id", panel_id),
            *list(exemplar.items()),
        ]
        for column, value in reversed(identifiers):
            table.insert(0, column, value)
        tables.append(table)
    if not tables:
        raise ValueError(f"{figure_id}{panel_id} has no schematic stimuli")
    return pd.concat(tables, ignore_index=True)


def _build_fig2a_schematic(ctx: BuilderContext) -> PanelResult:
    panel_id = "a"
    asset_relative = "results/paper_figures/outputs/DMS-enhanced.svg"
    protocol_root = (
        ctx.repo_root
        / "results"
        / "multi_seed_rollout"
        / "fig2"
        / "fixed_b_mechanism_confirmatory"
        / "frozen_protocol"
    )
    history_specs_path = protocol_root / "history_specs.csv"
    history_manifest_path = protocol_root / "history_input_manifest.csv"
    b_anchor_specs_path = protocol_root / "b_anchor_specs.csv"
    raw_images_path = ctx.repo_root / "data" / "MNIST" / "raw" / "t10k-images-idx3-ubyte"
    history_specs = pd.read_csv(history_specs_path)
    history_manifest = pd.read_csv(history_manifest_path)
    b_anchor_specs = pd.read_csv(b_anchor_specs_path)

    selected = history_specs.loc[
        history_specs["history_family_id"].eq(1)
        & history_specs["history_condition"].isin(["A", "C"])
        & history_specs["prefix_k"].eq(1)
    ].copy()
    if set(selected["history_condition"]) != {"A", "C"} or len(selected) != 2:
        raise ValueError("fig2a requires the frozen family-1 History A/C exemplar pair")

    b_rows = b_anchor_specs.loc[b_anchor_specs["b_anchor_id"].eq(30)]
    if len(b_rows) != 1:
        raise ValueError("fig2a requires the unique frozen test B anchor 30")
    b_row = b_rows.iloc[0]
    if int(b_row["B_image_id"]) != 3536 or int(b_row["B_label"]) != 0:
        raise ValueError("fig2a frozen B anchor 30 identity changed")

    exemplars: list[dict[str, Any]] = []
    for sequence_index, condition in enumerate(("A", "C"), start=1):
        row = selected.loc[selected["history_condition"].eq(condition)].iloc[0]
        image_ids = json.loads(str(row["sequence_image_ids"]))
        labels = json.loads(str(row["sequence_labels"]))
        if len(image_ids) != 1 or len(labels) != 1:
            raise ValueError(f"fig2a History {condition} is not a one-item history")
        image_id = int(image_ids[0])
        label = int(labels[0])
        manifest_rows = history_manifest.loc[
            history_manifest["image_id"].eq(image_id)
            & history_manifest["label"].eq(label)
        ]
        if len(manifest_rows) != 1:
            raise ValueError(
                f"fig2a History {condition} exemplar is ambiguous in frozen manifest"
            )
        manifest_row = manifest_rows.iloc[0]
        exemplars.append(
            {
                "stimulus_role": condition,
                "sequence_index": sequence_index,
                "protocol_context": f"history_{condition}",
                "protocol_row_id": int(row["history_row_id"]),
                "image_id": image_id,
                "label": label,
                "encoding_seed": int(manifest_row["encoding_seed"]),
                "image_sha256": str(manifest_row["image_sha256"]),
                "render_occurrences": 1,
            }
        )
    exemplars.append(
        {
            "stimulus_role": "B",
            "sequence_index": 3,
            "protocol_context": "fixed_B",
            "protocol_row_id": int(b_row["b_anchor_id"]),
            "image_id": int(b_row["B_image_id"]),
            "label": int(b_row["B_label"]),
            "encoding_seed": int(b_row["encoding_seed"]),
            "image_sha256": str(b_row["B_image_sha256"]),
            "render_occurrences": 2,
        }
    )
    raw_images = _load_mnist_test_images(raw_images_path, panel_label="fig2a")
    stimulus_frame = _stimulus_pixel_frame(
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        raw_images=raw_images,
        exemplars=exemplars,
    )

    source_specs = (
        (
            asset_relative,
            "fig2.a.layout_reference",
            "manual_asset",
            "DMS layout reference",
            "design_reference_only",
            "scientific identities come from frozen CSV sources",
            "manual DMS composition -> layout reference only",
        ),
        (
            history_specs_path.relative_to(ctx.repo_root).as_posix(),
            "fig2.a.history_specs",
            "frozen_protocol",
            "frozen History A/C identities",
            "history_family_id=1; prefix_k=1; conditions=A,C",
            "one-item histories; protocol seed 20260724",
            "frozen history identities -> selected image IDs",
        ),
        (
            history_manifest_path.relative_to(ctx.repo_root).as_posix(),
            "fig2.a.history_input_manifest",
            "validated_artifact",
            "encoded-history row and hash manifest",
            "selected A/C image IDs and labels",
            "persisted encoding seeds and image hashes",
            "selected IDs -> validated history stimulus metadata",
        ),
        (
            b_anchor_specs_path.relative_to(ctx.repo_root).as_posix(),
            "fig2.a.fixed_b_anchor",
            "frozen_protocol",
            "frozen exact-B anchor identities",
            "b_anchor_id=30; test row",
            "one byte-identical B is rendered in both counterfactual lanes",
            "frozen B identity -> shared schematic stimulus",
        ),
        (
            raw_images_path.relative_to(ctx.repo_root).as_posix(),
            "fig2.a.raw_input_images",
            "dataset",
            "raw MNIST test stimuli selected by the frozen protocol",
            "image IDs 9301, 9589, and 3536",
            "pixel values are not transformed beyond normalization to [0,1]",
            "raw MNIST pixels -> schematic stimulus tiles",
        ),
    )
    source_row_counts = {
        "fig2.a.layout_reference": (1, 1),
        "fig2.a.history_specs": (len(history_specs), len(selected)),
        "fig2.a.history_input_manifest": (
            len(history_manifest),
            len(selected),
        ),
        "fig2.a.fixed_b_anchor": (len(b_anchor_specs), len(b_rows)),
        "fig2.a.raw_input_images": (len(raw_images), len(exemplars)),
    }
    source_records = []
    for (
        relative_path,
        key,
        source_level,
        producer_task,
        filters,
        held_fixed,
        aggregation_path,
    ) in source_specs:
        descriptor = SourceDescriptor(
            key=key,
            pattern=relative_path,
            source_level=source_level,
            producer_task=producer_task,
            filters=filters,
            held_fixed=held_fixed,
            aggregation_path=aggregation_path,
            seeded=False,
        )
        source_records.append(
            record_file_source(
                repo_root=ctx.repo_root,
                figure_id=ctx.figure_id,
                panel_id=panel_id,
                descriptor=descriptor,
                path=ctx.repo_root / relative_path,
                input_rows=int(source_row_counts[key][0]),
            )
        )
    manifest = _selected_source_manifest(
        ctx,
        source_records,
        [source_row_counts[spec[1]][1] for spec in source_specs],
        output_csvs=(f"{ctx.figure_id}/data/panel_a_input_stimuli.csv",),
    )
    return PanelResult(
        panel_id=panel_id,
        panel_type="schematic",
        plot_data=None,
        statistics=schematic_statistics(ctx.figure_id, panel_id),
        source_manifest=manifest,
        extra_data={"panel_a_input_stimuli.csv": stimulus_frame},
    )


def build_fig2(ctx: BuilderContext) -> list[PanelResult]:
    return [
        _build_fig2a_schematic(ctx),
        _build_fig2b(ctx),
        _build_fig2c(ctx),
        _build_fig2d(ctx),
    ]


def _fig2_sources() -> tuple[SourceDescriptor, SourceDescriptor, SourceDescriptor, SourceDescriptor]:
    return (
        SourceDescriptor(
            key="fig2.rollouts",
            pattern=(
                "results/multi_seed_rollout/fig2/fixed_b_mechanism_confirmatory/"
                "seed_*/data/intermediates/fixed_b_rollout_bank/rollout_rows.csv"
            ),
            source_level="validated_artifact",
            producer_task="fixed-B rollout bank",
            filters="track=stsp_isolated; branch=free; prefix_k=1",
            held_fixed="exact B per network_seed x b_anchor_id; seeds=1000-1019",
            aggregation_path="history rollout -> anchor opportunity -> network rate",
            required_columns=(
                "network_seed",
                "rollout_row_id",
                "prefix_k",
                "history_row_id",
                "history_family_id",
                "history_condition",
                "b_anchor_id",
                "B_label",
                "track",
                "branch",
                "exact_b_tensor_sha256",
                "prediction",
            ),
        ),
        SourceDescriptor(
            key="fig2.history_bank",
            pattern=(
                "results/multi_seed_rollout/fig2/fixed_b_mechanism_confirmatory/"
                "seed_*/data/intermediates/fixed_b_history_bank/history_specs.csv"
            ),
            source_level="validated_artifact",
            producer_task="fixed-B history bank",
            filters="prefix_k=1; history_condition in A,C,S0",
            held_fixed="frozen history families",
            aggregation_path="history spec -> history-content relation",
            required_columns=(
                "history_row_id",
                "history_family_id",
                "history_condition",
                "prefix_k",
                "sequence_labels",
            ),
        ),
        SourceDescriptor(
            key="fig2.trajectory",
            pattern=(
                "results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/"
                "seed_*/data/raw/fixed_b_state_trajectory_rows.csv"
            ),
            source_level="raw",
            producer_task="fixed-B state trajectory",
            filters="track=stsp_isolated; branch=free; prefix_k=1",
            held_fixed="rollout_row_id must be represented in trajectory checkpoints",
            aggregation_path="trajectory identity audit only",
            required_columns=(
                "network_seed",
                "rollout_row_id",
                "prefix_k",
                "history_row_id",
                "history_family_id",
                "history_condition",
                "b_anchor_id",
                "B_label",
                "track",
                "branch",
            ),
        ),
        SourceDescriptor(
            key="fig2.trial_specs",
            pattern=(
                "results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/"
                "seed_*/data/trial_specs/fixed_b_history_specs.csv"
            ),
            source_level="validated_artifact",
            producer_task="fixed-B trial specs",
            filters="prefix_k=1",
            held_fixed="must match history-bank protocol rows",
            aggregation_path="trial-spec identity audit only",
            required_columns=(
                "history_row_id",
                "history_family_id",
                "history_condition",
                "prefix_k",
                "sequence_labels",
            ),
        ),
    )


def _build_fig2b(ctx: BuilderContext) -> PanelResult:
    panel_id = "b"
    rollout_desc, history_desc, trajectory_desc, trial_desc = _fig2_sources()
    rollout = _load(ctx, panel_id, rollout_desc)
    histories = _load(ctx, panel_id, history_desc)
    trajectory = _load(ctx, panel_id, trajectory_desc)
    trial_specs = _load(ctx, panel_id, trial_desc)
    rows = rollout.frame.loc[
        rollout.frame["track"].eq("stsp_isolated")
        & rollout.frame["branch"].eq("free")
        & rollout.frame["prefix_k"].eq(1)
    ].copy()
    if rows.empty:
        raise ValueError("fig2b: no rows remain after frozen filters")
    anchor_key = ["network_seed", "b_anchor_id"]
    s0 = rows.loc[rows["history_condition"].eq("S0")].copy()
    for column in ("prediction", "B_label", "exact_b_tensor_sha256"):
        counts = s0.groupby(anchor_key)[column].nunique(dropna=False)
        if not counts.eq(1).all():
            bad = counts.loc[~counts.eq(1)].index.tolist()[:10]
            raise ValueError(f"fig2b: repeated S0 {column} inconsistent for anchors {bad}")
    for column in ("B_label", "exact_b_tensor_sha256"):
        counts = rows.groupby(anchor_key)[column].nunique(dropna=False)
        if not counts.eq(1).all():
            bad = counts.loc[~counts.eq(1)].index.tolist()[:10]
            raise ValueError(f"fig2b: exact-B {column} inconsistent for anchors {bad}")
    s0_anchor = (
        s0.groupby(anchor_key, as_index=False)
        .agg(S0_prediction=("prediction", "first"), B_label=("B_label", "first"))
    )
    history_rows = histories.frame.loc[
        histories.frame["prefix_k"].eq(1)
        & histories.frame["history_condition"].isin(["A", "C"])
    ].copy()
    history_rows["history_label"] = history_rows["sequence_labels"].map(_last_label)
    history_key = [
        "network_seed",
        "history_row_id",
        "history_family_id",
        "history_condition",
    ]
    work = rows.loc[rows["history_condition"].isin(["A", "C"])].merge(
        history_rows[history_key + ["history_label"]],
        on=history_key,
        how="left",
        validate="many_to_one",
    )
    if work["history_label"].isna().any():
        raise ValueError("fig2b: history labels failed to join")
    work = work.merge(
        s0_anchor,
        on=anchor_key,
        how="left",
        validate="many_to_one",
        suffixes=("", "_s0"),
    )
    work["history_relation"] = np.where(
        work["history_label"].astype(int).eq(work["B_label"].astype(int)),
        "aligned",
        "mismatched",
    )
    work["S0_correct"] = work["S0_prediction"].eq(work["B_label"])
    work["history_correct"] = work["prediction"].eq(work["B_label"])
    _validate_fig2_auxiliary_sources(rows, histories.frame, trajectory.frame, trial_specs.frame)
    records: list[dict[str, Any]] = []
    for (network_seed, outcome, relation), subset in _fig2_opportunity_rows(work).groupby(
        ["network_seed", "outcome_type", "history_relation"],
        sort=False,
    ):
        records.append(
            {
                "network_seed": int(network_seed),
                "outcome_type": str(outcome),
                "history_relation": str(relation),
                "rate_percent": float(subset["event"].mean() * 100.0),
                "eligible_anchors": int(subset["b_anchor_id"].nunique()),
                "history_rows": int(len(subset)),
            }
        )
    network = pd.DataFrame(records)
    expected_rows = len(EXPECTED_SEEDS) * 4
    if len(network) != expected_rows:
        raise ValueError(f"fig2b: expected {expected_rows} network rows, observed {len(network)}")
    plot = make_plot_data(
        network,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_opportunity_rate",
        endpoint="outcome_type",
        condition="history_relation",
        value="rate_percent",
        unit="percent",
        dimensions=("outcome_type", "history_relation", "eligible_anchors", "history_rows"),
    )
    plot["endpoint"] = network["outcome_type"].to_numpy()
    plot["condition"] = network["history_relation"].to_numpy()
    plot = plot.sort_values(
        ["network_seed", "outcome_type", "history_relation"],
        kind="mergesort",
    )
    descriptive = _statistics_values(
        plot,
        group_columns=("outcome_type", "history_relation"),
        status="descriptive_only",
        null_by_endpoint={"rescue": 0.0, "loss": 0.0},
    )
    contrast_values: list[pd.DataFrame] = []
    for outcome in ("rescue", "loss"):
        pivot = plot.loc[plot["outcome_type"].eq(outcome)].pivot(
            index="network_seed",
            columns="history_relation",
            values="value",
        )
        contrast = (
            pivot["aligned"] - pivot["mismatched"]
        ).rename("value").reset_index()
        contrast_values.append(
            _contrast_statistics_values(
                contrast,
                endpoint=outcome,
                contrast="aligned_minus_mismatched",
                unit="percentage_points",
                p_adjust_family="fig2b_behavioral_outcomes",
            )
        )
    statistics = build_statistics(
        pd.concat([descriptive, *contrast_values], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    all_records = (
        rollout.records + histories.records + trajectory.records + trial_specs.records
    )
    input_rows = sum(
        len(source.frame) for source in (rollout, histories, trajectory, trial_specs)
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        all_records,
        input_rows=input_rows,
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "outcome_type",
            "history_relation",
        ),
        exclusion_reason=(
            "K5, natural, replay, and S0 reference rows excluded from displayed rates; "
            "S0 rows retained only for opportunity eligibility"
        ),
    )


def _last_label(value: Any) -> int:
    labels = ast.literal_eval(str(value))
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"invalid non-empty sequence_labels: {value}")
    return int(labels[-1])


def _fig2_opportunity_rows(work: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for outcome in ("rescue", "loss"):
        eligible = ~work["S0_correct"] if outcome == "rescue" else work["S0_correct"]
        subset = work.loc[eligible].copy()
        subset["outcome_type"] = outcome
        subset["event"] = (
            subset["history_correct"] if outcome == "rescue" else ~subset["history_correct"]
        ).astype(float)
        outputs.append(subset)
    return pd.concat(outputs, ignore_index=True, sort=False)


def _validate_fig2_auxiliary_sources(
    rollout_rows: pd.DataFrame,
    histories: pd.DataFrame,
    trajectory: pd.DataFrame,
    trial_specs: pd.DataFrame,
) -> None:
    compare_columns = (
        "network_seed",
        "history_row_id",
        "history_family_id",
        "history_condition",
        "prefix_k",
        "sequence_labels",
    )
    left = (
        histories.loc[histories["prefix_k"].eq(1), compare_columns]
        .drop_duplicates()
        .sort_values(list(compare_columns), kind="mergesort")
        .reset_index(drop=True)
    )
    right = (
        trial_specs.loc[trial_specs["prefix_k"].eq(1), compare_columns]
        .drop_duplicates()
        .sort_values(list(compare_columns), kind="mergesort")
        .reset_index(drop=True)
    )
    if not left.equals(right):
        raise ValueError("fig2b: history bank and trial-spec protocol rows differ")
    trajectory_filtered = trajectory.loc[
        trajectory["track"].eq("stsp_isolated")
        & trajectory["branch"].eq("free")
        & trajectory["prefix_k"].eq(1)
    ]
    expected_ids = set(pd.to_numeric(rollout_rows["rollout_row_id"], errors="raise").astype(int))
    observed_ids = set(
        pd.to_numeric(trajectory_filtered["rollout_row_id"], errors="raise").astype(int)
    )
    if expected_ids != observed_ids:
        raise ValueError(
            "fig2b: filtered rollout/trajectory row identities differ; "
            f"missing={sorted(expected_ids-observed_ids)[:10]}, "
            f"extra={sorted(observed_ids-expected_ids)[:10]}"
        )
    key = [
        "network_seed",
        "rollout_row_id",
        "history_row_id",
        "history_family_id",
        "history_condition",
        "b_anchor_id",
        "B_label",
    ]
    rollout_identity = rollout_rows.loc[:, key].drop_duplicates()
    trajectory_identity = trajectory_filtered.loc[:, key].drop_duplicates()
    merged = rollout_identity.merge(
        trajectory_identity,
        on=key,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("fig2b: rollout/trajectory scientific identity mismatch")


def _fixed_scalar_source(ctx: BuilderContext, panel_id: str) -> LoadedSource:
    return _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key=f"{ctx.figure_id}.{panel_id}.fixed_scalars",
            pattern=(
                "results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/"
                "aggregate/fixed_b_confirmatory_network_scalars.csv"
            ),
            source_level="validated_artifact",
            producer_task="fixed-B confirmatory aggregate",
            filters="frozen endpoint and prefix filters",
            held_fixed="network_seed=1000-1019",
            aggregation_path="validated per-network scalar",
            seeded=False,
            required_columns=(
                "network_seed",
                "family",
                "endpoint",
                "prefix_k",
                "value",
                "role",
                "threshold",
            ),
        ),
    )


def _fig2_event_component_source(
    ctx: BuilderContext,
    panel_id: str,
) -> LoadedSource:
    return _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key=f"{ctx.figure_id}.{panel_id}.event_components",
            pattern=(
                "results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/"
                "seed_*/data/metrics/fixed_b_event_gamma_cell_metrics.csv"
            ),
            source_level="validated_artifact",
            producer_task="fixed-B event-Gamma analysis",
            filters="prefix_k=1; valid=1",
            held_fixed=(
                "matched-random coordinates use the same count as changed-event "
                "coordinates within each analysis cell; network_seed=1000-1019"
            ),
            aggregation_path=(
                "valid analysis cell -> condition mean within network -> paired "
                "network comparison"
            ),
            required_columns=(
                "network_seed",
                "prefix_k",
                "changed_coordinate_gamma_mean_abs",
                "matched_random_gamma_mean_abs",
                "valid",
            ),
        ),
    )


def _build_fig2c(ctx: BuilderContext) -> PanelResult:
    panel_id = "c"
    source = _fixed_scalar_source(ctx, panel_id)
    source_endpoints = (
        "same_B_common_update_cosine",
        "processing_residual_gamma_energy_fraction",
    )
    selected = source.frame.loc[
        source.frame["prefix_k"].eq(1)
        & source.frame["endpoint"].isin(source_endpoints)
    ].copy()
    threshold_by_endpoint = (
        selected.groupby("endpoint")["threshold"].first().astype(float).to_dict()
    )
    expected_source_thresholds = {
        "same_B_common_update_cosine": 0.5,
        "processing_residual_gamma_energy_fraction": 0.05,
    }
    for endpoint, expected in expected_source_thresholds.items():
        observed = float(threshold_by_endpoint.get(endpoint, math.nan))
        if not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"fig2c: endpoint {endpoint} threshold {observed} != frozen {expected}"
            )
    selected["source_endpoint"] = selected["endpoint"].astype(str)
    selected["display_endpoint"] = selected["source_endpoint"].replace(
        {
            "processing_residual_gamma_energy_fraction": (
                "processing_residual_gamma_norm_ratio"
            )
        }
    )
    plot = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_metric",
        endpoint="display_endpoint",
        condition="prefix_k1",
        value="value",
        unit="dimensionless",
        dimensions=("prefix_k", "source_endpoint"),
    )
    plot["endpoint"] = selected["display_endpoint"].to_numpy()
    plot = plot.sort_values(["network_seed", "endpoint"], kind="mergesort")
    expected_thresholds = {
        "same_B_common_update_cosine": 0.5,
        "processing_residual_gamma_norm_ratio": 0.05,
    }
    values = _statistics_values(
        plot,
        group_columns=("condition",),
        status="predeclared_recomputed",
        null_by_endpoint=expected_thresholds,
        p_adjust_family="fig2c_predeclared_thresholds",
    )
    values["contrast"] = "estimate_minus_predeclared_threshold"
    statistics = build_statistics(values, figure_id=ctx.figure_id, panel_id=panel_id)
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=("figure_id", "panel_id", "network_seed", "endpoint"),
        exclusion_reason="K5 and unrelated fixed-B endpoints excluded",
    )


def _build_fig2d(ctx: BuilderContext) -> PanelResult:
    panel_id = "d"
    source = _fig2_event_component_source(ctx, panel_id)
    selected = source.frame.loc[
        source.frame["prefix_k"].eq(1)
        & source.frame["valid"].astype(bool)
    ].copy()
    if selected.empty:
        raise ValueError("fig2d: no valid K1 event-analysis cells remain")
    network = selected.groupby("network_seed", as_index=False)[
        [
            "changed_coordinate_gamma_mean_abs",
            "matched_random_gamma_mean_abs",
        ]
    ].mean()
    network["prefix_k"] = 1
    long = network.melt(
        id_vars=("network_seed", "prefix_k"),
        value_vars=(
            "matched_random_gamma_mean_abs",
            "changed_coordinate_gamma_mean_abs",
        ),
        var_name="source_condition",
        value_name="residual_magnitude",
    )
    long["condition_name"] = long["source_condition"].map(
        {
            "matched_random_gamma_mean_abs": "matched_random",
            "changed_coordinate_gamma_mean_abs": "changed_events",
        }
    )
    plot = make_plot_data(
        long,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_condition",
        endpoint="residual_magnitude",
        condition="condition_name",
        value="residual_magnitude",
        unit="mean_absolute_residual",
        dimensions=("prefix_k", "source_condition"),
    )
    plot["condition"] = long["condition_name"].to_numpy()
    plot = plot.sort_values(["network_seed", "condition"], kind="mergesort")
    descriptive = _statistics_values(
        plot,
        group_columns=("condition",),
        status="descriptive_only",
    )
    paired = network.loc[:, ["network_seed"]].copy()
    paired["value"] = (
        pd.to_numeric(
            network["changed_coordinate_gamma_mean_abs"], errors="coerce"
        )
        - pd.to_numeric(network["matched_random_gamma_mean_abs"], errors="coerce")
    )
    inference = _contrast_statistics_values(
        paired,
        endpoint="residual_magnitude",
        contrast="changed_events_minus_matched_random",
        unit="mean_absolute_residual",
        p_adjust_family="fig2d_event_residual",
    )
    statistics = build_statistics(
        pd.concat([descriptive, inference], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "condition",
        ),
        exclusion_reason=(
            "K5 and invalid event-analysis cells excluded; changed-event and "
            "within-cell size-matched-random magnitudes are displayed directly, "
            "with no ratio or fold transformation"
        ),
    )


def _build_fig2e(ctx: BuilderContext) -> PanelResult:
    panel_id = "e"
    source = _fixed_scalar_source(ctx, panel_id)
    endpoints = (
        "layer1_only_layer2_update_donor_transfer",
        "layer1_only_early_class_score_donor_transfer",
    )
    selected = source.frame.loc[
        source.frame["prefix_k"].eq(1) & source.frame["endpoint"].isin(endpoints)
    ].copy()
    plot = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_contrast",
        endpoint="endpoint",
        condition="layer1_only",
        value="value",
        unit="donor_transfer_index",
        dimensions=("prefix_k",),
    )
    plot["endpoint"] = selected["endpoint"].to_numpy()
    plot = plot.sort_values(["network_seed", "endpoint"], kind="mergesort")
    values = _statistics_values(
        plot,
        group_columns=("condition",),
        status="predeclared_recomputed",
        null_by_endpoint={endpoint: 0.0 for endpoint in endpoints},
        p_adjust_family="fig2e_donor_transfer",
    )
    values["contrast"] = "layer1_only_donor_transfer_vs_zero"
    statistics = build_statistics(values, figure_id=ctx.figure_id, panel_id=panel_id)
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=("figure_id", "panel_id", "network_seed", "endpoint"),
        exclusion_reason="K5 and all-layer plumbing controls excluded",
    )


def build_fig3(ctx: BuilderContext) -> list[PanelResult]:
    return [
        _build_fig3a(ctx),
        _build_fig3b(ctx),
        _build_fig3c(ctx),
        _build_fig3d_necessity(ctx),
        _build_fig3e(ctx),
        _build_fig3f_successor(ctx),
        _asset_panel(
            ctx,
            "g",
            relative_path="src/plotting/paper_fig/assets/fig3_state_evolution.svg",
            role="Fig.3-end conceptual synthesis of iterative inherited-state updating",
            semantics=(
                "illustrative state-transition summary; quantitative evidence remains "
                "in Fig.3a-f"
            ),
            allowed_cleanup=(
                "typography, spacing, rounded-card styling, and embedded stimulus rendering"
            ),
        ),
    ]


def _build_fig3a(ctx: BuilderContext) -> PanelResult:
    panel_id = "a"
    source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig3.overlap_perturbation",
            pattern=(
                "results/paper_figure_multi_seed/fig4_overlap_reentry/seed_*/"
                "data/metrics/panel_d_l1_stsp_overlap_perturbation_contrast.csv"
            ),
            source_level="network_metric",
            producer_task="overlap-specific L1 STSP reset",
            filters=(
                "endpoints=dynamic_minus_overlap_reset,"
                "nonoverlap_reset_minus_overlap_reset,"
                "random_reset_minus_overlap_reset"
            ),
            held_fixed="Layer 1 STSP; paired accuracy-drop contrasts",
            aggregation_path="paired trials -> network accuracy-drop contrast",
            required_columns=(
                "network_seed",
                "dynamic_minus_overlap_reset",
                "nonoverlap_reset_minus_overlap_reset",
                "random_reset_minus_overlap_reset",
            ),
        ),
    )
    long = source.frame.melt(
        id_vars=("network_seed",),
        value_vars=(
            "dynamic_minus_overlap_reset",
            "nonoverlap_reset_minus_overlap_reset",
            "random_reset_minus_overlap_reset",
        ),
        var_name="endpoint_name",
        value_name="contrast_value",
    )
    long["contrast_pp"] = pd.to_numeric(long["contrast_value"], errors="coerce") * 100.0
    plot = make_plot_data(
        long,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_contrast",
        endpoint="endpoint_name",
        condition="overlap_reset",
        value="contrast_pp",
        unit="percentage_points",
    )
    plot["endpoint"] = long["endpoint_name"].to_numpy()
    plot = plot.sort_values(["network_seed", "endpoint"], kind="mergesort")
    values = _statistics_values(
        plot,
        group_columns=("condition",),
        status="predeclared_recomputed",
        null_by_endpoint={
            "dynamic_minus_overlap_reset": 0.0,
            "nonoverlap_reset_minus_overlap_reset": 0.0,
            "random_reset_minus_overlap_reset": 0.0,
        },
        p_adjust_family="fig3a_overlap_reset",
    )
    values["contrast"] = values["endpoint"]
    statistics = build_statistics(values, figure_id=ctx.figure_id, panel_id=panel_id)
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=("figure_id", "panel_id", "network_seed", "endpoint"),
        exclusion_reason="absolute accuracy-drop columns retained in source audit",
    )


def _competition_source(
    ctx: BuilderContext,
    panel_id: str,
    *,
    key: str,
    filename: str,
    source_level: str,
    filters: str,
    held_fixed: str,
    aggregation_path: str,
    required_columns: Sequence[str],
) -> LoadedSource:
    return _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key=key,
            pattern=(
                "results/paper_figure_multi_seed/fig5_local_support_competition/"
                f"seed_*/data/metrics/{filename}"
            ),
            source_level=source_level,
            producer_task="local-support/competition",
            filters=filters,
            held_fixed=held_fixed,
            aggregation_path=aggregation_path,
            required_columns=required_columns,
        ),
    )


def _build_fig3b(ctx: BuilderContext) -> PanelResult:
    panel_id = "b"
    source = _competition_source(
        ctx,
        panel_id,
        key="fig3.preprobe_support",
        filename="panel_a_preprobe_support_metrics.csv",
        source_level="trial",
        filters="layer=layer1; state_variable=g; four frozen unit groups",
        held_fixed="pre-input support; no trial or unit pseudoreplication",
        aggregation_path="units -> trial group mean -> network group mean",
        required_columns=(
            "network_seed",
            "trial_id",
            "unit_group",
            "layer",
            "state_variable",
            "mean_support",
        ),
    )
    selected = source.frame.loc[
        source.frame["layer"].eq("layer1")
        & source.frame["state_variable"].eq("g")
        & source.frame["unit_group"].isin(GROUP_ORDER)
    ].copy()
    network = (
        selected.groupby(["network_seed", "unit_group"], as_index=False)["mean_support"]
        .mean()
        .rename(columns={"mean_support": "preprobe_mean_support"})
    )
    plot = make_plot_data(
        network,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_metric",
        endpoint="preprobe_mean_support",
        condition="unit_group",
        value="preprobe_mean_support",
        unit="support_index",
        dimensions=("unit_group",),
    )
    plot["condition"] = network["unit_group"].to_numpy()
    plot = plot.sort_values(["network_seed", "unit_group"], kind="mergesort")
    descriptive = _statistics_values(
        plot,
        group_columns=("unit_group",),
        status="descriptive_only",
    )
    pivot = plot.pivot(index="network_seed", columns="unit_group", values="value")
    contrasts: list[pd.DataFrame] = []
    for control in GROUP_ORDER[1:]:
        values = (pivot["overlap_dominant"] - pivot[control]).rename("value").reset_index()
        contrasts.append(
            _contrast_statistics_values(
                values,
                endpoint="preprobe_mean_support",
                contrast=f"overlap_dominant_minus_{control}",
                unit="support_index",
                p_adjust_family="fig3b_support_controls",
            )
        )
    statistics = build_statistics(
        pd.concat([descriptive, *contrasts], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "unit_group",
        ),
    )


def _build_fig3c(ctx: BuilderContext) -> PanelResult:
    panel_id = "c"
    source = _competition_source(
        ctx,
        panel_id,
        key="fig3.transitions_30ms",
        filename="panel_b_early_firing_transition_metrics.csv",
        source_level="unit",
        filters=(
            "first_spike_dynamic/static restricted to [0,30) ms; "
            "endpoints=P_advance,P_recruit,P_loss; "
            "groups=overlap_dominant,probe_only_dominant,random_matched"
        ),
        held_fixed=(
            "dt=1 ms; descriptive 30-ms window; balanced and unchanged "
            "excluded from the main panel"
        ),
        aggregation_path=(
            "unit first-spike pair -> trial transition probability -> "
            "network group probability"
        ),
        required_columns=(
            "network_seed",
            "trial_id",
            "unit_group",
            "first_spike_dynamic",
            "first_spike_static",
        ),
    )
    display_groups = (
        "overlap_dominant",
        "probe_only_dominant",
        "random_matched",
    )
    selected = source.frame.loc[
        source.frame["unit_group"].isin(display_groups)
    ].copy()
    dynamic = pd.to_numeric(
        selected["first_spike_dynamic"],
        errors="coerce",
    ).fillna(-1)
    static = pd.to_numeric(
        selected["first_spike_static"],
        errors="coerce",
    ).fillna(-1)
    dynamic = dynamic.where(dynamic.ge(0) & dynamic.lt(30), -1)
    static = static.where(static.ge(0) & static.lt(30), -1)
    transitions = pd.Series(
        "unchanged",
        index=selected.index,
        dtype=object,
    )
    transitions.loc[
        dynamic.ge(0) & static.ge(0) & dynamic.lt(static)
    ] = "advance"
    transitions.loc[dynamic.ge(0) & static.lt(0)] = "recruit"
    transitions.loc[dynamic.lt(0) & static.ge(0)] = "loss"
    selected["P_advance"] = transitions.eq("advance").astype(float)
    selected["P_recruit"] = transitions.eq("recruit").astype(float)
    selected["P_loss"] = transitions.eq("loss").astype(float)
    trial = selected.groupby(
        ["network_seed", "trial_id", "unit_group"],
        as_index=False,
    )[["P_advance", "P_recruit", "P_loss"]].mean()
    network = (
        trial.groupby(
            ["network_seed", "unit_group"],
            as_index=False,
        )[["P_advance", "P_recruit", "P_loss"]]
        .mean()
        .melt(
            id_vars=("network_seed", "unit_group"),
            value_vars=("P_advance", "P_recruit", "P_loss"),
            var_name="endpoint_name",
            value_name="probability",
        )
    )
    network["early_window_ms"] = 30
    network["probability_percent"] = (
        pd.to_numeric(network["probability"], errors="coerce") * 100.0
    )
    plot = make_plot_data(
        network,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_metric",
        endpoint="endpoint_name",
        condition="unit_group",
        value="probability_percent",
        unit="percent",
        dimensions=("unit_group", "early_window_ms"),
    )
    plot["endpoint"] = network["endpoint_name"].to_numpy()
    plot["condition"] = network["unit_group"].to_numpy()
    plot = plot.sort_values(
        ["network_seed", "endpoint", "unit_group"],
        kind="mergesort",
    )
    descriptive = _statistics_values(
        plot,
        group_columns=("unit_group", "early_window_ms"),
        status="descriptive_only",
    )
    statistics = build_statistics(
        descriptive,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "unit_group",
            "early_window_ms",
        ),
        exclusion_reason=(
            "spikes at or after 30 ms, the balanced group, and P_unchanged "
            "excluded from the main-panel plot data"
        ),
    )


def _build_fig3d_necessity(ctx: BuilderContext) -> PanelResult:
    panel_id = "d"
    descriptor = SourceDescriptor(
        key="fig3.l1_stsp_unit_transitions",
        pattern=(
            "results/paper_figure_multi_seed/fig5_local_support_competition/"
            "seed_*/data/metrics/panel_d_l1_stsp_perturbation_unit_transitions.csv"
        ),
        source_level="raw",
        producer_task="Layer 1 STSP attenuation/reset",
        filters=(
            "included_in_main=true; condition in dynamic_intact,attenuate_l1_stsp,"
            "reset_l1_stsp; first_spike cropped to 0..49 ms"
        ),
        held_fixed="primary first-50-ms advance-or-recruit endpoint",
        aggregation_path="unit transition -> trial probability -> network probability -> paired contrast",
        required_columns=(
            "network_seed",
            "trial_id",
            "condition",
            "included_in_main",
            "first_spike_static",
            "first_spike_condition",
        ),
    )
    paths = resolve_source_paths(ctx.repo_root, descriptor)
    source_records: list[dict[str, Any]] = []
    network_condition_rows: list[dict[str, Any]] = []
    total_input_rows = 0
    for path in paths:
        source_records.append(
            record_file_source(
                repo_root=ctx.repo_root,
                figure_id=ctx.figure_id,
                panel_id=panel_id,
                descriptor=descriptor,
                path=path,
                input_rows=_csv_row_count(path),
            )
        )
        total_input_rows += int(source_records[-1]["input_rows"])
        partials: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            path,
            usecols=(
                "network_seed",
                "trial_id",
                "condition",
                "included_in_main",
                "first_spike_static",
                "first_spike_condition",
            ),
            chunksize=250_000,
            low_memory=False,
        ):
            selected = chunk.loc[
                chunk["included_in_main"].astype(bool)
                & chunk["condition"].isin(
                    ["dynamic_intact", "attenuate_l1_stsp", "reset_l1_stsp"]
                )
            ].copy()
            static = pd.to_numeric(selected["first_spike_static"], errors="coerce")
            condition = pd.to_numeric(selected["first_spike_condition"], errors="coerce")
            static_in_window = static.ge(0) & static.lt(50)
            condition_in_window = condition.ge(0) & condition.lt(50)
            selected["advance_or_recruit"] = (
                condition_in_window & (~static_in_window | condition.lt(static))
            ).astype(int)
            partial = (
                selected.groupby(["network_seed", "trial_id", "condition"])[
                    "advance_or_recruit"
                ]
                .agg(sum="sum", count="count")
                .reset_index()
            )
            partials.append(partial)
        if not partials:
            raise ValueError(f"fig3d: no included rows in {path}")
        trial_counts = (
            pd.concat(partials, ignore_index=True)
            .groupby(["network_seed", "trial_id", "condition"], as_index=False)[["sum", "count"]]
            .sum()
        )
        trial_counts["trial_probability"] = trial_counts["sum"] / trial_counts["count"]
        per_network = (
            trial_counts.groupby(["network_seed", "condition"], as_index=False)[
                "trial_probability"
            ]
            .mean()
        )
        network_condition_rows.extend(per_network.to_dict(orient="records"))
    condition_frame = pd.DataFrame(network_condition_rows)
    pivot = condition_frame.pivot(
        index="network_seed",
        columns="condition",
        values="trial_probability",
    )
    network = pd.DataFrame(
        {
            "network_seed": pivot.index.astype(int),
            "dynamic_minus_attenuation": (
                pivot["dynamic_intact"] - pivot["attenuate_l1_stsp"]
            ).to_numpy(dtype=float)
            * 100.0,
            "dynamic_minus_reset": (
                pivot["dynamic_intact"] - pivot["reset_l1_stsp"]
            ).to_numpy(dtype=float)
            * 100.0,
        }
    )
    long = network.melt(
        id_vars=("network_seed",),
        value_vars=("dynamic_minus_attenuation", "dynamic_minus_reset"),
        var_name="endpoint_name",
        value_name="contrast_pp",
    )
    plot = make_plot_data(
        long,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_contrast",
        endpoint="endpoint_name",
        condition="first_50_ms",
        value="contrast_pp",
        unit="percentage_points",
    )
    plot["endpoint"] = long["endpoint_name"].to_numpy()
    plot["time_window_ms"] = 50
    plot = plot.sort_values(["network_seed", "endpoint"], kind="mergesort")
    values = _statistics_values(
        plot,
        group_columns=("condition",),
        status="predeclared_recomputed",
        null_by_endpoint={
            "dynamic_minus_attenuation": 0.0,
            "dynamic_minus_reset": 0.0,
        },
        p_adjust_family="fig3d_stsp_necessity",
    )
    values["contrast"] = values["endpoint"]
    statistics = build_statistics(values, figure_id=ctx.figure_id, panel_id=panel_id)
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source_records,
        input_rows=total_input_rows,
        unique_key=("figure_id", "panel_id", "network_seed", "endpoint"),
        exclusion_reason=(
            "rows outside included unit groups excluded; first spikes at >=50 ms treated "
            "as absent for the frozen primary window"
        ),
    )


def _csv_row_count(path: Path) -> int:
    lines = 0
    last = b""
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            lines += block.count(b"\n")
            last = block[-1:]
    if path.stat().st_size and last != b"\n":
        lines += 1
    return max(0, lines - 1)


def _build_fig3e(ctx: BuilderContext) -> PanelResult:
    panel_id = "e"
    source = _competition_source(
        ctx,
        panel_id,
        key="fig3.l2_writeback",
        filename="panel_postprobe_l2_reupdate_history_composition.csv",
        source_level="network_metric",
        filters=(
            "condition in dynamic_intact,static_opportunity; "
            "history_status in prior-updated,not-prior-updated"
        ),
        held_fixed="static values are update opportunity, not mutation",
        aggregation_path="L2 sites -> network conditional update probability and DID",
        required_columns=(
            "network_seed",
            "condition",
            "history_status",
            "update_probability_given_history",
            "conditional_difference_in_differences",
        ),
    )
    selected = source.frame.loc[
        source.frame["condition"].isin(["dynamic_intact", "static_opportunity"])
    ].copy()
    selected["probability_percent"] = (
        pd.to_numeric(selected["update_probability_given_history"], errors="coerce")
        * 100.0
    )
    cells = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_probability",
        endpoint="l2_update_probability",
        condition="condition",
        value="probability_percent",
        unit="percent",
        dimensions=("history_status",),
    )
    cells["condition"] = selected["condition"].to_numpy()
    did_counts = selected.groupby("network_seed")[
        "conditional_difference_in_differences"
    ].nunique(dropna=False)
    if not did_counts.eq(1).all():
        raise ValueError("fig3e: DID is inconsistent across source cell rows")
    did = (
        selected.groupby("network_seed", as_index=False)[
            "conditional_difference_in_differences"
        ]
        .first()
        .rename(columns={"conditional_difference_in_differences": "did"})
    )
    did["did_pp"] = pd.to_numeric(did["did"], errors="coerce") * 100.0
    did_plot = make_plot_data(
        did,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_contrast",
        endpoint="dynamic_minus_static_difference_in_differences",
        condition="did",
        value="did_pp",
        unit="percentage_points",
    )
    plot = pd.concat([cells, did_plot], ignore_index=True, sort=False).sort_values(
        ["network_seed", "record_type", "condition", "history_status"],
        kind="mergesort",
        na_position="last",
    )
    descriptive = _statistics_values(
        cells,
        group_columns=("condition", "history_status"),
        status="descriptive_only",
    )
    inference = _statistics_values(
        did_plot,
        group_columns=("condition",),
        status="predeclared_recomputed",
        null_by_endpoint={"dynamic_minus_static_difference_in_differences": 0.0},
    )
    inference["contrast"] = "dynamic_minus_static_difference_in_differences"
    statistics = build_statistics(
        pd.concat([descriptive, inference], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "record_type",
            "endpoint",
            "condition",
            "history_status",
        ),
    )


def _build_fig3f_successor(ctx: BuilderContext) -> PanelResult:
    panel_id = "f"
    source = _fixed_scalar_source(ctx, panel_id)
    endpoint = "layer1_only_layer2_update_donor_transfer"
    selected = source.frame.loc[
        source.frame["prefix_k"].eq(1)
        & source.frame["endpoint"].eq(endpoint)
    ].copy()
    if len(selected) != len(EXPECTED_SEEDS):
        raise ValueError(
            "fig3f: the K1 Layer-1-only to Layer-2 donor-transfer endpoint "
            f"must contain one value per network; observed={len(selected)}"
        )
    plot = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_contrast",
        endpoint=endpoint,
        condition="layer1_only_ux_swap",
        value="value",
        unit="donor_transfer_index",
        dimensions=("prefix_k",),
    ).sort_values(["network_seed"], kind="mergesort")
    values = _statistics_values(
        plot,
        group_columns=("condition",),
        status="predeclared_recomputed",
        null_by_endpoint={endpoint: 0.0},
        p_adjust_family="fig3f_successor_formation",
    )
    values["contrast"] = "layer1_only_layer2_successor_transfer_vs_zero"
    statistics = build_statistics(
        values,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
        ),
        exclusion_reason=(
            "K5, early Layer-3 consequences, all-layer swaps, and robustness "
            "endpoints remain outside the main panel"
        ),
    )


FIG4_ACCUMULATED_ROOT = (
    "results/paper_figure_multi_seed/fig4_accumulated_history_statistics"
)


def _fig4_accumulated_source(
    ctx: BuilderContext,
    panel_id: str,
    *,
    subdir: str,
    filename: str,
    required_columns: Sequence[str],
    filters: str,
    aggregation_path: str,
) -> LoadedSource:
    return _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key=f"fig4.{panel_id}.accumulated_history.{filename}",
            pattern=f"{FIG4_ACCUMULATED_ROOT}/{subdir}/{filename}",
            source_level="validated_artifact",
            producer_task="load-only accumulated-history Fig.4 statistics",
            filters=filters,
            held_fixed=(
                "network_seed=1000-1019; 20,000-draw network bootstrap; "
                "candidate gates passed; excluded K5 relation contrasts remain audit-only"
            ),
            aggregation_path=aggregation_path,
            seeded=False,
            required_columns=required_columns,
        ),
    )


def _optional_number(row: pd.Series, field: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(field, pd.NA)]), errors="coerce").iloc[0]
    return float(value) if not pd.isna(value) else math.nan


def _candidate_descriptive_statistics(
    frame: pd.DataFrame,
    *,
    figure_id: str,
    panel_id: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record = {column: pd.NA for column in STATISTICS_COLUMNS}
        record.update(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "endpoint": str(row["endpoint"]),
                "contrast": "",
                "group": str(row["plot_group"]),
                "n_networks": int(row["n_networks"]),
                "estimate": _optional_number(row, "mean"),
                "mean": _optional_number(row, "mean"),
                "sd": _optional_number(row, "sd"),
                "sem": _optional_number(row, "sem"),
                "ci95_low": _optional_number(row, "ci95_low"),
                "ci95_high": _optional_number(row, "ci95_high"),
                "min": _optional_number(row, "minimum"),
                "max": _optional_number(row, "maximum"),
                "null_value": math.nan,
                "test_name": "",
                "p_adjust_method": "",
                "alternative": "",
                "unit": str(row["unit"]),
                "statistics_status": "supplied_descriptive_bootstrap",
                "source_file": str(row.get("source_file", "")),
                "interval_method": (
                    "20,000-draw percentile bootstrap of the network mean"
                ),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _candidate_inference_statistics(
    frame: pd.DataFrame,
    *,
    figure_id: str,
    panel_id: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        adjusted = _optional_number(row, "p_holm_all_new")
        if not np.isfinite(adjusted):
            adjusted = _optional_number(row, "p_holm_family")
        if not np.isfinite(adjusted):
            adjusted = _optional_number(row, "p_value")
        origin = str(row.get("inference_origin", ""))
        status = (
            "supplied_confirmatory_bootstrap"
            if origin == "supplied_confirmatory_inference"
            else "supplied_secondary_bootstrap"
        )
        record = {column: pd.NA for column in STATISTICS_COLUMNS}
        record.update(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "endpoint": str(row["endpoint"]),
                "contrast": str(row["endpoint"]),
                "group": str(row["plot_group"]),
                "n_networks": int(row["n_networks"]),
                "estimate": _optional_number(row, "mean"),
                "mean": _optional_number(row, "mean"),
                "sd": _optional_number(row, "sd"),
                "sem": _optional_number(row, "sem"),
                "ci95_low": _optional_number(row, "ci95_low"),
                "ci95_high": _optional_number(row, "ci95_high"),
                "null_value": _optional_number(row, "null_value"),
                "test_name": "exact_sign_flip",
                "p_value": _optional_number(row, "p_value"),
                "p_adjust_method": "Holm",
                "p_adjusted": adjusted,
                "alternative": str(row.get("alternative", "")),
                "unit": str(row["unit"]),
                "statistics_status": status,
                "claim_id": str(row.get("claim_id", "")),
                "inference_origin": origin,
                "n_above_null": _optional_number(row, "n_above_null"),
                "n_below_null": _optional_number(row, "n_below_null"),
                "p_holm_family": _optional_number(row, "p_holm_family"),
                "p_holm_all_new": _optional_number(row, "p_holm_all_new"),
                "method": str(row.get("method", "")),
                "source_file": str(row.get("source_file", "")),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _attach_persisted_summary(
    plot: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    keys: Sequence[str],
) -> pd.DataFrame:
    summary_columns = [*keys, "mean", "ci95_low", "ci95_high"]
    lookup = summary.loc[:, summary_columns].copy()
    if lookup.duplicated(list(keys)).any():
        duplicates = lookup.loc[
            lookup.duplicated(list(keys), keep=False), list(keys)
        ].to_dict("records")
        raise ValueError(f"duplicate Fig.4 persisted summaries: {duplicates}")
    lookup = lookup.rename(
        columns={
            "mean": "summary_mean",
            "ci95_low": "summary_ci95_low",
            "ci95_high": "summary_ci95_high",
        }
    )
    merged = plot.merge(lookup, on=list(keys), how="left", validate="many_to_one")
    summary_fields = ["summary_mean", "summary_ci95_low", "summary_ci95_high"]
    if merged[summary_fields].isna().any(axis=None):
        missing = merged.loc[merged[summary_fields].isna().any(axis=1), list(keys)]
        raise ValueError(
            f"Fig.4 plot rows lack persisted summaries: {missing.drop_duplicates().to_dict('records')}"
        )
    observed = merged.groupby(list(keys), as_index=False)["value"].mean()
    expected = merged.groupby(list(keys), as_index=False)["summary_mean"].first()
    checked = observed.merge(expected, on=list(keys), validate="one_to_one")
    if not np.allclose(
        checked["value"].to_numpy(dtype=float),
        checked["summary_mean"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Fig.4 persisted summaries disagree with network-level plot means")
    return merged


def _build_fig4a_accumulated(
    ctx: BuilderContext,
    *,
    panel_id: str = "a",
) -> PanelResult:
    data = _fig4_accumulated_source(
        ctx,
        panel_id,
        subdir="data",
        filename="fig4a_progressive_network_stage.csv",
        required_columns=(
            "network_seed",
            "stage_k",
            "observed_displacement",
            "passive_displacement",
            "observed_minus_passive",
        ),
        filters="stage_k=2..10; joint u/x observed and equal-time passive only",
        aggregation_path="sequence -> network x stage joint-state displacement",
    )
    descriptive = _fig4_accumulated_source(
        ctx,
        panel_id,
        subdir="metrics",
        filename="fig4_candidate_descriptive.csv",
        required_columns=(
            "panel",
            "endpoint",
            "condition",
            "n_networks",
            "mean",
            "ci95_low",
            "ci95_high",
        ),
        filters="panel=a; joint observed and passive stage summaries",
        aggregation_path="20 network values -> persisted bootstrap mean and CI",
    )
    inference = _fig4_accumulated_source(
        ctx,
        panel_id,
        subdir="metrics",
        filename="fig4_candidate_inference.csv",
        required_columns=(
            "panel",
            "claim_id",
            "endpoint",
            "n_networks",
            "mean",
            "ci95_low",
            "ci95_high",
            "p_value",
        ),
        filters="panel=a recurrence endpoints",
        aggregation_path="paired network contrasts -> exact sign-flip and Holm inference",
    )
    selected = data.frame.loc[data.frame["stage_k"].isin(range(2, 11))].copy()
    long = selected.melt(
        id_vars=("network_seed", "stage_k"),
        value_vars=("observed_displacement", "passive_displacement"),
        var_name="source_endpoint",
        value_name="displacement",
    ).reset_index(drop=True)
    long["condition"] = long["source_endpoint"].map(
        {"observed_displacement": "observed", "passive_displacement": "passive"}
    )
    long["endpoint_name"] = long["source_endpoint"].map(
        {
            "observed_displacement": "joint_state_displacement",
            "passive_displacement": "joint_passive_displacement",
        }
    )
    plot = make_plot_data(
        long,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_stage_condition",
        endpoint="endpoint_name",
        condition="condition",
        value="displacement",
        unit="cosine_distance",
        dimensions=("stage_k",),
    )
    plot["endpoint"] = long["endpoint_name"].to_numpy()
    plot["condition"] = long["condition"].to_numpy()
    desc = descriptive.frame.loc[
        descriptive.frame["panel"].eq("a")
        & descriptive.frame["endpoint"].isin(
            ["joint_state_displacement", "joint_passive_displacement"]
        )
    ].copy()
    desc["stage_k"] = (
        desc["condition"].astype(str).str.removeprefix("stage_").astype(int)
    )
    desc["plot_group"] = (
        desc["endpoint"].astype(str) + "|" + desc["stage_k"].astype(str)
    )
    plot = _attach_persisted_summary(
        plot,
        desc,
        keys=("endpoint", "stage_k"),
    ).sort_values(["network_seed", "condition", "stage_k"], kind="mergesort")
    statistics = _candidate_descriptive_statistics(
        desc,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    inference_rows = inference.frame.loc[inference.frame["panel"].eq("a")].copy()
    inference_rows["plot_group"] = inference_rows["endpoint"].astype(str)
    inference_statistics = _candidate_inference_statistics(
        inference_rows,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        [*data.records, *descriptive.records, *inference.records],
        input_rows=len(data.frame) + len(descriptive.frame) + len(inference.frame),
        unique_key=("figure_id", "panel_id", "network_seed", "condition", "stage_k"),
        extra_metrics={
            f"panel_{panel_id}_recurrence_inference.csv": inference_statistics
        },
        exclusion_reason="u-only and x-only trajectories are retained outside the main artwork",
    )


def _build_fig4b_accumulated(
    ctx: BuilderContext,
    *,
    panel_id: str = "b",
) -> PanelResult:
    data = _fig4_accumulated_source(
        ctx,
        panel_id,
        subdir="data",
        filename="fig4b_behavior_depth_network_rates.csv",
        required_columns=(
            "network_seed",
            "outcome_type",
            "prefix_k",
            "relation_balanced_rate_percent",
        ),
        filters="outcome in rescue,loss; prefix_k in 1,5; relations equally weighted",
        aggregation_path="eligible anchors -> relation-balanced network outcome rate",
    )
    descriptive = _fig4_accumulated_source(
        ctx,
        panel_id,
        subdir="metrics",
        filename="fig4_candidate_descriptive.csv",
        required_columns=(
            "panel",
            "endpoint",
            "condition",
            "n_networks",
            "mean",
            "ci95_low",
            "ci95_high",
        ),
        filters="panel=b; relation-balanced K1 and K5 level summaries",
        aggregation_path="20 network rates -> persisted bootstrap mean and CI",
    )
    inference = _fig4_accumulated_source(
        ctx,
        panel_id,
        subdir="metrics",
        filename="fig4_candidate_inference.csv",
        required_columns=(
            "panel",
            "claim_id",
            "endpoint",
            "n_networks",
            "mean",
            "ci95_low",
            "ci95_high",
            "p_value",
        ),
        filters="panel=b; depth shifts selected; relation contrasts retained for audit",
        aggregation_path="paired K5-minus-K1 network shifts -> exact sign-flip and Holm",
    )
    selected = data.frame.loc[
        data.frame["outcome_type"].isin(["rescue", "loss"])
        & data.frame["prefix_k"].isin([1, 5])
    ].copy().reset_index(drop=True)
    selected["prefix_label"] = selected["prefix_k"].map(lambda value: f"K{int(value)}")
    selected["endpoint_name"] = selected["outcome_type"].map(
        lambda value: f"{value}_relation_balanced_rate"
    )
    plot = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_depth_outcome_rate",
        endpoint="endpoint_name",
        condition="outcome_type",
        value="relation_balanced_rate_percent",
        unit="percent",
        dimensions=("prefix_label", "outcome_type"),
    )
    plot["endpoint"] = selected["endpoint_name"].to_numpy()
    plot["condition"] = selected["outcome_type"].to_numpy()
    plot = plot.rename(columns={"prefix_label": "prefix_k"})
    desc = descriptive.frame.loc[
        descriptive.frame["panel"].eq("b")
        & descriptive.frame["endpoint"].isin(
            ["rescue_relation_balanced_rate", "loss_relation_balanced_rate"]
        )
        & descriptive.frame["condition"].isin(["K1", "K5"])
    ].copy()
    desc["prefix_k"] = desc["condition"].astype(str)
    desc["plot_group"] = desc["endpoint"].astype(str) + "|" + desc["prefix_k"]
    plot = _attach_persisted_summary(
        plot,
        desc,
        keys=("endpoint", "prefix_k"),
    ).sort_values(["network_seed", "prefix_k", "outcome_type"], kind="mergesort")
    statistics = _candidate_descriptive_statistics(
        desc,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    all_inference = inference.frame.loc[inference.frame["panel"].eq("b")].copy()
    selected_endpoints = {
        "rescue_relation_balanced_K5_minus_K1",
        "loss_relation_balanced_K5_minus_K1",
    }
    depth_rows = all_inference.loc[
        all_inference["endpoint"].isin(selected_endpoints)
    ].copy()
    depth_rows["plot_group"] = depth_rows["endpoint"].astype(str)
    depth_statistics = _candidate_inference_statistics(
        depth_rows,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    audit = all_inference.loc[~all_inference["endpoint"].isin(selected_endpoints)].copy()
    audit["artwork_status"] = "established_in_fig2_not_repeated"
    audit.loc[
        audit["endpoint"].isin(
            [
                "K5_rescue_aligned_minus_mismatched",
                "K5_loss_aligned_minus_mismatched",
            ]
        ),
        "artwork_status",
    ] = "excluded_underpowered_k5_relation_contrast"
    audit.loc[
        audit["endpoint"].astype(str).str.contains("depth_by_relation_interaction"),
        "artwork_status",
    ] = "statistics_only_not_artwork"
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        [*data.records, *descriptive.records, *inference.records],
        input_rows=len(data.frame) + len(descriptive.frame) + len(inference.frame),
        unique_key=("figure_id", "panel_id", "network_seed", "prefix_k", "outcome_type"),
        extra_metrics={
            f"panel_{panel_id}_depth_inference.csv": depth_statistics,
            f"panel_{panel_id}_excluded_audit_statistics.csv": audit,
        },
        exclusion_reason=(
            "K5 relation contrasts and depth-by-relation interactions are excluded from artwork"
        ),
    )



C5_ROOT = "results/causal_closure_multi_seed_20260803/c5_l2_successor"


def _load_c5_population_sources(
    ctx: BuilderContext,
    panel_id: str,
) -> tuple[LoadedSource, LoadedSource]:
    effects = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key=f"fig4.{panel_id}.c5_network_effects",
            pattern=f"{C5_ROOT}/aggregate/data/metrics/c5_network_effects.csv",
            source_level="network_metric",
            producer_task="C5 Layer-2 successor closure",
            filters="screening_pass=1; endpoint-specific; prefix_k in 1,5",
            held_fixed=(
                "post-B Layer-2 u/x only; receiver Layer-1/3 u/x preserved; "
                "fast state reset; identical C"
            ),
            aggregation_path=(
                "paired cells -> within-network mean donor-transfer coefficient"
            ),
            seeded=False,
            required_columns=(
                "network_seed",
                "prefix_k",
                "endpoint",
                "mean_transfer",
                "screening_pass",
            ),
        ),
    )
    inference = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key=f"fig4.{panel_id}.c5_population_inference",
            pattern=f"{C5_ROOT}/aggregate/data/metrics/c5_population_inference.csv",
            source_level="population_inference",
            producer_task="C5 20-network exact sign-flip and bootstrap inference",
            filters="cohort=all_20; endpoint-specific; prefix_k in 1,5",
            held_fixed=(
                "independent unit=independently trained network; Holm family=four "
                "endpoint-by-K cells"
            ),
            aggregation_path="network effects -> bootstrap CI and exact sign-flip test",
            seeded=False,
            required_columns=(
                "cohort",
                "endpoint",
                "prefix_k",
                "n_networks",
                "mean_transfer",
                "median_transfer",
                "sd_across_networks",
                "min_network_transfer",
                "max_network_transfer",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "p_one_sided_exact_sign_flip",
                "holm_adjusted_p",
            ),
        ),
    )
    return effects, inference


def _c5_statistics(
    summary: pd.DataFrame,
    *,
    figure_id: str,
    panel_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        n_networks = int(row.n_networks)
        record = {column: pd.NA for column in STATISTICS_COLUMNS}
        prefix_label = f"K{int(row.prefix_k)}"
        record.update(
            {
                "figure_id": figure_id,
                "panel_id": panel_id,
                "endpoint": str(row.endpoint),
                "contrast": f"{row.endpoint}_{prefix_label}_vs_zero",
                "group": f"{row.endpoint}|{prefix_label}",
                "n_networks": n_networks,
                "estimate": float(row.mean_transfer),
                "mean": float(row.mean_transfer),
                "sd": float(row.sd_across_networks),
                "sem": float(row.sd_across_networks) / math.sqrt(n_networks),
                "ci95_low": float(row.bootstrap_ci95_low),
                "ci95_high": float(row.bootstrap_ci95_high),
                "median": float(row.median_transfer),
                "min": float(row.min_network_transfer),
                "max": float(row.max_network_transfer),
                "null_value": 0.0,
                "test_name": "one_sided_exact_sign_flip",
                "p_value": float(row.p_one_sided_exact_sign_flip),
                "p_adjust_method": "Holm",
                "p_adjusted": float(row.holm_adjusted_p),
                "alternative": "greater",
                "unit": "donor_transfer_index",
                "statistics_status": "supplied_confirmatory_bootstrap",
            }
        )
        rows.append(record)
    return pd.DataFrame(rows, columns=STATISTICS_COLUMNS)


def _build_fig4_c5_endpoint(
    ctx: BuilderContext,
    *,
    panel_id: str,
    endpoint: str,
) -> PanelResult:
    effects, inference = _load_c5_population_sources(ctx, panel_id)
    selected = effects.frame.loc[
        effects.frame["endpoint"].eq(endpoint)
        & effects.frame["prefix_k"].isin([1, 5])
        & effects.frame["screening_pass"].astype(bool)
    ].copy()
    summary = inference.frame.loc[
        inference.frame["cohort"].eq("all_20")
        & inference.frame["endpoint"].eq(endpoint)
        & inference.frame["prefix_k"].isin([1, 5])
    ].copy()
    if len(selected) != len(EXPECTED_SEEDS) * 2 or len(summary) != 2:
        raise ValueError(
            f"fig4{panel_id}: incomplete C5 endpoint {endpoint}; "
            f"effects={len(selected)}, summary={len(summary)}"
        )
    selected["prefix_label"] = selected["prefix_k"].map(
        lambda value: f"K{int(value)}"
    )
    summary = summary.assign(
        prefix_label=summary["prefix_k"].map(lambda value: f"K{int(value)}")
    )
    selected = selected.merge(
        summary.loc[
            :,
            [
                "endpoint",
                "prefix_k",
                "mean_transfer",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
            ],
        ].rename(
            columns={
                "mean_transfer": "summary_mean",
                "bootstrap_ci95_low": "summary_ci95_low",
                "bootstrap_ci95_high": "summary_ci95_high",
            }
        ),
        on=["endpoint", "prefix_k"],
        how="left",
        validate="many_to_one",
    )
    plot = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_causal_effect",
        endpoint=endpoint,
        condition="prefix_label",
        value="mean_transfer",
        unit="donor_transfer_index",
        dimensions=(
            "prefix_k",
            "prefix_label",
            "summary_mean",
            "summary_ci95_low",
            "summary_ci95_high",
        ),
    )
    plot["condition"] = selected["prefix_label"].to_numpy()
    plot = plot.sort_values(["network_seed", "prefix_k"], kind="mergesort")
    statistics = _c5_statistics(
        summary,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        [*effects.records, *inference.records],
        input_rows=len(effects.frame) + len(inference.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "prefix_k",
        ),
        exclusion_reason=(
            "the other C5 endpoint and confirmatory-19 sensitivity rows are retained "
            "outside this main panel"
        ),
    )


def _build_fig4a_c5_processing(ctx: BuilderContext) -> PanelResult:
    return _build_fig4_c5_endpoint(
        ctx,
        panel_id="a",
        endpoint="early_layer2_event_map_donor_transfer",
    )


def _build_fig4b_c5_successor(ctx: BuilderContext) -> PanelResult:
    return _build_fig4_c5_endpoint(
        ctx,
        panel_id="b",
        endpoint="layer3_successor_ux_donor_transfer",
    )


def build_fig4(ctx: BuilderContext) -> list[PanelResult]:
    return [
        _build_fig4a_c5_processing(ctx),
        _build_fig4b_c5_successor(ctx),
        _build_fig4a_accumulated(ctx, panel_id="c"),
        _build_fig4b_accumulated(ctx, panel_id="d"),
    ]


def build_fig5(ctx: BuilderContext) -> list[PanelResult]:
    return [
        _build_fig5a(ctx),
        _build_fig5b(ctx),
        _build_fig5c_neff(ctx),
        _build_fig5d_latest_item(ctx),
        _build_fig5e_effective_area(ctx),
        _build_fig5f(ctx),
    ]


def _pair_network_source(ctx: BuilderContext, panel_id: str) -> LoadedSource:
    return _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key=f"fig5.{panel_id}.pair_network",
            pattern=(
                "results/paper_figure_multi_seed/new_results_reanalysis/metrics/"
                "fig6_layer2_pair_network_metrics.csv"
            ),
            source_level="validated_artifact",
            producer_task="new_results_reanalysis Layer 2 pair metrics",
            filters="layer=layer2; state_variable=ux_concat",
            held_fixed="network_seed=1000-1019; pair is not an independent replicate",
            aggregation_path="pair metrics -> network estimate",
            seeded=False,
            required_columns=(
                "network_seed",
                "min_component_similarity",
                "true_minus_shuffled",
                "residual_pair_specificity",
                "layer",
                "state_variable",
            ),
        ),
    )


def _build_fig5_pair_endpoint(
    ctx: BuilderContext,
    panel_id: str,
    endpoint: str,
    infer_zero: bool,
) -> PanelResult:
    source = _pair_network_source(ctx, panel_id)
    selected = source.frame.loc[
        source.frame["layer"].eq("layer2")
        & source.frame["state_variable"].eq("ux_concat")
    ].copy()
    plot = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_metric",
        endpoint=endpoint,
        condition="layer2_ux",
        value=endpoint,
        unit="similarity_index",
        dimensions=("layer", "state_variable"),
    ).sort_values(["network_seed"], kind="mergesort")
    status = "predeclared_recomputed" if infer_zero else "descriptive_only"
    nulls = {endpoint: 0.0} if infer_zero else {}
    values = _statistics_values(
        plot,
        group_columns=("condition",),
        status=status,
        null_by_endpoint=nulls,
    )
    values["contrast"] = f"{endpoint}_vs_zero" if infer_zero else ""
    statistics = build_statistics(values, figure_id=ctx.figure_id, panel_id=panel_id)
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=("figure_id", "panel_id", "network_seed", "endpoint"),
        exclusion_reason=(
            "linear mixture diagnostics and non-frozen pair endpoints excluded"
            if panel_id == "c"
            else ""
        ),
    )


def _fig5_pair_condition_source(
    ctx: BuilderContext,
    panel_id: str,
    *,
    filename: str,
    required_columns: Sequence[str],
) -> LoadedSource:
    return _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key=f"fig5.{panel_id}.pair_conditions",
            pattern=(
                "results/paper_figure_multi_seed/fig2_pair_fused_stsp_state/"
                "fig2_pair_fused_stsp_state/seed_*/data/metrics/"
                f"{filename}"
            ),
            source_level="intermediate",
            producer_task="pair-state Layer 2 morphology",
            filters="layer=layer2; state_variable=ux_concat",
            held_fixed="pair rows aggregate within each independently trained network",
            aggregation_path="pair metrics -> network x displayed condition mean",
            required_columns=tuple(required_columns),
        ),
    )


def _build_fig5a(ctx: BuilderContext) -> PanelResult:
    panel_id = "a"
    source = _fig5_pair_condition_source(
        ctx,
        panel_id,
        filename="panel_b_dual_retention_metrics.csv",
        required_columns=(
            "network_seed",
            "pair_id",
            "layer",
            "state_variable",
            "sim_to_A",
            "sim_to_B",
            "min_component_similarity",
        ),
    )
    selected = source.frame.loc[
        source.frame["layer"].eq("layer2")
        & source.frame["state_variable"].eq("ux_concat")
    ].copy()
    network = selected.groupby("network_seed", as_index=False)[
        ["sim_to_A", "sim_to_B", "min_component_similarity"]
    ].mean()
    long = network.melt(
        id_vars=("network_seed",),
        value_vars=("sim_to_A", "sim_to_B"),
        var_name="constituent",
        value_name="similarity",
    )
    long["condition_name"] = long["constituent"].map(
        {"sim_to_A": "item_a", "sim_to_B": "item_b"}
    )
    plot = make_plot_data(
        long,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_constituent_similarity",
        endpoint="constituent_similarity",
        condition="constituent",
        value="similarity",
        unit="similarity_index",
        dimensions=("condition_name",),
    )
    plot["condition"] = long["condition_name"].to_numpy()
    plot = plot.sort_values(["network_seed", "condition"], kind="mergesort")
    min_plot = make_plot_data(
        network,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_guard_endpoint",
        endpoint="min_component_similarity",
        condition="layer2_ux",
        value="min_component_similarity",
        unit="similarity_index",
    ).sort_values(["network_seed"], kind="mergesort")
    descriptive = _statistics_values(
        plot,
        group_columns=("condition",),
        status="descriptive_only",
    )
    guard = _statistics_values(
        min_plot,
        group_columns=("condition",),
        status="descriptive_only",
    )
    statistics = build_statistics(
        pd.concat([descriptive, guard], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "condition",
        ),
        extra_data={"panel_a_min_component_similarity.csv": min_plot},
        exclusion_reason=(
            "pair rows aggregate within network; the minimum-constituent guard remains "
            "available as auxiliary source data"
        ),
    )


def _build_fig5b(ctx: BuilderContext) -> PanelResult:
    panel_id = "b"
    source = _fig5_pair_condition_source(
        ctx,
        panel_id,
        filename="panel_c_pair_specificity_metrics.csv",
        required_columns=(
            "network_seed",
            "pair_id",
            "layer",
            "state_variable",
            "true_pair_score",
            "shuffled_pair_score",
            "true_minus_shuffled",
        ),
    )
    selected = source.frame.loc[
        source.frame["layer"].eq("layer2")
        & source.frame["state_variable"].eq("ux_concat")
    ].copy()
    network = selected.groupby("network_seed", as_index=False)[
        ["true_pair_score", "shuffled_pair_score", "true_minus_shuffled"]
    ].mean()
    long = network.melt(
        id_vars=("network_seed",),
        value_vars=("true_pair_score", "shuffled_pair_score"),
        var_name="pair_condition",
        value_name="similarity",
    )
    long["condition_name"] = long["pair_condition"].map(
        {
            "true_pair_score": "experienced_pair",
            "shuffled_pair_score": "shuffled_pair",
        }
    )
    plot = make_plot_data(
        long,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_pair_similarity",
        endpoint="pair_similarity",
        condition="pair_condition",
        value="similarity",
        unit="similarity_index",
        dimensions=("condition_name",),
    )
    plot["condition"] = long["condition_name"].to_numpy()
    plot = plot.sort_values(["network_seed", "condition"], kind="mergesort")
    contrast_plot = make_plot_data(
        network,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_contrast",
        endpoint="true_minus_shuffled",
        condition="layer2_ux",
        value="true_minus_shuffled",
        unit="similarity_index",
    ).sort_values(["network_seed"], kind="mergesort")
    descriptive = _statistics_values(
        plot,
        group_columns=("condition",),
        status="descriptive_only",
    )
    inference = _statistics_values(
        contrast_plot,
        group_columns=("condition",),
        status="predeclared_recomputed",
        null_by_endpoint={"true_minus_shuffled": 0.0},
        p_adjust_family="fig5b_pair_specificity",
    )
    inference["contrast"] = "true_minus_shuffled"
    statistics = build_statistics(
        pd.concat([descriptive, inference], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "condition",
        ),
        extra_data={"panel_b_true_minus_shuffled.csv": contrast_plot},
        exclusion_reason=(
            "pair rows aggregate within network; the paired contrast remains in Source Data"
        ),
    )


def _build_fig5c(ctx: BuilderContext) -> PanelResult:
    panel_id = "c"
    source = _fig5_pair_condition_source(
        ctx,
        panel_id,
        filename="panel_d_linear_residual_pair_specificity_metrics.csv",
        required_columns=(
            "network_seed",
            "pair_id",
            "layer",
            "state_variable",
            "residual_true_pair_score",
            "residual_shuffled_pair_score",
            "residual_pair_specificity",
        ),
    )
    selected = source.frame.loc[
        source.frame["layer"].eq("layer2")
        & source.frame["state_variable"].eq("ux_concat")
    ].copy()
    network = selected.groupby("network_seed", as_index=False)[
        [
            "residual_true_pair_score",
            "residual_shuffled_pair_score",
            "residual_pair_specificity",
        ]
    ].mean()
    long = network.melt(
        id_vars=("network_seed",),
        value_vars=("residual_true_pair_score", "residual_shuffled_pair_score"),
        var_name="residual_condition",
        value_name="residual_similarity",
    )
    long["condition_name"] = long["residual_condition"].map(
        {
            "residual_true_pair_score": "experienced_residual",
            "residual_shuffled_pair_score": "shuffled_residual",
        }
    )
    plot = make_plot_data(
        long,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_residual_similarity",
        endpoint="residual_pair_similarity",
        condition="residual_condition",
        value="residual_similarity",
        unit="similarity_index",
        dimensions=("condition_name",),
    )
    plot["condition"] = long["condition_name"].to_numpy()
    plot = plot.sort_values(["network_seed", "condition"], kind="mergesort")
    contrast_plot = make_plot_data(
        network,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_contrast",
        endpoint="residual_pair_specificity",
        condition="layer2_ux",
        value="residual_pair_specificity",
        unit="similarity_index",
    ).sort_values(["network_seed"], kind="mergesort")
    descriptive = _statistics_values(
        plot,
        group_columns=("condition",),
        status="descriptive_only",
    )
    inference = _statistics_values(
        contrast_plot,
        group_columns=("condition",),
        status="predeclared_recomputed",
        null_by_endpoint={"residual_pair_specificity": 0.0},
        p_adjust_family="fig5c_residual_specificity",
    )
    inference["contrast"] = "residual_pair_specificity"
    statistics = build_statistics(
        pd.concat([descriptive, inference], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "condition",
        ),
        extra_data={"panel_c_residual_pair_specificity.csv": contrast_plot},
        exclusion_reason=(
            "residual pair rows aggregate within network; model diagnostics remain outside the main panel"
        ),
    )


def _build_fig5c_neff(ctx: BuilderContext) -> PanelResult:
    panel_id = "c"
    source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig5.multi_component_neff",
            pattern=(
                "results/paper_figure_multi_seed/new_results_reanalysis/metrics/"
                "fig6_layer2_multi_network_metrics.csv"
            ),
            source_level="validated_artifact",
            producer_task="new_results_reanalysis Layer 2 multi-item metrics",
            filters="seq_len in 3,5,7,10; endpoint=n_eff",
            held_fixed=(
                "Layer 2 joint u/x; native primary N_eff definition; N_eff is an "
                "effective component number, not accessible-item count or capacity"
            ),
            aggregation_path="sequences -> network x sequence-length N_eff",
            seeded=False,
            required_columns=("network_seed", "seq_len", "n_eff"),
        ),
    )
    selected = source.frame.loc[source.frame["seq_len"].isin([3, 5, 7, 10])].copy()
    plot = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_metric",
        endpoint="N_eff",
        condition="sequence_length",
        value="n_eff",
        unit="effective_items",
        dimensions=("seq_len",),
    ).sort_values(["network_seed", "seq_len"], kind="mergesort")
    values = _statistics_values(
        plot,
        group_columns=("seq_len",),
        status="descriptive_only",
    )
    statistics = build_statistics(values, figure_id=ctx.figure_id, panel_id=panel_id)
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "seq_len",
        ),
    )


def _build_fig5d_latest_item(ctx: BuilderContext) -> PanelResult:
    panel_id = "d"
    source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig5.latest_item_weight",
            pattern=(
                "results/paper_figure_multi_seed/new_results_reanalysis/metrics/"
                "fig6_layer2_multi_item_weights.csv"
            ),
            source_level="intermediate",
            producer_task="new_results_reanalysis Layer 2 constituent weights",
            filters=(
                "seq_len in 3,5,7,10; item_position=seq_len after validating all "
                "within-sequence normalized weights"
            ),
            held_fixed=(
                "native primary constituent-weight definition; the 0.5 line is a "
                "latest-item-only exclusion reference, not a primacy/recency test"
            ),
            aggregation_path=(
                "item coefficient -> sequence-normalized weight -> latest item -> "
                "network x K mean"
            ),
            seeded=False,
            required_columns=(
                "network_seed",
                "sequence_id",
                "seq_len",
                "item_position",
                "item_weight",
            ),
        ),
    )
    selected = source.frame.loc[source.frame["seq_len"].isin([3, 5, 7, 10])].copy()
    invalid_position = selected["item_position"].gt(selected["seq_len"])
    if invalid_position.any():
        raise ValueError("fig5e: source contains item positions beyond sequence length")
    sums = selected.groupby(
        ["network_seed", "sequence_id", "seq_len"],
        as_index=False,
    )["item_weight"].sum(min_count=1)
    finite_sums = pd.to_numeric(sums["item_weight"], errors="coerce").dropna()
    if not np.allclose(finite_sums.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("fig5e: sequence item weights are not normalized to one")
    latest = selected.loc[selected["item_position"].eq(selected["seq_len"])].copy()
    network = latest.groupby(
        ["network_seed", "seq_len"],
        as_index=False,
    )["item_weight"].mean()
    plot = make_plot_data(
        network,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_metric",
        endpoint="latest_item_weight",
        condition="latest_item",
        value="item_weight",
        unit="proportion",
        dimensions=("seq_len",),
    ).sort_values(
        ["network_seed", "seq_len"],
        kind="mergesort",
    )
    values = _statistics_values(
        plot,
        group_columns=("seq_len",),
        status="descriptive_only",
    )
    statistics = build_statistics(values, figure_id=ctx.figure_id, panel_id=panel_id)
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "seq_len",
        ),
        exclusion_reason=(
            "non-latest positions are used only to validate within-sequence "
            "normalization and are not displayed; method sensitivity remains in New S6"
        ),
    )


def _build_fig5e_effective_area(ctx: BuilderContext) -> PanelResult:
    panel_id = "e"
    source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig5.layer1_g_effective_area",
            pattern=(
                "results/paper_figure_multi_seed/supplementary_v5/"
                "data/source_data/s6_b.csv"
            ),
            source_level="validated_artifact",
            producer_task="supplementary_v5 S6 coefficient-free morphology analysis",
            filters="seq_len=3,5,7,10; delay_ms=100,200,400,800",
            held_fixed=(
                "Layer 1 g _map28 positive-delta map; entropy effective area "
                "normalized by 28x28 sites"
            ),
            aggregation_path=(
                "persisted Layer-1 g maps -> positive-delta spatial distribution -> "
                "network x K x delay effective area"
            ),
            seeded=False,
            required_columns=("network_seed", "seq_len", "delay_ms", "value"),
        ),
    )
    selected = source.frame.loc[
        source.frame["seq_len"].isin([3, 5, 7, 10])
        & source.frame["delay_ms"].isin([100, 200, 400, 800])
    ].copy()
    plot = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_heatmap_cell",
        endpoint="layer1_g_effective_area",
        condition="coefficient_free_spatial_footprint",
        value="value",
        unit="fraction_of_map_area",
        dimensions=("seq_len", "delay_ms"),
    ).sort_values(
        ["network_seed", "seq_len", "delay_ms"],
        kind="mergesort",
    )
    values = _statistics_values(
        plot,
        group_columns=("seq_len", "delay_ms"),
        status="descriptive_only",
    )
    statistics = build_statistics(
        values,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "seq_len",
            "delay_ms",
        ),
    )


def _build_fig5f(ctx: BuilderContext) -> PanelResult:
    panel_id = "f"
    source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig5.coefficient_free_morphology_boundary",
            pattern=(
                "results/paper_figure_multi_seed/supplementary_v5/"
                "data/source_data/s6_d_cells.csv"
            ),
            source_level="validated_artifact",
            producer_task=(
                "supplementary_v5 S6 coefficient-free morphology analysis"
            ),
            filters="seq_len=3,5,7,10; delay_ms=100,200,400,800",
            held_fixed=(
                "Layer 1 g maps; equal-weight singleton composite; "
                "within-cell matched and deranged sequence controls"
            ),
            aggregation_path=(
                "sequence map -> matched/deranged cosine difference -> "
                "network x K x delay mean"
            ),
            seeded=False,
            required_columns=(
                "network_seed",
                "seq_len",
                "delay_ms",
                "matched_cosine",
                "deranged_cosine",
                "value",
            ),
        ),
    )
    selected = source.frame.loc[
        source.frame["seq_len"].isin([3, 5, 7, 10])
        & source.frame["delay_ms"].isin([100, 200, 400, 800])
    ].copy()
    expected = (
        pd.to_numeric(selected["matched_cosine"], errors="coerce")
        - pd.to_numeric(selected["deranged_cosine"], errors="coerce")
    )
    observed = pd.to_numeric(selected["value"], errors="coerce")
    if not np.allclose(
        observed.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
        equal_nan=False,
    ):
        raise ValueError(
            "fig5f: matched-minus-deranged morphology identity failed"
        )
    plot = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_heatmap_cell",
        endpoint="matched_minus_deranged_cosine",
        condition="equal_weight_singleton_composite",
        value="value",
        unit="cosine_difference",
        dimensions=("seq_len", "delay_ms"),
    ).sort_values(
        ["network_seed", "seq_len", "delay_ms"],
        kind="mergesort",
    )
    values = _statistics_values(
        plot,
        group_columns=("seq_len", "delay_ms"),
        status="descriptive_only",
    )
    statistics = build_statistics(
        values,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "seq_len",
            "delay_ms",
        ),
    )


def build_fig6(ctx: BuilderContext) -> list[PanelResult]:
    return [
        _build_fig6a(ctx),
        _build_fig6b(ctx),
        _build_fig6c(ctx),
        _build_fig6d(ctx),
        _build_fig6e(ctx),
        _build_fig6f(ctx),
    ]


def _build_fig6a(ctx: BuilderContext) -> PanelResult:
    panel_id = "a"
    curve_source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig6.partial_cue_curves",
            pattern=(
                "results/paper_figure_multi_seed/fig2_pair_fused_stsp_state/"
                "fig2_pair_fused_stsp_state/seed_*/data/metrics/"
                "panel_f_partial_cue_metrics.csv"
            ),
            source_level="network_metric",
            producer_task="pair-state partial-cue recovery sweep",
            filters="target_item in A,B; state_condition in S0,S_A,S_B,S_AB",
            held_fixed="complete pre-existing keep-probability sweep",
            aggregation_path="cue trials -> network x target x state x keep probability",
            required_columns=(
                "network_seed",
                "state_condition",
                "target_item",
                "keep_prob",
                "P_target",
            ),
        ),
    )
    auc_source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig6.partial_cue_auc",
            pattern=(
                "results/paper_figure_multi_seed/fig2_pair_fused_stsp_state/"
                "fig2_pair_fused_stsp_state/seed_*/data/metrics/"
                "panel_f_partial_cue_auc_metrics.csv"
            ),
            source_level="network_metric",
            producer_task="pair-state partial-cue AUC",
            filters="state_condition=S_AB; target_item in A,B",
            held_fixed="existing cue-strength sweep; no single keep-probability selection",
            aggregation_path="cue-strength recovery curve -> network AUC gain",
            required_columns=(
                "network_seed",
                "state_condition",
                "target_item",
                "SAB_vs_S0_auc_gain",
                "SAB_vs_relevant_single_auc_gain",
            ),
        ),
    )
    curve_selected = curve_source.frame.loc[
        curve_source.frame["state_condition"].isin(["S0", "S_A", "S_B", "S_AB"])
        & curve_source.frame["target_item"].isin(["A", "B"])
        & curve_source.frame["keep_prob"].isin([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ].copy()
    curve_network = curve_selected.groupby(
        ["network_seed", "target_item", "state_condition", "keep_prob"],
        as_index=False,
    )["P_target"].mean()
    curve_network["target_percent"] = (
        pd.to_numeric(curve_network["P_target"], errors="coerce") * 100.0
    )
    plot = make_plot_data(
        curve_network,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_partial_cue_curve",
        endpoint="P_target",
        condition="state_condition",
        value="target_percent",
        unit="percent",
        dimensions=("target_item", "state_condition", "keep_prob"),
    )
    plot["condition"] = curve_network["state_condition"].to_numpy()
    plot = plot.sort_values(
        ["network_seed", "target_item", "condition", "keep_prob"],
        kind="mergesort",
    )
    auc_selected = auc_source.frame.loc[
        auc_source.frame["state_condition"].eq("S_AB")
        & auc_source.frame["target_item"].isin(["A", "B"])
    ].copy()
    long = auc_selected.melt(
        id_vars=("network_seed", "target_item", "state_condition"),
        value_vars=("SAB_vs_S0_auc_gain", "SAB_vs_relevant_single_auc_gain"),
        var_name="endpoint_name",
        value_name="auc_gain",
    )
    auc_plot = make_plot_data(
        long,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_auc_contrast",
        endpoint="endpoint_name",
        condition="target_item",
        value="auc_gain",
        unit="normalized_auc_gain",
        dimensions=("target_item",),
    )
    auc_plot["endpoint"] = long["endpoint_name"].to_numpy()
    auc_plot["condition"] = long["target_item"].to_numpy()
    auc_plot = auc_plot.sort_values(
        ["network_seed", "target_item", "endpoint"], kind="mergesort"
    )
    curve_values = _statistics_values(
        plot,
        group_columns=("target_item", "state_condition", "keep_prob"),
        status="descriptive_only",
    )
    auc_values = _statistics_values(
        auc_plot,
        group_columns=("target_item",),
        status="predeclared_recomputed",
        null_by_endpoint={
            "SAB_vs_S0_auc_gain": 0.0,
            "SAB_vs_relevant_single_auc_gain": 0.0,
        },
        p_adjust_family="fig6a_partial_cue_auc",
    )
    auc_values["contrast"] = auc_values["endpoint"]
    statistics = build_statistics(
        pd.concat([curve_values, auc_values], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        [*curve_source.records, *auc_source.records],
        input_rows=len(curve_source.frame) + len(auc_source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "target_item",
            "condition",
            "keep_prob",
        ),
        extra_data={"panel_a_auc_contrasts.csv": auc_plot},
        exclusion_reason=(
            "all four pre-existing state conditions are displayed; supplied AUC contrasts "
            "remain the inferential endpoint in Source Data"
        ),
    )


def _multiitem_source(
    ctx: BuilderContext,
    panel_id: str,
    *,
    key: str,
    filename: str,
    source_level: str,
    filters: str,
    held_fixed: str,
    aggregation_path: str,
    required_columns: Sequence[str],
) -> LoadedSource:
    return _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key=key,
            pattern=(
                "results/paper_figure_multi_seed/fig3_multiitem_peak_landscape/"
                f"seed_*/data/metrics/{filename}"
            ),
            source_level=source_level,
            producer_task="multi-item peak landscape",
            filters=filters,
            held_fixed=held_fixed,
            aggregation_path=aggregation_path,
            required_columns=required_columns,
        ),
    )


def _build_fig6b(ctx: BuilderContext) -> PanelResult:
    panel_id = "b"
    source = _multiitem_source(
        ctx,
        panel_id,
        key="fig6.serial_access",
        filename="panel_d_item_functional_gain.csv",
        source_level="intermediate",
        filters="seq_len=10; delay_ms=400; keep_prob=0.5",
        held_fixed="all target positions; slot-matched singleton comparison",
        aggregation_path="sequence x target access -> network x target-position mean",
        required_columns=(
            "network_seed",
            "sequence_id",
            "seq_len",
            "delay_ms",
            "target_position",
            "keep_prob",
            "P_target_sequence_state",
            "P_target_single_item_memory",
            "P_target_cue_only",
            "G_i",
        ),
    )
    selected = source.frame.loc[
        source.frame["seq_len"].eq(10)
        & source.frame["delay_ms"].eq(400)
        & np.isclose(
            pd.to_numeric(source.frame["keep_prob"], errors="coerce"),
            0.5,
            rtol=0.0,
            atol=1e-12,
        )
    ].copy()
    expected_g = (
        pd.to_numeric(selected["P_target_sequence_state"], errors="coerce")
        - pd.to_numeric(selected["P_target_cue_only"], errors="coerce")
    )
    if not np.allclose(
        pd.to_numeric(selected["G_i"], errors="coerce"),
        expected_g,
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError("fig6b: supplied G_i is not sequence minus cue-only")
    network = selected.groupby(
        ["network_seed", "target_position"], as_index=False
    )[
        [
            "P_target_sequence_state",
            "P_target_single_item_memory",
            "P_target_cue_only",
        ]
    ].mean()
    endpoint = "sequence_minus_singleton_access_gain"
    network[endpoint] = (
        pd.to_numeric(network["P_target_sequence_state"], errors="coerce")
        - pd.to_numeric(network["P_target_single_item_memory"], errors="coerce")
    ) * 100.0
    plot = make_plot_data(
        network,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_serial_access_gain",
        endpoint=endpoint,
        condition="K10_D400",
        value=endpoint,
        unit="percent",
        dimensions=("target_position",),
    )
    plot["seq_len"] = 10
    plot["delay_ms"] = 400
    plot = plot.sort_values(
        ["network_seed", "target_position"],
        kind="mergesort",
    )
    descriptive = _statistics_values(
        plot,
        group_columns=("target_position",),
        status="descriptive_only",
    )
    gain = plot.groupby("network_seed", as_index=False)["value"].mean()
    inference = _contrast_statistics_values(
        gain,
        endpoint=endpoint,
        contrast="mean_sequence_minus_singleton_access_gain_vs_zero",
        unit="percent",
        p_adjust_family="fig6b_sequence_singleton_access",
    )
    statistics = build_statistics(
        pd.concat([descriptive, inference], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    absolute_long = network.melt(
        id_vars=("network_seed", "target_position"),
        value_vars=(
            "P_target_sequence_state",
            "P_target_single_item_memory",
            "P_target_cue_only",
        ),
        var_name="endpoint_name",
        value_name="probability",
    )
    absolute_long["probability_percent"] = (
        pd.to_numeric(absolute_long["probability"], errors="coerce") * 100.0
    )
    absolute_plot = make_plot_data(
        absolute_long,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="auxiliary_absolute_access",
        endpoint="endpoint_name",
        condition="K10_D400",
        value="probability_percent",
        unit="percent",
        dimensions=("target_position",),
    )
    absolute_plot["endpoint"] = absolute_long["endpoint_name"].to_numpy()
    absolute_plot["seq_len"] = 10
    absolute_plot["delay_ms"] = 400
    absolute_plot = absolute_plot.sort_values(
        ["network_seed", "target_position", "endpoint"], kind="mergesort"
    )
    absolute_values = _statistics_values(
        absolute_plot,
        group_columns=("target_position",),
        status="descriptive_only",
    )
    absolute_statistics = build_statistics(
        absolute_values,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "target_position",
            "endpoint",
        ),
        extra_data={"panel_b_absolute_access.csv": absolute_plot},
        extra_metrics={"panel_b_absolute_access_statistics.csv": absolute_statistics},
        exclusion_reason=(
            "all non-K10/D400 conditions excluded by frozen focus protocol; "
            "absolute conditions retained as auxiliary source data"
        ),
    )


def _build_fig6c(ctx: BuilderContext) -> PanelResult:
    panel_id = "c"
    source = _multiitem_source(
        ctx,
        panel_id,
        key="fig6.cue_specificity",
        filename="panel_c_cue_specificity_metrics.csv",
        source_level="intermediate",
        filters=(
            "seq_len=7; delay_ms=400; state_condition=S_final; "
            "memory_condition=sequence_state"
        ),
        held_fixed=(
            "cue_type in matched,mismatched,unseen; keep_prob=0.5; persisted "
            "mismatched means an out-of-sequence same-label novel exemplar"
        ),
        aggregation_path=(
            "cue trials -> sequence x target probability -> network x target-position mean"
        ),
        required_columns=(
            "network_seed",
            "sequence_id",
            "seq_len",
            "delay_ms",
            "target_position",
            "cue_type",
            "state_condition",
            "memory_condition",
            "keep_prob",
            "P_target",
        ),
    )
    selected = source.frame.loc[
        source.frame["seq_len"].eq(7)
        & source.frame["delay_ms"].eq(400)
        & source.frame["state_condition"].eq("S_final")
        & source.frame["memory_condition"].eq("sequence_state")
        & source.frame["cue_type"].isin(["matched", "mismatched", "unseen"])
        & np.isclose(
            pd.to_numeric(source.frame["keep_prob"], errors="coerce"),
            0.5,
            rtol=0.0,
            atol=1e-12,
        )
    ].copy()
    selected["cue_condition"] = selected["cue_type"].map(
        {
            "matched": "matched",
            "mismatched": "same_label_novel",
            "unseen": "unseen",
        }
    )
    network_cells = (
        selected.groupby(
            ["network_seed", "cue_condition", "target_position"],
            as_index=False,
        )["P_target"]
        .mean()
    )
    network_cells["probability_percent"] = (
        pd.to_numeric(network_cells["P_target"], errors="coerce") * 100.0
    )
    cells = make_plot_data(
        network_cells,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_target_probability",
        endpoint="target_probability",
        condition="cue_condition",
        value="probability_percent",
        unit="percent",
        dimensions=("cue_condition", "target_position"),
    )
    cells["condition"] = network_cells["cue_condition"].to_numpy()
    network_cue = (
        network_cells.groupby(["network_seed", "cue_condition"], as_index=False)[
            "probability_percent"
        ]
        .mean()
        .pivot(
            index="network_seed",
            columns="cue_condition",
            values="probability_percent",
        )
    )
    contrast_frame = pd.DataFrame(
        {
            "network_seed": network_cue.index.astype(int),
            "matched_minus_same_label_novel": (
                network_cue["matched"] - network_cue["same_label_novel"]
            ).to_numpy(dtype=float),
            "matched_minus_unseen": (
                network_cue["matched"] - network_cue["unseen"]
            ).to_numpy(dtype=float),
        }
    ).melt(
        id_vars=("network_seed",),
        value_vars=("matched_minus_same_label_novel", "matched_minus_unseen"),
        var_name="endpoint_name",
        value_name="contrast_percent",
    )
    contrasts = make_plot_data(
        contrast_frame,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_contrast",
        endpoint="endpoint_name",
        condition="cue_specificity",
        value="contrast_percent",
        unit="percent",
    )
    contrasts["endpoint"] = contrast_frame["endpoint_name"].to_numpy()
    plot = contrasts.sort_values(
        ["network_seed", "endpoint"], kind="mergesort"
    )
    descriptive = _statistics_values(
        cells,
        group_columns=("cue_condition", "target_position"),
        status="descriptive_only",
    )
    inference = _statistics_values(
        contrasts,
        group_columns=("condition",),
        status="predeclared_recomputed",
        null_by_endpoint={
            "matched_minus_same_label_novel": 0.0,
            "matched_minus_unseen": 0.0,
        },
        p_adjust_family="fig6c_cue_specificity",
    )
    inference["contrast"] = inference["endpoint"]
    statistics = build_statistics(
        inference,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    position_statistics = build_statistics(
        descriptive,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "condition",
        ),
        extra_data={"panel_c_position_profiles.csv": cells},
        extra_metrics={"panel_c_position_statistics.csv": position_statistics},
        exclusion_reason=(
            "S0 and non-frozen protocol rows excluded; positional condition profiles "
            "retained as auxiliary source data"
        ),
    )


def _build_fig6d(ctx: BuilderContext) -> PanelResult:
    panel_id = "d"
    source = _multiitem_source(
        ctx,
        panel_id,
        key="fig6.functional_boundary",
        filename="panel_f_boundary_summary.csv",
        source_level="network_metric",
        filters="seq_len=3,5,7,10; delay_ms=100,200,400,800",
        held_fixed="functional endpoint=rescued_fraction; support_gain_corr excluded",
        aggregation_path="sequence and cue trials -> network x K x delay rescued fraction",
        required_columns=("network_seed", "seq_len", "delay_ms", "rescued_fraction"),
    )
    selected = source.frame.loc[
        source.frame["seq_len"].isin([3, 5, 7, 10])
        & source.frame["delay_ms"].isin([100, 200, 400, 800])
    ].copy()
    plot = make_plot_data(
        selected,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_heatmap_cell",
        endpoint="rescued_fraction",
        condition="functional_boundary",
        value="rescued_fraction",
        unit="fraction",
        dimensions=("seq_len", "delay_ms"),
    ).sort_values(
        ["network_seed", "seq_len", "delay_ms"],
        kind="mergesort",
    )
    descriptive = _statistics_values(
        plot,
        group_columns=("seq_len", "delay_ms"),
        status="descriptive_only",
    )
    interaction = _network_interaction_coefficients(selected, "rescued_fraction")
    inference = _contrast_statistics_values(
        interaction,
        endpoint="rescued_fraction",
        contrast="standardized_seq_len_x_delay_interaction",
        unit="standardized_interaction_coefficient",
    )
    statistics = build_statistics(
        pd.concat([descriptive, inference], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "seq_len",
            "delay_ms",
        ),
        exclusion_reason="support_gain_corr and nonfunctional structural metrics excluded",
    )


def _network_interaction_coefficients(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for seed, part in frame.groupby("network_seed"):
        work = part.loc[:, ["seq_len", "delay_ms", value_column]].dropna().copy()
        k = pd.to_numeric(work["seq_len"], errors="coerce").to_numpy(dtype=float)
        delay = pd.to_numeric(work["delay_ms"], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(work[value_column], errors="coerce").to_numpy(dtype=float)
        k = (k - np.mean(k)) / max(float(np.std(k, ddof=0)), 1.0)
        delay = (delay - np.mean(delay)) / max(float(np.std(delay, ddof=0)), 1.0)
        keep = np.isfinite(k) & np.isfinite(delay) & np.isfinite(values)
        if int(keep.sum()) != 16:
            raise ValueError(f"fig6d: seed {seed} does not contain the complete 4x4 grid")
        design = np.column_stack(
            [
                np.ones(int(keep.sum())),
                k[keep],
                delay[keep],
                k[keep] * delay[keep],
            ]
        )
        beta, *_ = np.linalg.lstsq(design, values[keep], rcond=None)
        records.append({"network_seed": int(seed), "value": float(beta[3])})
    return pd.DataFrame(records)


def _build_fig6e(ctx: BuilderContext) -> PanelResult:
    panel_id = "e"
    source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig6.high_stsp_ablation_exact_match",
            pattern=(
                "results/paper_figure_multi_seed/fig6_peak_amplified_reentry/"
                "seed_*/data/metrics/panel_f_high_stsp_overlap_ablation_summary.csv"
            ),
            source_level="intermediate",
            producer_task="high-STSP-overlap ablation",
            filters=(
                "early_window_ms=10; loss_condition in high_stsp_overlap,matched_removal; "
                "retain only trials with exactly equal removed_active_area and "
                "removed_input_energy"
            ),
            held_fixed=(
                "same network, sequence, probe, and early window; exact area and "
                "input-energy matching before within-network aggregation"
            ),
            aggregation_path=(
                "paired per-condition ablation summary -> exact-match trial subset -> "
                "network condition means and paired difference"
            ),
            required_columns=(
                "network_seed",
                "sequence_id",
                "probe_id",
                "early_window_ms",
                "loss_condition",
                "loss_delta_spike_probability",
                "removed_active_area",
                "removed_input_energy",
            ),
        ),
    )
    endpoint = "high_stsp_overlap_minus_matched_loss"
    selected = source.frame.loc[
        source.frame["early_window_ms"].eq(10)
        & source.frame["loss_condition"].isin(
            ["high_stsp_overlap", "matched_removal"]
        )
    ].copy()
    identity = ["network_seed", "sequence_id", "probe_id", "early_window_ms"]
    if selected.duplicated([*identity, "loss_condition"]).any():
        raise ValueError("fig6e: duplicate condition rows within a paired trial")
    loss = selected.pivot(
        index=identity,
        columns="loss_condition",
        values="loss_delta_spike_probability",
    )
    area = selected.pivot(
        index=identity,
        columns="loss_condition",
        values="removed_active_area",
    )
    energy = selected.pivot(
        index=identity,
        columns="loss_condition",
        values="removed_input_energy",
    )
    required_conditions = {"high_stsp_overlap", "matched_removal"}
    for name, frame in (("loss", loss), ("area", area), ("energy", energy)):
        if not required_conditions.issubset(frame.columns):
            raise ValueError(f"fig6e: {name} pivot lacks paired conditions")
    paired = loss.join(
        area.add_prefix("area_"),
        how="inner",
    ).join(
        energy.add_prefix("energy_"),
        how="inner",
    ).reset_index()
    paired["exact_area_match"] = np.isclose(
        pd.to_numeric(paired["area_high_stsp_overlap"], errors="coerce"),
        pd.to_numeric(paired["area_matched_removal"], errors="coerce"),
        rtol=0.0,
        atol=0.0,
    )
    paired["exact_energy_match"] = np.isclose(
        pd.to_numeric(paired["energy_high_stsp_overlap"], errors="coerce"),
        pd.to_numeric(paired["energy_matched_removal"], errors="coerce"),
        rtol=0.0,
        atol=0.0,
    )
    paired["exact_match"] = (
        paired["exact_area_match"] & paired["exact_energy_match"]
    )
    coverage = paired.groupby("network_seed", as_index=False).agg(
        n_trials=("exact_match", "size"),
        n_exact_trials=("exact_match", "sum"),
        exact_match_fraction=("exact_match", "mean"),
    )
    exact = paired.loc[paired["exact_match"]].copy()
    network = exact.groupby("network_seed", as_index=False)[
        ["high_stsp_overlap", "matched_removal"]
    ].mean()
    network = network.merge(
        coverage,
        on="network_seed",
        how="left",
        validate="one_to_one",
    )
    if len(network) != len(EXPECTED_SEEDS) or network["n_exact_trials"].le(0).any():
        raise ValueError("fig6e: exact-match subset lacks one or more networks")
    long = network.melt(
        id_vars=(
            "network_seed",
            "n_trials",
            "n_exact_trials",
            "exact_match_fraction",
        ),
        value_vars=("high_stsp_overlap", "matched_removal"),
        var_name="condition_name",
        value_name="recruitment_loss",
    )
    long["recruitment_loss_percent"] = (
        pd.to_numeric(long["recruitment_loss"], errors="coerce") * 100.0
    )
    plot = make_plot_data(
        long,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="paired_network_condition",
        endpoint="recruitment_loss",
        condition="condition_name",
        value="recruitment_loss_percent",
        unit="percent",
        dimensions=("n_trials", "n_exact_trials", "exact_match_fraction"),
    )
    plot["condition"] = long["condition_name"].to_numpy()
    plot = plot.sort_values(["network_seed", "condition"], kind="mergesort")
    descriptive = _statistics_values(
        plot,
        group_columns=("condition",),
        status="descriptive_only",
    )
    paired = pd.DataFrame(
        {
            "network_seed": network["network_seed"].astype(int),
            "value": (
                pd.to_numeric(network["high_stsp_overlap"], errors="coerce")
                - pd.to_numeric(network["matched_removal"], errors="coerce")
            )
            * 100.0,
        }
    )
    inference = _contrast_statistics_values(
        paired,
        endpoint=endpoint,
        contrast=endpoint,
        unit="percent",
        p_adjust_family="fig6e_targeted_removal",
    )
    statistics = build_statistics(
        pd.concat([descriptive, inference], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "condition",
        ),
        extra_data={"panel_e_exact_match_coverage.csv": coverage},
        exclusion_reason=(
            "trials without exact removed-area and removed-input-energy equality are "
            "excluded before within-network aggregation; all-trial estimates remain "
            "outside the main panel"
        ),
    )


def _build_fig6f(ctx: BuilderContext) -> PanelResult:
    panel_id = "f"
    source = _load(
        ctx,
        panel_id,
        SourceDescriptor(
            key="fig6.overlap_interaction",
            pattern=(
                "results/paper_figure_multi_seed/fig6_peak_amplified_reentry/"
                "seed_*/data/metrics/panel_e_overlap_gated_stsp_interaction.csv"
            ),
            source_level="intermediate",
            producer_task="overlap-gated STSP recruitment interaction",
            filters=(
                "stsp_group_quantile=0.50; overlap_threshold=0.05; "
                "early_window_ms=5,10,15,20"
            ),
            held_fixed="row-wise complete 2x2 high/low STSP x overlap/no-overlap",
            aggregation_path=(
                "sites -> sequence x probe interaction -> "
                "network x early-window mean"
            ),
            required_columns=(
                "network_seed",
                "sequence_id",
                "probe_id",
                "early_window_ms",
                "stsp_group_quantile",
                "overlap_threshold",
                "stsp_effect_with_overlap",
                "stsp_effect_without_overlap",
                "interaction_delta",
                "high_overlap_delta",
                "low_overlap_delta",
                "high_nooverlap_delta",
                "low_nooverlap_delta",
            ),
        ),
    )
    selected = source.frame.loc[
        np.isclose(
            pd.to_numeric(
                source.frame["stsp_group_quantile"],
                errors="coerce",
            ),
            0.50,
            rtol=0.0,
            atol=1e-12,
        )
        & np.isclose(
            pd.to_numeric(
                source.frame["overlap_threshold"],
                errors="coerce",
            ),
            0.05,
            rtol=0.0,
            atol=1e-12,
        )
        & source.frame["early_window_ms"].isin([5, 10, 15, 20])
    ].copy()
    complete_columns = (
        "stsp_effect_with_overlap",
        "stsp_effect_without_overlap",
        "interaction_delta",
        "high_overlap_delta",
        "low_overlap_delta",
        "high_nooverlap_delta",
        "low_nooverlap_delta",
    )
    complete_rows = (
        selected.loc[:, list(complete_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
    )
    selected = selected.loc[complete_rows].copy()
    if selected.empty:
        raise ValueError("fig6f: no row-wise complete 2x2 interaction rows remain")
    _validate_interaction_identity(selected)
    interaction_network = selected.groupby(
        ["network_seed", "early_window_ms"],
        as_index=False,
    )["interaction_delta"].mean()
    interaction_network["interaction_percentage_points"] = (
        pd.to_numeric(
            interaction_network["interaction_delta"], errors="coerce"
        )
        * 100.0
    )
    robustness_plot = make_plot_data(
        interaction_network,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_interaction",
        endpoint="overlap_gated_stsp_interaction",
        condition="overlap_interaction",
        value="interaction_percentage_points",
        unit="percentage_points",
        dimensions=("early_window_ms",),
    ).sort_values(
        ["network_seed", "early_window_ms"],
        kind="mergesort",
    )
    main = selected.loc[selected["early_window_ms"].eq(10)].copy()
    cell_columns = (
        "high_overlap_delta",
        "low_overlap_delta",
        "high_nooverlap_delta",
        "low_nooverlap_delta",
    )
    cell_network = main.groupby("network_seed", as_index=False)[
        list(cell_columns)
    ].mean()
    cell_network.loc[:, list(cell_columns)] = (
        cell_network.loc[:, list(cell_columns)]
        .apply(pd.to_numeric, errors="coerce")
        * 100.0
    )
    primary_cells = cell_network.melt(
        id_vars=("network_seed",),
        value_vars=cell_columns,
        var_name="cell",
        value_name="firing_delta_percentage_points",
    )
    cell_mapping = {
        "high_nooverlap_delta": ("high", "no_overlap"),
        "high_overlap_delta": ("high", "overlap"),
        "low_nooverlap_delta": ("low", "no_overlap"),
        "low_overlap_delta": ("low", "overlap"),
    }
    primary_cells["stsp_group"] = primary_cells["cell"].map(
        lambda value: cell_mapping[str(value)][0]
    )
    primary_cells["overlap_group"] = primary_cells["cell"].map(
        lambda value: cell_mapping[str(value)][1]
    )
    primary_cells["early_window_ms"] = 10
    plot = make_plot_data(
        primary_cells,
        figure_id=ctx.figure_id,
        panel_id=panel_id,
        record_type="network_2x2_cell",
        endpoint="early_firing_delta",
        condition="cell",
        value="firing_delta_percentage_points",
        unit="percentage_points",
        dimensions=(
            "cell",
            "stsp_group",
            "overlap_group",
            "early_window_ms",
        ),
    )
    plot["condition"] = primary_cells["cell"].to_numpy()
    plot = plot.sort_values(
        ["network_seed", "overlap_group", "stsp_group"],
        kind="mergesort",
    )
    descriptive = _statistics_values(
        plot,
        group_columns=("cell",),
        status="descriptive_only",
    )
    primary_interaction = robustness_plot.loc[
        robustness_plot["early_window_ms"].eq(10),
        ["network_seed", "value"],
    ].copy()
    inference = _contrast_statistics_values(
        primary_interaction,
        endpoint="overlap_gated_stsp_interaction",
        contrast="stsp_effect_with_overlap_minus_without_overlap_at_10ms",
        unit="percentage_points",
        p_adjust_family="fig6f_primary_interaction",
    )
    statistics = build_statistics(
        pd.concat([descriptive, inference], ignore_index=True, sort=False),
        figure_id=ctx.figure_id,
        panel_id=panel_id,
    )
    return _finalize_quantitative_panel(
        ctx,
        panel_id,
        plot,
        statistics,
        source.records,
        input_rows=len(source.frame),
        unique_key=(
            "figure_id",
            "panel_id",
            "network_seed",
            "endpoint",
            "cell",
        ),
        extra_data={
            "panel_f_window_robustness.csv": robustness_plot,
        },
        exclusion_reason=(
            "rows lacking any 2x2 cell, effect, or interaction value were excluded "
            "before aggregation; the main plot displays the prespecified 10-ms 2x2 "
            "cells, while 5-, 15-, and 20-ms interactions remain auxiliary robustness "
            "data and inference remains restricted to 10 ms"
        ),
    )


def _validate_interaction_identity(frame: pd.DataFrame) -> None:
    numeric = frame.copy()
    for column in (
        "high_overlap_delta",
        "low_overlap_delta",
        "high_nooverlap_delta",
        "low_nooverlap_delta",
        "stsp_effect_with_overlap",
        "stsp_effect_without_overlap",
        "interaction_delta",
    ):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    expected_with = (
        pd.to_numeric(numeric["high_overlap_delta"], errors="coerce")
        - pd.to_numeric(numeric["low_overlap_delta"], errors="coerce")
    )
    expected_without = (
        pd.to_numeric(numeric["high_nooverlap_delta"], errors="coerce")
        - pd.to_numeric(numeric["low_nooverlap_delta"], errors="coerce")
    )
    expected_interaction = expected_with - expected_without
    checks = (
        (
            pd.to_numeric(numeric["stsp_effect_with_overlap"], errors="coerce"),
            expected_with,
            "with-overlap effect",
        ),
        (
            pd.to_numeric(numeric["stsp_effect_without_overlap"], errors="coerce"),
            expected_without,
            "without-overlap effect",
        ),
        (
            pd.to_numeric(numeric["interaction_delta"], errors="coerce"),
            expected_interaction,
            "interaction",
        ),
    )
    for observed, expected, label in checks:
        if not np.allclose(
            observed.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"fig6f: source {label} identity failed")


FIGURE_BUILDERS: dict[str, Callable[[BuilderContext], list[PanelResult]]] = {
    "fig1": build_fig1,
    "fig2": build_fig2,
    "fig3": build_fig3,
    "fig4": build_fig4,
    "fig5": build_fig5,
    "fig6": build_fig6,
}
