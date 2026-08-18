from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .common import EXPECTED_NETWORK_SEEDS, SourceBuildContext


FIGURE_BUILDERS: dict[str, Callable[[SourceBuildContext], None]] = {}


def _register(figure_id: str):
    def decorator(function: Callable[[SourceBuildContext], None]):
        FIGURE_BUILDERS[figure_id] = function
        return function

    return decorator


def _map28(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.shape == (28, 28):
        return values
    if values.size != 2 * 28 * 28:
        raise ValueError(f"Expected a 28x28 map or two 28x28 channels, got {values.shape}")
    return values.reshape(2, 28, 28).mean(axis=0)


def _centered_cosine(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=float).reshape(-1)
    y = np.asarray(second, dtype=float).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def _moran_rook(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=float)
    centered = values - values.mean()
    denominator = float(np.square(centered).sum())
    if denominator <= 0:
        return float("nan")
    cross = float(
        (centered[:, :-1] * centered[:, 1:]).sum()
        + (centered[:-1, :] * centered[1:, :]).sum()
    )
    n_pixels = centered.size
    undirected_edges = centered.shape[0] * (centered.shape[1] - 1) + (centered.shape[0] - 1) * centered.shape[1]
    return float((n_pixels / (2 * undirected_edges)) * (2 * cross / denominator))


def _recut_transition(first_dynamic: pd.Series, first_static: pd.Series, window_ms: int) -> pd.Series:
    dynamic = first_dynamic.where((first_dynamic >= 0) & (first_dynamic < int(window_ms)), -1)
    static = first_static.where((first_static >= 0) & (first_static < int(window_ms)), -1)
    return ((dynamic >= 0) & (((static >= 0) & (dynamic < static)) | (static < 0))).astype(float)


@_register("s1")
def build_s1(context: SourceBuildContext) -> None:
    root = "fig1_functional_stsp_substrate/fig1_functional_stsp_substrate/seed_*/data"
    delay = context.read_many(
        f"{root}/raw/supp_dms_delay_sweep_trial_readout.csv",
        usecols=[
            "network_seed",
            "trial_id",
            "delay_ms",
            "condition",
            "is_correct_probe",
            "pred_is_original_sample",
        ],
    )
    delay_network = delay.groupby(["network_seed", "delay_ms", "condition"], as_index=False).agg(
        accuracy=("is_correct_probe", "mean"),
        sample_rate=("pred_is_original_sample", "mean"),
    )
    context.require_networks(delay_network, label="S1 delay sweep")
    pivot = delay_network.pivot(index=["network_seed", "delay_ms"], columns="condition")
    contrasts = pd.DataFrame(
        {
            "accuracy_gap": pivot["accuracy"]["static_frozen"] - pivot["accuracy"]["dynamic_intact"],
            "sample_bias": pivot["sample_rate"]["dynamic_intact"] - pivot["sample_rate"]["static_frozen"],
        }
    ).reset_index()
    panel_a = contrasts[["network_seed", "delay_ms", "accuracy_gap"]].rename(columns={"accuracy_gap": "value"})
    panel_a["value"] *= 100.0
    panel_b = contrasts[["network_seed", "delay_ms", "sample_bias"]].rename(columns={"sample_bias": "value"})
    panel_b["value"] *= 100.0
    context.write_panel("s1", "a", panel_a)
    context.write_panel("s1", "b", panel_b)
    context.add_summaries("s1", "a", panel_a, groups=["delay_ms"])
    context.add_summaries("s1", "b", panel_b, groups=["delay_ms"])
    a_pivot = panel_a.pivot(index="network_seed", columns="delay_ms", values="value")
    b_pivot = panel_b.pivot(index="network_seed", columns="delay_ms", values="value")
    context.add_test("s1", "a", "100_minus_1200_accuracy_gap", a_pivot[100] - a_pivot[1200], family="holm4")
    context.add_test("s1", "b", "100_minus_1200_sample_bias", b_pivot[100] - b_pivot[1200], family="holm4")

    donor = context.read_many(
        f"{root}/raw/panel_d_dms_condition_trial_readout.csv",
        usecols=[
            "network_seed",
            "trial_id",
            "condition",
            "sample_label",
            "probe_label",
            "donor_sample_label",
            "prediction",
        ],
    )
    required_conditions = {"dynamic_intact", "ux_trial_shuffle"}
    if not required_conditions.issubset(set(donor["condition"].astype(str))):
        raise ValueError("S1 donor trial table is missing dynamic_intact or ux_trial_shuffle")
    paired = donor.loc[donor["condition"].isin(required_conditions)].pivot(
        index=["network_seed", "trial_id"], columns="condition", values="prediction"
    ).reset_index()
    labels = donor.drop_duplicates(["network_seed", "trial_id"])[
        ["network_seed", "trial_id", "sample_label", "probe_label", "donor_sample_label"]
    ]
    paired = paired.merge(labels, on=["network_seed", "trial_id"], validate="one_to_one")
    intact = "dynamic_intact"
    reconstructed = "ux_trial_shuffle"
    paired["Inflow"] = (
        (paired[intact] != paired["donor_sample_label"])
        & (paired[reconstructed] == paired["donor_sample_label"])
    ).astype(float)
    paired["Outflow"] = (
        (paired[intact] == paired["donor_sample_label"])
        & (paired[reconstructed] != paired["donor_sample_label"])
    ).astype(float)
    flux_network = paired.groupby("network_seed", as_index=False)[["Inflow", "Outflow"]].mean()
    flux_network["Net"] = flux_network["Inflow"] - flux_network["Outflow"]
    panel_c = flux_network.melt(id_vars="network_seed", var_name="endpoint", value_name="value")
    panel_c["value"] *= 100.0
    context.write_panel("s1", "c", panel_c)
    context.add_summaries("s1", "c", panel_c, groups=["endpoint"])
    context.add_test(
        "s1", "c", "paired_net_donor_flux", flux_network.set_index("network_seed")["Net"] * 100.0, family="holm4"
    )

    calibration_rows: list[dict[str, float | int | str]] = []
    for seed, seed_rows in donor.groupby("network_seed", sort=True):
        run_config = context.source_root / (
            "fig1_functional_stsp_substrate/fig1_functional_stsp_substrate/"
            f"seed_{int(seed)}/run_config.json"
        )
        config = json.loads(context.track(run_config).read_text(encoding="utf-8"))
        batch_size = int(config["dms_batch_size"])
        receivers = seed_rows.drop_duplicates("trial_id").sort_values("trial_id").copy()
        receivers["batch_id"] = np.arange(len(receivers), dtype=int) // batch_size
        for condition in (intact, reconstructed):
            part = seed_rows.loc[seed_rows["condition"].eq(condition)].sort_values("trial_id")
            observed_values: list[float] = []
            null_values: list[float] = []
            for row in part.itertuples(index=False):
                receiver = receivers.loc[receivers["trial_id"].eq(int(row.trial_id))].iloc[0]
                candidates = receivers.loc[receivers["batch_id"].eq(int(receiver["batch_id"]))]
                candidate_labels = candidates["sample_label"].to_numpy(dtype=int)
                candidate_ids = candidates["trial_id"].to_numpy(dtype=int)
                eligible = (
                    (candidate_ids != int(row.trial_id))
                    & (candidate_labels != int(row.sample_label))
                    & (candidate_labels != int(row.probe_label))
                )
                if not bool(eligible.any()):
                    raise ValueError(f"S1 donor opportunity has no eligible source: seed={seed}, trial={row.trial_id}")
                observed_values.append(float(int(row.prediction) == int(row.donor_sample_label)))
                null_values.append(float(np.mean(candidate_labels[eligible] == int(row.prediction))))
            observed = float(np.mean(observed_values))
            null = float(np.mean(null_values))
            calibration_rows.append(
                {
                    "network_seed": int(seed),
                    "condition": condition,
                    "observed": 100.0 * observed,
                    "opportunity_null": 100.0 * null,
                    "excess": 100.0 * (observed - null),
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    excess = calibration.pivot(index="network_seed", columns="condition", values="excess")
    panel_d = (excess[reconstructed] - excess[intact]).rename("value").reset_index()
    context.write_panel("s1", "d", panel_d)
    context.write_panel("s1", "d", calibration, suffix="_calibration")
    context.add_summaries("s1", "d", panel_d)
    context.add_summaries("s1", "d", calibration, value="excess", groups=["condition"], role="calibration")
    context.add_test("s1", "d", "difference_in_donor_excess", panel_d["value"], family="holm4")


def _fixed_endpoint_row(inference: pd.DataFrame, endpoint: str) -> pd.Series:
    rows = inference.loc[(inference["endpoint"] == endpoint) & (inference["prefix_k"] == 1)]
    if len(rows) != 1:
        raise ValueError(f"Expected one fixed-B inference row for {endpoint}, found {len(rows)}")
    return rows.iloc[0]


def _c5_inference_row(
    inference: pd.DataFrame,
    *,
    cohort: str,
    endpoint: str,
    prefix_k: int,
) -> pd.Series:
    rows = inference.loc[
        inference["cohort"].eq(cohort)
        & inference["endpoint"].eq(endpoint)
        & inference["prefix_k"].eq(int(prefix_k))
    ]
    if len(rows) != 1:
        raise ValueError(
            "Expected one C5 inference row for "
            f"cohort={cohort}, endpoint={endpoint}, K={prefix_k}; found {len(rows)}"
        )
    row = rows.iloc[0]
    return pd.Series(
        {
            "n_networks": int(row["n_networks"]),
            "mean": float(row["mean_transfer"]),
            "ci95_low": float(row["bootstrap_ci95_low"]),
            "ci95_high": float(row["bootstrap_ci95_high"]),
            "threshold": 0.0,
            "fraction_above_zero": float(row["positive_network_fraction"]),
            "p_one_sided": float(row["p_one_sided_exact_sign_flip"]),
            "holm_adjusted_p": float(row["holm_adjusted_p"]),
        }
    )


@_register("s2")
def build_s2(context: SourceBuildContext) -> None:
    root = "fig2_fixed_b_mechanism_confirmatory"
    swap = context.read_many(
        f"{root}/seed_*/data/metrics/fixed_b_swap_summary.csv",
        seed_from_path=True,
    )
    gates = context.read_many(f"{root}/seed_*/data/metrics/fixed_b_engineering_gates.csv")
    context.require_networks(swap, label="S2 fixed-B summaries")
    context.require_networks(gates, label="S2 engineering gates")
    selected = swap.loc[
        (swap["prefix_k"] == 1)
        & swap["swap_scope"].eq("layer1_only")
        & swap["endpoint"].isin(["layer2_update", "early_class_score"])
    ].copy()
    if len(selected) != 40:
        raise ValueError(f"S2 expected 40 primary network rows, found {len(selected)}")

    gate_roles = {
        "exact_B_hash_identity": "identical current input",
        "layer1_only_swap_isolation": "selective boundary and fast-state equality",
        "full_trace_content_hash": "event-trace integrity",
        "layer1_update_donor_valid_coverage": "downstream update coverage",
        "layer1_early_score_donor_valid_coverage": "early-score coverage",
    }
    audit_parts: list[pd.DataFrame] = []
    for source_gate, gate_role in gate_roles.items():
        part = gates.loc[
            gates["gate"].eq(source_gate), ["network_seed", "passed"]
        ].copy()
        if len(part) != 20 or part["network_seed"].nunique() != 20:
            raise ValueError(f"S2 engineering gate is incomplete: {source_gate}")
        audit_parts.append(
            part.assign(source_gate=source_gate, gate_role=gate_role)
        )
    panel_a_audit = pd.concat(audit_parts, ignore_index=True)
    panel_a_audit["valid_count"] = 20
    for source_gate, endpoint in (
        ("layer1_update_donor_valid_coverage", "layer2_update"),
        ("layer1_early_score_donor_valid_coverage", "early_class_score"),
    ):
        counts = selected.loc[
            selected["endpoint"].eq(endpoint), ["network_seed", "n_valid"]
        ].set_index("network_seed")["n_valid"]
        selector = panel_a_audit["source_gate"].eq(source_gate)
        panel_a_audit.loc[selector, "valid_count"] = panel_a_audit.loc[
            selector, "network_seed"
        ].map(counts)
    if not bool(panel_a_audit["passed"].astype(bool).all()):
        raise ValueError("S2 causal identity hard gate failed")

    panel_a = pd.DataFrame(
        [
            {
                "element_id": "donor_component",
                "component": "Layer 1 u/x",
                "owner_before": "donor",
                "owner_after": "receiver",
                "operation": "transfer",
                "display_order": 1,
            },
            {
                "element_id": "receiver_component",
                "component": "Layer 1 u/x",
                "owner_before": "receiver",
                "owner_after": "displaced",
                "operation": "replace",
                "display_order": 2,
            },
            {
                "element_id": "receiver_after",
                "component": "receiver carrier + Layer 2/3 state",
                "owner_before": "receiver",
                "owner_after": "receiver",
                "operation": "retain_carrier",
                "display_order": 3,
            },
        ]
    )
    context.write_panel("s2", "a", panel_a)
    context.write_panel("s2", "a", panel_a_audit, suffix="_identity_audit")

    inference_path = context.source_root / root / "aggregate" / "fixed_b_confirmatory_inference.csv"
    untouched_path = context.source_root / root / "aggregate" / "fixed_b_untouched_confirmatory_inference.csv"
    inference = context.read_csv(inference_path)
    untouched = context.read_csv(untouched_path)
    endpoint_specs = [
        ("layer2_update", "L2 update", "layer1_only_layer2_update_donor_transfer"),
        ("early_class_score", "Early L3", "layer1_only_early_class_score_donor_transfer"),
    ]
    b_rows: list[pd.DataFrame] = []
    for endpoint, layer, inference_endpoint in endpoint_specs:
        panel = selected.loc[selected["endpoint"].eq(endpoint), ["network_seed", "mean_donor_transfer_index"]].rename(
            columns={"mean_donor_transfer_index": "value"}
        )
        b_rows.append(panel.assign(layer=layer))
        source_row = _fixed_endpoint_row(inference, inference_endpoint)
        context.add_frozen_summary("s2", "b", metric="value", source_row=source_row, groups={"layer": layer})
        context.add_frozen_test("s2", "b", inference_endpoint, family="confirmatory_core8", source_row=source_row)
    panel_b = pd.concat(b_rows, ignore_index=True)
    context.write_panel("s2", "b", panel_b)

    metric_specs = [
        ("Median DTI", "median_donor_transfer_index", 0.0),
        ("Alignment", "mean_effect_alignment_cosine", 0.0),
        ("Positive frac.", "fraction_positive", 0.5),
    ]
    c_rows: list[pd.DataFrame] = []
    for endpoint, layer in (("layer2_update", "L2 update"), ("early_class_score", "Early L3")):
        endpoint_rows = selected.loc[selected["endpoint"].eq(endpoint)].copy()
        for metric_label, column, null in metric_specs:
            c_rows.append(
                endpoint_rows[["network_seed", column]]
                .rename(columns={column: "raw_value"})
                .assign(layer=layer, metric=metric_label, metric_null=float(null), value=lambda data: data["raw_value"] - float(null))
            )
    panel_c = pd.concat(c_rows, ignore_index=True)
    context.write_panel("s2", "c", panel_c)
    context.add_summaries("s2", "c", panel_c, groups=["layer", "metric"])
    for (layer, metric), part in panel_c.groupby(["layer", "metric"], sort=True):
        context.add_test(
            "s2",
            "c",
            f"{layer}_{metric}_above_null",
            part["value"],
            family="secondary_holm6",
            alternative="greater",
        )

    d_rows: list[pd.DataFrame] = []
    for endpoint, layer, inference_endpoint in endpoint_specs:
        full = selected.loc[selected["endpoint"].eq(endpoint), ["network_seed", "mean_donor_transfer_index"]].rename(
            columns={"mean_donor_transfer_index": "value"}
        )
        d_rows.append(full.assign(layer=layer, cohort="Full 20"))
        d_rows.append(full.loc[full["network_seed"] != 1000].assign(layer=layer, cohort="Untouched 19"))
        context.add_frozen_summary(
            "s2",
            "d",
            metric="value",
            source_row=_fixed_endpoint_row(inference, inference_endpoint),
            groups={"layer": layer, "cohort": "Full 20"},
            role="reference",
        )
        untouched_row = _fixed_endpoint_row(untouched, inference_endpoint)
        context.add_frozen_summary(
            "s2",
            "d",
            metric="value",
            source_row=untouched_row,
            groups={"layer": layer, "cohort": "Untouched 19"},
        )
        context.add_frozen_test(
            "s2", "d", f"untouched_{inference_endpoint}", family="untouched_core8", source_row=untouched_row
        )
    panel_d = pd.concat(d_rows, ignore_index=True)
    context.write_panel("s2", "d", panel_d)


@_register("s3")
def build_s3(context: SourceBuildContext) -> None:
    root = "fig5_local_support_competition/seed_*/data/metrics"
    a_rows: list[dict[str, float | int | str]] = []
    for path in context.paths(f"{root}/panel_b_early_firing_transition_metrics.csv"):
        frame = pd.read_csv(
            path,
            usecols=["network_seed", "trial_id", "unit_group", "first_spike_dynamic", "first_spike_static"],
        )
        seed = int(frame["network_seed"].iloc[0])
        for window in (5, 10, 15, 20, 30):
            temp = frame.assign(value=_recut_transition(frame["first_spike_dynamic"], frame["first_spike_static"], window))
            temp = temp.groupby(["trial_id", "unit_group"], as_index=False)["value"].mean()
            group_means = temp.groupby("unit_group")["value"].mean()
            a_rows.extend(
                [
                    {
                        "network_seed": seed,
                        "window_ms": window,
                        "comparator": "Probe-only",
                        "value": 100.0 * float(group_means["overlap_dominant"] - group_means["probe_only_dominant"]),
                    },
                    {
                        "network_seed": seed,
                        "window_ms": window,
                        "comparator": "Random",
                        "value": 100.0 * float(group_means["overlap_dominant"] - group_means["random_matched"]),
                    },
                ]
            )
    panel_a = pd.DataFrame(a_rows)
    context.require_networks(panel_a, label="S3 true-window recut")
    context.write_panel("s3", "a", panel_a)
    context.add_summaries("s3", "a", panel_a, groups=["comparator", "window_ms"])
    minima = panel_a.groupby(["network_seed", "comparator"])["value"].min().unstack("comparator")
    context.add_test("s3", "a", "minimum_overlap_minus_probe", minima["Probe-only"], family="holm6")
    context.add_test("s3", "a", "minimum_overlap_minus_random", minima["Random"], family="holm6")

    b_rows: list[dict[str, float | int]] = []
    c_rows: list[dict[str, float | int]] = []
    for path in context.paths(f"{root}/panel_c_winner_loser_event_metrics.csv"):
        frame = pd.read_csv(
            path,
            usecols=[
                "network_seed",
                "trial_id",
                "selection_rank_within_trial",
                "local_distance",
                "winner_minus_loser_full_pre_delta_v_mean",
                "complete_alignment_window",
            ],
        )
        frame = frame.loc[frame["complete_alignment_window"].astype(bool)]
        seed = int(frame["network_seed"].iloc[0])
        for cap in (1, 2, 3):
            part = frame.loc[frame["selection_rank_within_trial"] <= cap]
            value = part.groupby("trial_id")["winner_minus_loser_full_pre_delta_v_mean"].mean().mean()
            b_rows.append({"network_seed": seed, "cap": cap, "value": 1000.0 * float(value)})
        for radius in (1, 2, 3):
            part = frame.loc[frame["local_distance"] <= 2 * radius]
            value = part.groupby("trial_id")["winner_minus_loser_full_pre_delta_v_mean"].mean().mean()
            c_rows.append({"network_seed": seed, "distance_limit": 2 * radius, "value": 1000.0 * float(value)})
    panel_b = pd.DataFrame(b_rows)
    panel_c = pd.DataFrame(c_rows)
    context.require_networks(panel_b, label="S3 winner-cap robustness")
    context.require_networks(panel_c, label="S3 distance robustness")
    context.write_panel("s3", "b", panel_b)
    context.write_panel("s3", "c", panel_c)
    context.add_summaries("s3", "b", panel_b, groups=["cap"])
    context.add_summaries("s3", "c", panel_c, groups=["distance_limit"])
    context.add_test("s3", "b", "minimum_across_winner_caps", panel_b.groupby("network_seed")["value"].min(), family="holm6")
    context.add_test(
        "s3", "c", "minimum_across_distance_limits", panel_c.groupby("network_seed")["value"].min(), family="holm6"
    )

    d_rows: list[dict[str, float | int | str]] = []
    conditions = {"dynamic_intact", "attenuate_l1_stsp", "reset_l1_stsp"}
    for path in context.paths(f"{root}/panel_d_l1_stsp_perturbation_unit_transitions.csv"):
        frame = pd.read_csv(
            path,
            usecols=[
                "network_seed",
                "trial_id",
                "condition",
                "unit_id",
                "unit_group",
                "included_in_main",
                "first_spike_static",
                "first_spike_condition",
            ],
        )
        frame = frame.loc[frame["condition"].isin(conditions) & frame["included_in_main"].astype(bool)]
        dynamic = frame.loc[frame["condition"].eq("dynamic_intact")].copy()
        dynamic_first = dynamic["first_spike_condition"].where(dynamic["first_spike_condition"].between(0, 49), -1)
        static_first = dynamic["first_spike_static"].where(dynamic["first_spike_static"].between(0, 49), -1)
        dynamic["winner"] = (dynamic_first >= 0) & (
            ((static_first >= 0) & (dynamic_first < static_first)) | (static_first < 0)
        )
        dynamic = dynamic.loc[
            dynamic["winner"],
            ["network_seed", "trial_id", "unit_id", "unit_group", "first_spike_condition"],
        ]
        seed = int(dynamic["network_seed"].iloc[0])
        for source_condition, label in (("attenuate_l1_stsp", "Attenuate"), ("reset_l1_stsp", "Reset")):
            perturbed = frame.loc[
                frame["condition"].eq(source_condition),
                ["network_seed", "trial_id", "unit_id", "unit_group", "first_spike_condition"],
            ]
            joined = dynamic.merge(
                perturbed,
                on=["network_seed", "trial_id", "unit_id", "unit_group"],
                suffixes=("_dynamic", "_perturbed"),
                validate="one_to_one",
            )
            lost = joined["first_spike_condition_perturbed"] < 0
            delayed = (~lost) & (
                joined["first_spike_condition_perturbed"] > joined["first_spike_condition_dynamic"]
            )
            trial_rates = pd.DataFrame(
                {
                    "trial_id": joined["trial_id"],
                    "Lost": lost.astype(float),
                    "Delayed": delayed.astype(float),
                    "Preserved": (~(lost | delayed)).astype(float),
                }
            ).groupby("trial_id").mean()
            for fate in ("Lost", "Delayed", "Preserved"):
                d_rows.append(
                    {
                        "network_seed": seed,
                        "condition": label,
                        "fate": fate,
                        "value": 100.0 * float(trial_rates[fate].mean()),
                        "n_dynamic_winners": int(len(joined)),
                    }
                )
    panel_d = pd.DataFrame(d_rows)
    totals = panel_d.groupby(["network_seed", "condition"])["value"].sum()
    if not np.allclose(totals.to_numpy(), 100.0, atol=1e-8):
        raise ValueError("S3 winner fates are not mutually exclusive and exhaustive")
    context.require_networks(panel_d, label="S3 original-winner fate")
    context.write_panel("s3", "d", panel_d)
    context.add_summaries("s3", "d", panel_d, groups=["condition", "fate"])
    disrupted = panel_d.loc[panel_d["fate"].isin(["Lost", "Delayed"])].groupby(
        ["network_seed", "condition"]
    )["value"].sum().unstack("condition")
    disrupted_long = disrupted.stack().rename("value").reset_index()
    disrupted_long["fate"] = "Disrupted"
    context.add_summaries("s3", "d", disrupted_long, groups=["condition", "fate"], role="cumulative")
    context.add_test("s3", "d", "attenuation_disrupted_fraction", disrupted["Attenuate"], family="holm6")
    context.add_test("s3", "d", "reset_minus_attenuation_disruption", disrupted["Reset"] - disrupted["Attenuate"], family="holm6")

    # The K=10 intervention is a persisted extension parent; controls are structural and descriptive.
    extension_root = context.repo_root / "results" / "successor_extension_v1_confirmatory_20seed"
    extension_paths = sorted(
        extension_root.glob("seed_*/data/metrics/exp_b_k10_l1_overlap_intervention/exp_b_network_summary.csv")
    )
    if len(extension_paths) != len(EXPECTED_NETWORK_SEEDS):
        raise ValueError(f"S3 expected 20 persisted K=10 intervention summaries, found {len(extension_paths)}")
    extension = pd.concat([context.read_csv(path) for path in extension_paths], ignore_index=True)
    context.require_networks(extension, label="S3 K=10 intervention summaries")
    extension_inference = context.read_csv(extension_root / "aggregate" / "population_inference.csv")
    endpoint_specs = (
        ("e", "early_layer2_b_history_contrast_attenuation", "Input response", "layer2"),
        ("f", "post_b_layer2_ux_history_contrast_attenuation", "Successor state", "layer3"),
    )
    condition_columns = (
        ("Overlap reset", "mean_overlap_attenuation"),
        ("Non-overlap reset", "mean_nonoverlap_attenuation"),
        ("Random reset", "mean_random_attenuation"),
    )
    for panel_id, endpoint, endpoint_label, _ in endpoint_specs:
        subset = extension.loc[extension["endpoint"].eq(endpoint)].copy()
        if len(subset) != len(EXPECTED_NETWORK_SEEDS) or set(subset["network_seed"].astype(int)) != set(EXPECTED_NETWORK_SEEDS):
            raise ValueError(f"S3{panel_id} K=10 intervention summary is not one row per network")
        rows = [
            {
                "network_seed": int(row.network_seed),
                "endpoint": endpoint_label,
                "condition": condition,
                "value": 100.0 * float(getattr(row, column)),
            }
            for row in subset.itertuples(index=False)
            for condition, column in condition_columns
        ]
        panel = pd.DataFrame(rows)
        context.write_panel("s3", panel_id, panel)
        context.add_summaries(
            "s3",
            panel_id,
            panel.loc[~panel["condition"].eq("Overlap reset")],
            groups=["condition", "endpoint"],
        )
        inference_rows = extension_inference.loc[
            extension_inference["cohort"].eq("full20")
            & extension_inference["experiment"].eq("exp_b_k10_l1_overlap_intervention")
            & extension_inference["endpoint"].eq(endpoint)
        ]
        if len(inference_rows) != 1:
            raise ValueError(f"S3{panel_id} expected one full-20 inference row for {endpoint}")
        inference_row = inference_rows.iloc[0]
        frozen = pd.Series(
            {
                "n_networks": int(inference_row["n_networks"]),
                "mean": float(inference_row["mean"] * 100.0),
                "ci95_low": float(inference_row["bootstrap_ci95_low"] * 100.0),
                "ci95_high": float(inference_row["bootstrap_ci95_high"] * 100.0),
                "threshold": 0.0,
                "fraction_above_zero": float(inference_row["positive_network_fraction"]),
                "p_one_sided": float(inference_row["p_one_sided_exact_sign_flip"]),
                "holm_adjusted_p": float(inference_row["holm_adjusted_p"]),
            }
        )
        context.add_frozen_summary(
            "s3",
            panel_id,
            metric="value",
            source_row=frozen,
            groups={"condition": "Overlap reset", "endpoint": endpoint_label},
        )
        context.add_frozen_test(
            "s3",
            panel_id,
            f"full20_{endpoint}",
            family="successor_extension_exp_b_primary2",
            source_row=frozen,
        )


@_register("s4")
def build_s4(context: SourceBuildContext) -> None:
    c5_root = context.repo_root / "results" / "causal_closure_multi_seed_20260803" / "c5_l2_successor"
    extension_root = context.repo_root / "results" / "successor_extension_v1_confirmatory_20seed"
    context.track(c5_root / "multiseed_protocol_freeze.json")
    context.track(c5_root / "C5_20SEED_REPORT_20260803.md")
    context.track(extension_root / "aggregate" / "artifact_manifest.json")
    context.track(extension_root / "aggregate" / "verdict.json")

    c5_aggregate = c5_root / "aggregate"
    c5_effects = context.read_csv(c5_aggregate / "data" / "metrics" / "c5_network_effects.csv")
    c5_inference = context.read_csv(c5_aggregate / "data" / "metrics" / "c5_population_inference.csv")
    c5_engineering = context.read_csv(c5_aggregate / "meta" / "c5_multiseed_engineering_audit.csv")
    context.require_networks(c5_effects, label="S4 C5 network effects")
    context.require_networks(c5_engineering, label="S4 C5 engineering audit")
    if not bool(c5_engineering["engineering_pass"].astype(bool).all()):
        raise ValueError("S4 C5 engineering gate failed")

    extension_effects = context.read_csv(extension_root / "aggregate" / "network_effects.csv")
    extension_inference = context.read_csv(extension_root / "aggregate" / "population_inference.csv")
    k10_summary_paths = sorted(
        extension_root.glob("seed_*/data/metrics/exp_a_c5_k10_successor/c5_k10_endpoint_summary.csv")
    )
    if len(k10_summary_paths) != len(EXPECTED_NETWORK_SEEDS):
        raise ValueError(f"S4 expected 20 K=10 endpoint summaries, found {len(k10_summary_paths)}")
    k10_summaries = pd.concat([context.read_csv(path) for path in k10_summary_paths], ignore_index=True)
    context.require_networks(k10_summaries, label="S4 K=10 endpoint summaries")

    def extension_row(experiment: str, endpoint: str) -> pd.Series:
        rows = extension_inference.loc[
            extension_inference["cohort"].eq("full20")
            & extension_inference["experiment"].eq(experiment)
            & extension_inference["endpoint"].eq(endpoint)
        ]
        if len(rows) != 1:
            raise ValueError(f"S4 expected one extension inference row for {experiment}/{endpoint}")
        row = rows.iloc[0]
        return pd.Series(
            {
                "n_networks": int(row["n_networks"]),
                "mean": float(row["mean"]),
                "ci95_low": float(row["bootstrap_ci95_low"]),
                "ci95_high": float(row["bootstrap_ci95_high"]),
                "threshold": 0.0,
                "fraction_above_zero": float(row["positive_network_fraction"]),
                "p_one_sided": float(row["p_one_sided_exact_sign_flip"]),
                "holm_adjusted_p": float(row["holm_adjusted_p"]),
            }
        )

    endpoint_specs = (
        ("a", "early_layer2_event_map_donor_transfer", "Input response", "layer2"),
        ("b", "layer3_successor_ux_donor_transfer", "Successor state", "layer3"),
    )
    for panel_id, endpoint, endpoint_label, _ in endpoint_specs:
        c5_panel = c5_effects.loc[
            c5_effects["endpoint"].eq(endpoint) & c5_effects["prefix_k"].isin([1, 5]),
            ["network_seed", "prefix_k", "mean_transfer", "positive_fraction", "valid_coverage"],
        ].rename(columns={"mean_transfer": "value"})
        k10_technical = endpoint
        extension_panel = extension_effects.loc[
            extension_effects["cohort"].eq("full20")
            & extension_effects["experiment"].eq("exp_a_c5_k10_successor")
            & extension_effects["endpoint"].eq(k10_technical),
            ["network_seed", "value"],
        ].copy()
        k10_quality = k10_summaries.loc[
            k10_summaries["endpoint"].eq(k10_technical),
            ["network_seed", "positive_fraction", "valid_coverage"],
        ]
        extension_panel = extension_panel.merge(k10_quality, on="network_seed", validate="one_to_one")
        extension_panel["prefix_k"] = 10
        panel = pd.concat([c5_panel, extension_panel], ignore_index=True)
        if len(panel) != 60 or set(panel["network_seed"].astype(int)) != set(EXPECTED_NETWORK_SEEDS):
            raise ValueError(f"S4{panel_id} expected 60 full-20 network/depth rows")
        if not bool((panel["value"] > 0).all()):
            raise ValueError(f"S4{panel_id} contains a non-positive full-20 transfer value")
        panel["endpoint"] = endpoint_label
        context.write_panel("s4", panel_id, panel)
        for prefix_k in (1, 5):
            frozen = _c5_inference_row(
                c5_inference,
                cohort="all_20",
                endpoint=endpoint,
                prefix_k=prefix_k,
            )
            context.add_frozen_summary(
                "s4", panel_id, metric="value", source_row=frozen, groups={"prefix_k": prefix_k, "endpoint": endpoint_label}
            )
            context.add_frozen_test(
                "s4", panel_id, f"all20_{endpoint}_K{prefix_k}", family="c5_primary4", source_row=frozen
            )
        frozen = extension_row("exp_a_c5_k10_successor", endpoint)
        context.add_frozen_summary(
            "s4", panel_id, metric="value", source_row=frozen, groups={"prefix_k": 10, "endpoint": endpoint_label}
        )
        context.add_frozen_test(
            "s4",
            panel_id,
            f"all20_{endpoint}_K10",
            family="successor_extension_exp_a_primary2",
            source_row=frozen,
        )

    # The following-transition panel is a separate persisted primary experiment.
    following_specs = (
        ("early_layer2_D_donor_transfer", "Input response"),
        ("layer3_postD_ux_donor_transfer", "Successor state"),
    )
    following_rows: list[pd.DataFrame] = []
    for endpoint, endpoint_label in following_specs:
        subset = extension_effects.loc[
            extension_effects["cohort"].eq("full20")
            & extension_effects["experiment"].eq("exp_c_c5_twohop_cd")
            & extension_effects["endpoint"].eq(endpoint),
            ["network_seed", "value"],
        ].copy()
        if len(subset) != len(EXPECTED_NETWORK_SEEDS) or set(subset["network_seed"].astype(int)) != set(EXPECTED_NETWORK_SEEDS):
            raise ValueError(f"S4c following-transition endpoint is not one row per network: {endpoint}")
        following_rows.append(subset.assign(endpoint=endpoint_label, transition="Following transition"))
        frozen = extension_row("exp_c_c5_twohop_cd", endpoint)
        context.add_frozen_summary(
            "s4", "c", metric="value", source_row=frozen, groups={"endpoint": endpoint_label, "transition": "Following transition"}
        )
        context.add_frozen_test(
            "s4",
            "c",
            f"full20_following_{endpoint}",
            family="successor_extension_exp_c_primary2",
            source_row=frozen,
        )
    panel_c = pd.concat(following_rows, ignore_index=True)
    context.write_panel("s4", "c", panel_c)

    # Paired complete-state and STSP-only attribution remains descriptive.
    cell_paths = sorted(extension_root.glob("seed_*/data/metrics/exp_c_c5_twohop_cd/exp_c_cell_metrics.csv"))
    if len(cell_paths) != len(EXPECTED_NETWORK_SEEDS):
        raise ValueError(f"S4d expected 20 following-transition cell tables, found {len(cell_paths)}")
    cells = pd.concat(
        [
            context.read_csv(
                path,
            )
            for path in cell_paths
        ],
        ignore_index=True,
    )
    context.require_networks(cells, label="S4 following-transition attribution cells")
    attribution_specs = (
        ("early_layer2_D_donor_transfer", "secondary_stsp_only_early_layer2_D_donor_transfer", "Input response"),
        ("layer3_postD_ux_donor_transfer", "secondary_stsp_only_layer3_postD_ux_donor_transfer", "Successor state"),
    )
    attribution_rows: list[pd.DataFrame] = []
    for primary, secondary, endpoint_label in attribution_specs:
        primary_valid = "early_layer2_D_transfer_valid" if primary.startswith("early_layer2") else "layer3_postD_ux_transfer_valid"
        subset = cells.loc[cells[primary_valid].eq(1) & cells["secondary_transfer_valid"].eq(1)].copy()
        if len(subset) != 240 * len(EXPECTED_NETWORK_SEEDS) // 1:
            raise ValueError(f"S4d unexpected following-transition cell count for {endpoint_label}: {len(subset)}")
        for state, column in (("Complete state", primary), ("STSP only", secondary)):
            values = subset.groupby("network_seed", as_index=False)[column].mean().rename(columns={column: "value"})
            if len(values) != len(EXPECTED_NETWORK_SEEDS):
                raise ValueError(f"S4d expected 20 paired attribution values for {endpoint_label}/{state}")
            attribution_rows.append(values.assign(endpoint=endpoint_label, state=state))
    panel_d = pd.concat(attribution_rows, ignore_index=True)
    context.write_panel("s4", "d", panel_d)
    context.add_summaries("s4", "d", panel_d, groups=["endpoint", "state"])

@_register("s5")
def build_s5(context: SourceBuildContext) -> None:
    root = context.source_root / "new_results_reanalysis" / "metrics"
    stage = context.read_csv(root / "fig4_layer2_progressive_stage_metrics.csv")
    network = context.read_csv(root / "fig4_layer2_progressive_network_metrics.csv")
    context.require_networks(stage, label="S5 progressive stages")
    context.require_networks(network, label="S5 progressive network endpoints")
    joint = stage.loc[
        stage["state_variable"].eq("ux_joint_mean"),
        ["network_seed", "stage_k", "observed_minus_natural_decay"],
    ].rename(columns={"observed_minus_natural_decay": "value"})
    if len(joint) != 180 or not bool((joint["value"] > 0).all()):
        raise ValueError("S5 joint network-by-stage table failed the 180-cell prevalence gate")
    context.write_panel("s5", "a", joint)
    context.add_summaries(
        "s5",
        "a",
        joint,
        groups=["stage_k"],
        ci_method="bootstrap_percentile",
    )
    worst = joint.groupby("network_seed", as_index=False)["value"].min()
    context.write_panel("s5", "b", worst)
    inference = context.read_csv(
        context.source_root
        / "fig4_accumulated_history_statistics"
        / "metrics"
        / "fig4_candidate_inference.csv"
    )
    rows = inference.loc[
        inference["endpoint"].eq("minimum_joint_observed_minus_passive_k2_k10")
    ]
    if len(rows) != 1:
        raise ValueError(f"S5b expected one frozen recurrence inference row, found {len(rows)}")
    row = rows.iloc[0]
    if int(row["n_networks"]) != len(worst) or not np.isclose(
        float(row["mean"]), float(worst["value"].mean()), rtol=0.0, atol=1e-12
    ):
        raise ValueError("S5b frozen recurrence inference does not match the plotted network endpoint")
    frozen = pd.Series(
        {
            "n_networks": int(row["n_networks"]),
            "mean": float(row["mean"]),
            "ci95_low": float(row["ci95_low"]),
            "ci95_high": float(row["ci95_high"]),
            "threshold": float(row["null_value"]),
            "fraction_above_zero": float(row["n_above_null"]) / float(row["n_networks"]),
            "p_one_sided": float(row["p_value"]),
            "holm_adjusted_p": float(row["p_holm_all_new"]),
        }
    )
    context.add_frozen_summary("s5", "b", metric="value", source_row=frozen)
    context.add_frozen_test(
        "s5",
        "b",
        "per_network_worst_stage_joint_displacement",
        family="fig4a_recurrence_all_new",
        source_row=frozen,
    )

    for panel_id, variable in (("c", "u"), ("d", "x")):
        panel = stage.loc[
            stage["state_variable"].eq(variable),
            ["network_seed", "stage_k", "observed_minus_natural_decay"],
        ].rename(columns={"observed_minus_natural_decay": "value"})
        context.write_panel("s5", panel_id, panel)
        context.add_summaries(
            "s5",
            panel_id,
            panel,
            groups=["stage_k"],
            ci_method="bootstrap_percentile",
            bootstrap_seed=20260801,
        )

    endpoints = network.loc[network["state_variable"].isin(["u", "x"])].copy()
    for variable, panel_id in (("u", "c"), ("x", "d")):
        part = endpoints.loc[endpoints["state_variable"].eq(variable)]
        for endpoint, column in (
            ("cross_stage", "mean_observed_minus_decay_k2_k10"),
            ("early_minus_late", "early_minus_late"),
            ("terminal_k10", "terminal_observed_minus_decay"),
        ):
            context.add_test(
                "s5",
                panel_id,
                f"{variable}_{endpoint}",
                part[column],
                family="variable_holm6",
                alternative="two-sided",
                role=(
                    "boundary"
                    if variable == "x" and endpoint == "early_minus_late"
                    else "primary"
                ),
            )


@_register("s6")
def build_s6(context: SourceBuildContext) -> None:
    metrics_root = context.source_root / "new_results_reanalysis" / "metrics"
    sequence = context.read_csv(metrics_root / "fig6_layer2_multi_sequence_metrics.csv")
    weights = context.read_csv(metrics_root / "fig6_layer2_multi_item_weights.csv")
    selected_k = [3, 5, 7, 10]
    sequence = sequence.loc[sequence["seq_len"].isin(selected_k)].copy()
    weights = weights.loc[weights["seq_len"].isin(selected_k)].copy()
    context.require_networks(sequence, label="S6 multi-item sequence metrics")
    context.require_networks(weights, label="S6 multi-item weights")

    sequence["similarity_n_eff"] = np.power(
        pd.to_numeric(sequence["seq_len"], errors="coerce"),
        pd.to_numeric(sequence["similarity_entropy"], errors="coerce"),
    )
    native_neff = sequence.groupby(["network_seed", "seq_len"], as_index=False)["n_eff"].mean().rename(
        columns={"n_eff": "value"}
    )
    native_neff["definition"] = "NNLS"
    similarity_neff = sequence.groupby(["network_seed", "seq_len"], as_index=False)["similarity_n_eff"].mean().rename(
        columns={"similarity_n_eff": "value"}
    )
    similarity_neff["definition"] = "Similarity"
    panel_a = pd.concat([native_neff, similarity_neff], ignore_index=True)
    context.write_panel("s6", "a", panel_a)
    context.add_summaries("s6", "a", panel_a, groups=["definition", "seq_len"])
    minimum_effective = panel_a.assign(margin=lambda frame: frame["value"] - 1.0).groupby(
        "network_seed"
    )["margin"].min()
    context.add_test(
        "s6",
        "a",
        "minimum_effective_components_above_one",
        minimum_effective,
        family="definition_holm2",
        alternative="greater",
    )

    positive_similarity = np.clip(
        pd.to_numeric(weights["constituent_similarity"], errors="coerce").to_numpy(dtype=float),
        0.0,
        None,
    )
    weights["positive_similarity"] = positive_similarity
    similarity_total = weights.groupby(["network_seed", "sequence_id", "seq_len"])[
        "positive_similarity"
    ].transform("sum")
    if not bool((similarity_total > 0).all()):
        raise ValueError("S6 similarity-based weights contain a zero denominator")
    weights["similarity_weight"] = weights["positive_similarity"] / similarity_total
    latest = weights.loc[weights["item_position"].eq(weights["seq_len"])].copy()
    native_latest = latest.groupby(["network_seed", "seq_len"], as_index=False)["item_weight"].mean().rename(
        columns={"item_weight": "value"}
    )
    native_latest["definition"] = "NNLS"
    similarity_latest = latest.groupby(["network_seed", "seq_len"], as_index=False)["similarity_weight"].mean().rename(
        columns={"similarity_weight": "value"}
    )
    similarity_latest["definition"] = "Similarity"
    panel_b = pd.concat([native_latest, similarity_latest], ignore_index=True)
    context.write_panel("s6", "b", panel_b)
    context.add_summaries("s6", "b", panel_b, groups=["definition", "seq_len"])
    maximum_latest = panel_b.groupby("network_seed")["value"].max()
    context.add_test(
        "s6",
        "b",
        "maximum_latest_weight_below_half",
        0.5 - maximum_latest,
        family="definition_holm2",
        alternative="greater",
    )

    root = "fig3_multiitem_peak_landscape/seed_*/data/intermediates/boundary_state_bank"
    map_rows: list[dict[str, float | int]] = []
    composition_rows: list[dict[str, float | int]] = []
    for meta_path in context.paths(f"{root}/sequence_meta.csv"):
        bank = meta_path.parent
        landscapes_path = context.track(bank / "landscapes.npz")
        states_path = context.track(bank / "state_bank_layer1.npz")
        meta = pd.read_csv(meta_path)
        seed = int(next(part for part in meta_path.parts if part.startswith("seed_")).split("_", 1)[1])
        with np.load(landscapes_path, allow_pickle=False) as landscapes, np.load(states_path, allow_pickle=False) as states:
            final_maps: dict[int, np.ndarray] = {}
            composites: dict[int, np.ndarray] = {}
            for row in meta.itertuples(index=False):
                sequence_id = int(row.sequence_id)
                seq_len = int(row.seq_len)
                delay_ms = int(row.delay_ms)
                baseline = _map28(landscapes[f"sequence_{sequence_id}_G_baseline"])
                final = _map28(landscapes[f"sequence_{sequence_id}_G_final"])
                delta = final - baseline
                map_rows.append(
                    {
                        "network_seed": seed,
                        "sequence_id": sequence_id,
                        "seq_len": seq_len,
                        "delay_ms": delay_ms,
                        "mean_delta_g": float(delta.mean()),
                        "moran_i": _moran_rook(delta),
                    }
                )
                s0 = _map28(states[f"sequence_{sequence_id}_S0_g"])
                s_final = _map28(states[f"sequence_{sequence_id}_S{seq_len}_g"])
                final_maps[sequence_id] = s_final - s0
                singleton_updates = [
                    _map28(states[f"sequence_{sequence_id}_singleton_reference_{position}_g"]) - s0
                    for position in range(1, seq_len + 1)
                ]
                composites[sequence_id] = np.mean(singleton_updates, axis=0)
            for (seq_len, delay_ms), group in meta.groupby(["seq_len", "delay_ms"], sort=True):
                sequence_ids = [int(value) for value in group["sequence_id"]]
                if len(sequence_ids) != 10:
                    raise ValueError(f"S6 expected 10 sequences per cell, got {len(sequence_ids)}")
                for sequence_id in sequence_ids:
                    matched = _centered_cosine(final_maps[sequence_id], composites[sequence_id])
                    deranged = float(
                        np.mean(
                            [
                                _centered_cosine(final_maps[sequence_id], composites[other])
                                for other in sequence_ids
                                if other != sequence_id
                            ]
                        )
                    )
                    composition_rows.append(
                        {
                            "network_seed": seed,
                            "sequence_id": sequence_id,
                            "seq_len": int(seq_len),
                            "delay_ms": int(delay_ms),
                            "matched_cosine": matched,
                            "deranged_cosine": deranged,
                            "value": matched - deranged,
                        }
                    )
    maps = pd.DataFrame(map_rows)
    map_network = maps.groupby(["network_seed", "seq_len", "delay_ms"], as_index=False)[
        ["mean_delta_g", "moran_i"]
    ].mean()
    context.require_networks(map_network, label="S6 coefficient-free maps")
    panel_c = map_network[["network_seed", "seq_len", "delay_ms", "mean_delta_g"]].rename(
        columns={"mean_delta_g": "value"}
    )
    moran_null = -1.0 / (28 * 28 - 1)
    panel_d = map_network.loc[
        map_network["seq_len"].eq(10),
        ["network_seed", "delay_ms", "moran_i"],
    ].copy()
    panel_d["value"] = panel_d["moran_i"] - moran_null
    context.write_panel("s6", "c", panel_c)
    context.write_panel("s6", "d", panel_d)
    context.add_summaries("s6", "c", panel_c, groups=["seq_len", "delay_ms"])
    context.add_summaries("s6", "d", panel_d, groups=["delay_ms"])
    c_pivot = panel_c.pivot(index="network_seed", columns=["seq_len", "delay_ms"], values="value")
    amplitude_did = (c_pivot[(10, 100)] - c_pivot[(10, 800)]) - (
        c_pivot[(3, 100)] - c_pivot[(3, 800)]
    )
    context.add_test("s6", "c", "load_by_delay_mean_delta_g", amplitude_did, family="morphology_holm3")
    context.add_test(
        "s6",
        "d",
        "k10_d800_moran_excess",
        panel_d.loc[panel_d["delay_ms"].eq(800), "value"],
        family="morphology_holm3",
    )

    composition = pd.DataFrame(composition_rows)
    cells = composition.groupby(["network_seed", "seq_len", "delay_ms"], as_index=False)[
        ["matched_cosine", "deranged_cosine", "value"]
    ].mean()
    min16 = cells.groupby("network_seed", as_index=False)["value"].min()
    if len(cells) != 320 or not bool((cells["value"] > 0).all()):
        raise ValueError("S6 matched-minus-deranged 16-cell grid is incomplete or non-positive")
    context.write_panel("s6", "e", cells)
    context.write_panel("s6", "f", min16)
    context.add_summaries("s6", "e", cells, groups=["seq_len", "delay_ms"])
    context.add_summaries("s6", "f", min16)
    context.add_test(
        "s6",
        "f",
        "minimum_matched_minus_deranged",
        min16["value"],
        family="morphology_holm3",
    )


@_register("s7")
def build_s7(context: SourceBuildContext) -> None:
    root = "fig6_peak_amplified_reentry/seed_*/data/metrics"
    removal = context.read_many(
        f"{root}/panel_f_high_stsp_overlap_ablation_summary.csv",
        usecols=[
            "network_seed",
            "sequence_id",
            "probe_id",
            "early_window_ms",
            "loss_condition",
            "loss_delta_spike_probability",
            "removed_active_area",
            "removed_input_energy",
        ],
    )
    removal = removal.loc[
        removal["early_window_ms"].eq(10)
        & removal["loss_condition"].isin(["high_stsp_overlap", "matched_removal"])
    ].copy()
    identity = ["network_seed", "sequence_id", "probe_id", "early_window_ms"]
    if removal.duplicated([*identity, "loss_condition"]).any():
        raise ValueError("S7 removal table contains duplicate paired-condition rows")
    loss = removal.pivot(index=identity, columns="loss_condition", values="loss_delta_spike_probability")
    area = removal.pivot(index=identity, columns="loss_condition", values="removed_active_area")
    energy = removal.pivot(index=identity, columns="loss_condition", values="removed_input_energy")
    required = {"high_stsp_overlap", "matched_removal"}
    for label, frame in (("loss", loss), ("area", area), ("energy", energy)):
        if not required.issubset(frame.columns):
            raise ValueError(f"S7 {label} pivot lacks the two removal conditions")
    paired = loss.join(area.add_prefix("area_"), how="inner").join(
        energy.add_prefix("energy_"), how="inner"
    ).reset_index()
    paired["exact_match"] = np.isclose(
        paired["area_high_stsp_overlap"],
        paired["area_matched_removal"],
        rtol=0.0,
        atol=0.0,
    ) & np.isclose(
        paired["energy_high_stsp_overlap"],
        paired["energy_matched_removal"],
        rtol=0.0,
        atol=0.0,
    )
    paired["difference_percent"] = 100.0 * (
        paired["high_stsp_overlap"] - paired["matched_removal"]
    )
    all_effect = paired.groupby("network_seed", as_index=False)["difference_percent"].mean().rename(
        columns={"difference_percent": "value"}
    )
    all_effect["subset"] = "All trials"
    exact_rows = paired.loc[paired["exact_match"]].copy()
    exact_effect = exact_rows.groupby("network_seed", as_index=False)["difference_percent"].mean().rename(
        columns={"difference_percent": "value"}
    )
    exact_effect["subset"] = "Exact match"
    panel_a = pd.concat([all_effect, exact_effect], ignore_index=True)
    context.require_networks(panel_a, label="S7 all versus exact removal effects")
    context.write_panel("s7", "a", panel_a)
    context.add_summaries("s7", "a", panel_a, groups=["subset"])
    for subset in ("All trials", "Exact match"):
        context.add_test(
            "s7",
            "a",
            f"{subset.lower().replace(' ', '_')}_removal_difference",
            panel_a.loc[panel_a["subset"].eq(subset), "value"],
            family="removal_holm2",
            alternative="greater",
        )

    panel_b = paired.groupby("network_seed", as_index=False).agg(
        n_trials=("exact_match", "size"),
        n_exact_trials=("exact_match", "sum"),
        value=("exact_match", lambda values: 100.0 * float(np.mean(values))),
    )
    context.require_networks(panel_b, label="S7 exact-match coverage")
    if panel_b["n_exact_trials"].le(0).any():
        raise ValueError("S7 exact-match removal subset is empty for at least one network")
    context.write_panel("s7", "b", panel_b)
    context.add_summaries("s7", "b", panel_b)

    windows = context.read_many(f"{root}/supp_s11d_overlap_interaction_window_robustness.csv")
    panel_c = windows.groupby(["network_seed", "early_window_ms"], as_index=False)["value"].mean().rename(
        columns={"early_window_ms": "window_ms"}
    )
    panel_c["value"] *= 100.0
    context.require_networks(panel_c, label="S7 window robustness")
    context.write_panel("s7", "c", panel_c)
    context.add_summaries("s7", "c", panel_c, groups=["window_ms"])
    for window in (5, 15, 20):
        context.add_test(
            "s7",
            "c",
            f"interaction_at_{window}ms",
            panel_c.loc[panel_c["window_ms"].eq(window), "value"],
            family="window_holm3",
        )

    threshold = context.read_many(
        f"{root}/supp_s11h_threshold_sensitivity.csv",
        usecols=["network_seed", "stsp_group_quantile", "overlap_threshold", "value"],
    )
    network_cells = threshold.groupby(
        ["network_seed", "stsp_group_quantile", "overlap_threshold"], as_index=False
    ).agg(value=("value", "mean"), n_valid=("value", "count"), n_rows=("value", "size"))
    network_cells["coverage"] = network_cells["n_valid"] / network_cells["n_rows"]
    panel_d = network_cells[["network_seed", "stsp_group_quantile", "overlap_threshold", "value"]].copy()
    panel_d["value"] *= 100.0
    panel_e = network_cells[["network_seed", "stsp_group_quantile", "overlap_threshold", "coverage", "n_valid", "n_rows"]].rename(
        columns={"coverage": "value"}
    )
    panel_e["value"] *= 100.0
    context.require_networks(panel_d, label="S7 threshold sensitivity")
    context.require_networks(panel_e, label="S7 coverage")
    context.write_panel("s7", "d", panel_d)
    context.write_panel("s7", "e", panel_e)
    context.add_summaries("s7", "d", panel_d, groups=["stsp_group_quantile", "overlap_threshold"])
    context.add_summaries(
        "s7",
        "e",
        panel_e,
        groups=["stsp_group_quantile", "overlap_threshold"],
        ci_method="none",
    )
    for (quantile, overlap), part in panel_d.groupby(["stsp_group_quantile", "overlap_threshold"], sort=True):
        context.add_test(
            "s7",
            "d",
            f"q{float(quantile):.2f}_overlap{float(overlap):.2f}",
            part["value"],
            family="definition_holm12",
        )

    availability = context.read_many(f"{root}/supp_s11e_overlap_site_availability.csv")
    context.write_panel("s7", "e", availability, suffix="_availability")

    score = context.read_many(
        f"{root}/supp_s11g_score_shuffle_null.csv",
        usecols=["network_seed", "endpoint", "observed_value", "null_value", "value"],
    )
    score = score.loc[score["endpoint"].eq("overlap_interaction")].dropna(
        subset=["observed_value", "null_value", "value"]
    )
    context.require_networks(score, label="S7 complete paired score shuffle rows")
    if not np.allclose(
        score["observed_value"] - score["null_value"],
        score["value"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("S7 score-shuffle rows do not satisfy observed - shuffled = difference")
    score_network = score.groupby("network_seed", as_index=False)[["observed_value", "null_value", "value"]].mean()
    panel_f = score_network.melt(
        id_vars="network_seed",
        value_vars=["observed_value", "null_value", "value"],
        var_name="endpoint",
        value_name="value_percent",
    )
    panel_f["endpoint"] = panel_f["endpoint"].map(
        {"observed_value": "Observed", "null_value": "Shuffled", "value": "Difference"}
    )
    panel_f = panel_f.rename(columns={"value_percent": "value"})
    panel_f["value"] *= 100.0
    context.require_networks(panel_f, label="S7 score shuffle")
    context.write_panel("s7", "f", panel_f)
    context.add_summaries("s7", "f", panel_f, groups=["endpoint"])
    context.add_test(
        "s7",
        "f",
        "observed_minus_spatial_shuffle",
        panel_f.loc[panel_f["endpoint"].eq("Difference"), "value"],
        family="score_shuffle",
    )


__all__ = ["FIGURE_BUILDERS"]
