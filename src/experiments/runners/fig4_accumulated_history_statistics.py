"""Rebuild the promoted Fig.4 accumulated-history statistics bundle.

Promoted on 2026-08-14 from
``.codex/tmp/fig4_candidate_statistics_20260731_KEEP/build_fig4_candidate_statistics.py``
(original SHA-256: ``302df6fc58490223374deef19318c9a1c035a91527523440caaa4059382b7ba9``).
The scientific analysis is unchanged. The default output is the stable result root, and a
hash-pinned fallback repairs the moved formal-Fig.2b parent path without changing bytes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED_SEEDS = tuple(range(1000, 1020))
STAGES = tuple(range(2, 11))
BOOTSTRAP_DRAWS = 20_000
RANDOM_SEED = 20_260_731
FORMAL_FIG2B_SHA256 = "6516038495c5a90a5b59733dc12fdbb41a2c25fa5a77d7abb35fb01566ffd7bf"


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / "src").is_dir():
        raise RuntimeError(f"Cannot resolve repository root from {__file__}")
    return root


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.paper_figures.new_results_reanalysis import (  # noqa: E402
    _bootstrap_mean_ci,
    _exact_sign_flip_p,
    _holm_adjust,
    _stable_seed,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _formal_fig2b_path() -> Path:
    candidates = (
        REPO_ROOT
        / "results/paper_figure_multi_seed/final_six_figures/fig2/data/panel_b_plot_data.csv",
        REPO_ROOT
        / (
            "results/paper_figure_multi_seed/"
            "final_six_figures_v5_fig3g_fig4_2x2_fixed_20260810/"
            "fig2/data/panel_b_plot_data.csv"
        ),
    )
    for candidate in candidates:
        if candidate.is_file() and _sha256(candidate) == FORMAL_FIG2B_SHA256:
            return candidate
    raise FileNotFoundError(
        "No hash-matched formal Fig.2b parent is available; expected SHA-256 "
        f"{FORMAL_FIG2B_SHA256} at one of: "
        + ", ".join(_relative(path) for path in candidates)
    )


def _seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.as_posix())
    if match is None:
        raise ValueError(f"Cannot infer network seed from {path}")
    return int(match.group(1))


def _require_seed_coverage(paths: Iterable[Path], label: str) -> list[Path]:
    ordered = sorted(paths, key=_seed_from_path)
    seeds = tuple(_seed_from_path(path) for path in ordered)
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"{label}: expected seeds {EXPECTED_SEEDS}, observed {seeds}")
    return ordered


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class CandidateAnalysis:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.data_dir = output_dir / "data"
        self.figure_dir = output_dir / "figures"
        self.log_dir = output_dir / "logs"
        self.metrics_dir = output_dir / "metrics"
        self.meta_dir = output_dir / "meta"
        for directory in (
            self.data_dir,
            self.figure_dir,
            self.log_dir,
            self.metrics_dir,
            self.meta_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.source_records: list[dict[str, Any]] = []
        self.descriptive_rows: list[dict[str, Any]] = []
        self.new_inference_rows: list[dict[str, Any]] = []
        self.logs: list[str] = []

    def load_csv(self, path: Path, *, bundle: str, seed: int | str) -> pd.DataFrame:
        frame = pd.read_csv(path)
        self.source_records.append(
            {
                "bundle": bundle,
                "network_seed": seed,
                "relative_path": _relative(path),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "sha256": _sha256(path),
            }
        )
        return frame

    def add_descriptive(
        self,
        *,
        panel: str,
        endpoint: str,
        condition: str,
        values: Iterable[float],
        unit: str,
        source: str,
    ) -> None:
        array = np.asarray(list(values), dtype=np.float64)
        array = array[np.isfinite(array)]
        if len(array) != len(EXPECTED_SEEDS):
            raise ValueError(
                f"{panel}/{endpoint}/{condition}: expected 20 network values, got {len(array)}"
            )
        low, high = _bootstrap_mean_ci(
            array,
            draws=BOOTSTRAP_DRAWS,
            seed=_stable_seed(RANDOM_SEED, panel, endpoint, condition),
        )
        self.descriptive_rows.append(
            {
                "panel": panel,
                "endpoint": endpoint,
                "condition": condition,
                "unit": unit,
                "n_networks": int(len(array)),
                "mean": float(array.mean()),
                "sd": float(array.std(ddof=1)),
                "sem": float(array.std(ddof=1) / math.sqrt(len(array))),
                "ci95_low": low,
                "ci95_high": high,
                "minimum": float(array.min()),
                "maximum": float(array.max()),
                "source_file": source,
            }
        )

    def add_inference(
        self,
        *,
        panel: str,
        claim_id: str,
        endpoint: str,
        values: Iterable[float],
        null: float,
        alternative: str,
        family: str,
        unit: str,
        source: str,
    ) -> None:
        array = np.asarray(list(values), dtype=np.float64)
        array = array[np.isfinite(array)]
        if len(array) != len(EXPECTED_SEEDS):
            raise ValueError(f"{panel}/{endpoint}: expected 20 network values, got {len(array)}")
        low, high = _bootstrap_mean_ci(
            array,
            draws=BOOTSTRAP_DRAWS,
            seed=_stable_seed(RANDOM_SEED, panel, endpoint, family),
        )
        centered = array - float(null)
        self.new_inference_rows.append(
            {
                "panel": panel,
                "claim_id": claim_id,
                "endpoint": endpoint,
                "unit": unit,
                "n_networks": int(len(array)),
                "mean": float(array.mean()),
                "sd": float(array.std(ddof=1)),
                "sem": float(array.std(ddof=1) / math.sqrt(len(array))),
                "ci95_low": low,
                "ci95_high": high,
                "null_value": float(null),
                "effect_vs_null": float(centered.mean()),
                "alternative": alternative,
                "n_above_null": int(np.sum(centered > 0)),
                "n_below_null": int(np.sum(centered < 0)),
                "p_value": _exact_sign_flip_p(centered, alternative=alternative),
                "p_holm_family": float("nan"),
                "p_holm_all_new": float("nan"),
                "correction_family": family,
                "inference_origin": "candidate_secondary_reanalysis",
                "method": (
                    "20,000-draw percentile bootstrap CI of network mean; "
                    "exact sign-flip test over independently trained network values"
                ),
                "source_file": source,
            }
        )


def _analyze_progressive(ctx: CandidateAnalysis) -> tuple[pd.DataFrame, dict[str, Any]]:
    pattern = (
        REPO_ROOT
        / "results/paper_figure_multi_seed/fig3_multiitem_peak_landscape"
    )
    paths = _require_seed_coverage(
        pattern.glob("seed_*/data/metrics/panel_b_progressive_update_metrics.csv"),
        "progressive-update",
    )
    frames: list[pd.DataFrame] = []
    required = {
        "network_seed",
        "sequence_id",
        "condition_id",
        "stage_k",
        "layer",
        "state_variable",
        "state_displacement",
        "natural_decay_displacement",
        "observed_minus_natural_decay",
    }
    for path in paths:
        seed = _seed_from_path(path)
        frame = ctx.load_csv(path, bundle="progressive_update", seed=seed)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    focus = data.loc[
        data["condition_id"].eq("K10_D200")
        & data["layer"].eq("layer2")
        & data["state_variable"].isin(["u", "x"])
        & data["stage_k"].between(2, 10)
    ].copy()
    observed_identity_error = np.abs(
        focus["state_displacement"]
        - focus["natural_decay_displacement"]
        - focus["observed_minus_natural_decay"]
    )
    if float(observed_identity_error.max()) > 1e-10:
        raise ValueError("Progressive observed-minus-passive identity failed")
    by_variable = (
        focus.groupby(
            ["network_seed", "stage_k", "state_variable"], as_index=False, sort=True
        )
        .agg(
            observed_displacement=("state_displacement", "mean"),
            passive_displacement=("natural_decay_displacement", "mean"),
            observed_minus_passive=("observed_minus_natural_decay", "mean"),
            n_sequences=("sequence_id", "nunique"),
        )
    )
    expected_rows = len(EXPECTED_SEEDS) * len(STAGES) * 2
    if len(by_variable) != expected_rows:
        raise ValueError(f"Progressive variable table: expected {expected_rows}, got {len(by_variable)}")
    if not by_variable["n_sequences"].eq(10).all():
        raise ValueError("Progressive table does not contain 10 sequences per network-stage-variable")
    joint = (
        by_variable.groupby(["network_seed", "stage_k"], as_index=False, sort=True)
        .agg(
            observed_displacement=("observed_displacement", "mean"),
            passive_displacement=("passive_displacement", "mean"),
            observed_minus_passive=("observed_minus_passive", "mean"),
            n_sequences=("n_sequences", "min"),
        )
    )
    piv = by_variable.pivot(
        index=["network_seed", "stage_k"],
        columns="state_variable",
        values="observed_minus_passive",
    ).reset_index()
    piv = piv.rename(columns={"u": "u_observed_minus_passive", "x": "x_observed_minus_passive"})
    joint = joint.merge(piv, on=["network_seed", "stage_k"], validate="one_to_one")
    if len(joint) != len(EXPECTED_SEEDS) * len(STAGES):
        raise ValueError("Progressive joint table has incomplete network-stage coverage")
    joint.to_csv(
        ctx.data_dir / "fig4a_progressive_network_stage.csv", index=False, encoding="utf-8"
    )

    source_name = "data/fig4a_progressive_network_stage.csv"
    for stage, part in joint.groupby("stage_k", sort=True):
        for endpoint, column in (
            ("joint_state_displacement", "observed_displacement"),
            ("joint_passive_displacement", "passive_displacement"),
            ("joint_observed_minus_passive", "observed_minus_passive"),
        ):
            ctx.add_descriptive(
                panel="a",
                endpoint=endpoint,
                condition=f"stage_{int(stage)}",
                values=part[column],
                unit="cosine_distance",
                source=source_name,
            )
    network_mean = joint.groupby("network_seed")["observed_minus_passive"].mean()
    network_min = joint.groupby("network_seed")["observed_minus_passive"].min()
    ctx.add_inference(
        panel="a",
        claim_id="recurrence_across_stages",
        endpoint="mean_joint_observed_minus_passive_k2_k10",
        values=network_mean,
        null=0.0,
        alternative="greater",
        family="fig4a_recurrence_11",
        unit="cosine_distance",
        source=source_name,
    )
    ctx.add_inference(
        panel="a",
        claim_id="recurrence_across_stages",
        endpoint="minimum_joint_observed_minus_passive_k2_k10",
        values=network_min,
        null=0.0,
        alternative="greater",
        family="fig4a_recurrence_11",
        unit="cosine_distance",
        source=source_name,
    )
    for stage, part in joint.groupby("stage_k", sort=True):
        ctx.add_inference(
            panel="a",
            claim_id="recurrence_at_each_stage",
            endpoint=f"joint_observed_minus_passive_stage_{int(stage)}",
            values=part["observed_minus_passive"],
            null=0.0,
            alternative="greater",
            family="fig4a_recurrence_11",
            unit="cosine_distance",
            source=source_name,
        )
    audit = {
        "n_networks": int(joint["network_seed"].nunique()),
        "n_stages": int(joint["stage_k"].nunique()),
        "n_network_stage_rows": int(len(joint)),
        "n_positive_network_stage_rows": int(joint["observed_minus_passive"].gt(0).sum()),
        "minimum_network_stage_value": float(joint["observed_minus_passive"].min()),
        "maximum_network_stage_value": float(joint["observed_minus_passive"].max()),
        "max_identity_error": float(observed_identity_error.max()),
    }
    ctx.logs.append(f"progressive audit {audit}")
    return joint, audit


def _last_label(value: Any) -> int:
    labels = ast.literal_eval(str(value))
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Invalid sequence_labels: {value}")
    return int(labels[-1])


def _analyze_behavior(
    ctx: CandidateAnalysis,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = REPO_ROOT / "results/multi_seed_rollout/fig2/fixed_b_mechanism_confirmatory"
    rollout_paths = _require_seed_coverage(
        base.glob("seed_*/data/intermediates/fixed_b_rollout_bank/rollout_rows.csv"),
        "fixed-B rollout bank",
    )
    history_paths = _require_seed_coverage(
        base.glob("seed_*/data/intermediates/fixed_b_history_bank/history_specs.csv"),
        "fixed-B history bank",
    )
    rollout_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    for path in rollout_paths:
        seed = _seed_from_path(path)
        rollout_frames.append(ctx.load_csv(path, bundle="fixed_b_rollout_bank", seed=seed))
    for path in history_paths:
        seed = _seed_from_path(path)
        frame = ctx.load_csv(path, bundle="fixed_b_history_bank", seed=seed)
        frame["network_seed"] = seed
        history_frames.append(frame)
    rollout = pd.concat(rollout_frames, ignore_index=True, sort=False)
    histories = pd.concat(history_frames, ignore_index=True, sort=False)
    rows = rollout.loc[
        rollout["track"].eq("stsp_isolated")
        & rollout["branch"].eq("free")
        & rollout["prefix_k"].isin([1, 5])
    ].copy()
    anchor_key = ["network_seed", "b_anchor_id"]
    s0_rows = rows.loc[rows["history_condition"].eq("S0")].copy()
    for column in ("prediction", "B_label", "exact_b_tensor_sha256"):
        within = s0_rows.groupby(anchor_key + ["prefix_k"])[column].nunique(dropna=False)
        if not within.eq(1).all():
            raise ValueError(f"Repeated S0 {column} is inconsistent within K")
    s0_unique = (
        s0_rows.groupby(anchor_key + ["prefix_k"], as_index=False)
        .agg(
            prediction=("prediction", "first"),
            B_label=("B_label", "first"),
            exact_b_tensor_sha256=("exact_b_tensor_sha256", "first"),
        )
    )
    identity_mismatch: dict[str, int] = {}
    for column in ("prediction", "B_label", "exact_b_tensor_sha256"):
        pivot = s0_unique.pivot(index=anchor_key, columns="prefix_k", values=column)
        if tuple(pivot.columns) != (1, 5):
            raise ValueError(f"S0 {column} does not cover both K1 and K5")
        identity_mismatch[column] = int(pivot[1].ne(pivot[5]).sum())
    if any(identity_mismatch.values()):
        raise ValueError(f"K1/K5 S0 identity mismatch: {identity_mismatch}")
    if len(s0_unique) != len(EXPECTED_SEEDS) * 50 * 2:
        raise ValueError(f"Expected 2,000 K-specific S0 anchor rows, got {len(s0_unique)}")

    records: list[dict[str, Any]] = []
    for prefix_k in (1, 5):
        subset = rows.loc[rows["prefix_k"].eq(prefix_k)].copy()
        s0_anchor = (
            subset.loc[subset["history_condition"].eq("S0")]
            .groupby(anchor_key, as_index=False)
            .agg(S0_prediction=("prediction", "first"), B_label=("B_label", "first"))
        )
        history_rows = histories.loc[
            histories["prefix_k"].eq(prefix_k)
            & histories["history_condition"].isin(["A", "C"])
        ].copy()
        history_rows["history_label"] = history_rows["sequence_labels"].map(_last_label)
        history_key = [
            "network_seed",
            "history_row_id",
            "history_family_id",
            "history_condition",
        ]
        work = subset.loc[subset["history_condition"].isin(["A", "C"])].merge(
            history_rows[history_key + ["history_label"]],
            on=history_key,
            how="left",
            validate="many_to_one",
        )
        work = work.merge(
            s0_anchor,
            on=anchor_key,
            how="left",
            validate="many_to_one",
            suffixes=("", "_s0"),
        )
        if work[["history_label", "S0_prediction", "B_label_s0"]].isna().any().any():
            raise ValueError(f"K{prefix_k}: behavior merge produced missing values")
        if not work["B_label"].eq(work["B_label_s0"]).all():
            raise ValueError(f"K{prefix_k}: B label changed during behavior merge")
        work["history_relation"] = np.where(
            work["history_label"].astype(int).eq(work["B_label"].astype(int)),
            "aligned",
            "mismatched",
        )
        work["S0_correct"] = work["S0_prediction"].eq(work["B_label"])
        work["history_correct"] = work["prediction"].eq(work["B_label"])
        for outcome in ("rescue", "loss"):
            eligible = ~work["S0_correct"] if outcome == "rescue" else work["S0_correct"]
            eligible_rows = work.loc[eligible].copy()
            eligible_rows["event"] = (
                eligible_rows["history_correct"]
                if outcome == "rescue"
                else ~eligible_rows["history_correct"]
            ).astype(float)
            grouped = (
                eligible_rows.groupby(
                    ["network_seed", "history_relation"], as_index=False, sort=True
                )
                .agg(
                    rate_percent=("event", lambda value: float(value.mean() * 100.0)),
                    eligible_anchors=("b_anchor_id", "nunique"),
                    history_rows=("event", "size"),
                )
            )
            grouped["prefix_k"] = prefix_k
            grouped["outcome_type"] = outcome
            records.extend(grouped.to_dict("records"))
    rates = pd.DataFrame(records).sort_values(
        ["network_seed", "prefix_k", "outcome_type", "history_relation"],
        kind="stable",
    )
    if len(rates) != len(EXPECTED_SEEDS) * 2 * 2 * 2:
        raise ValueError(f"Expected 160 behavior-rate rows, got {len(rates)}")
    rates.to_csv(ctx.data_dir / "fig4b_behavior_network_rates.csv", index=False, encoding="utf-8")
    source_name = "data/fig4b_behavior_network_rates.csv"
    formal_fig2b_path = _formal_fig2b_path()
    formal_fig2b = ctx.load_csv(
        formal_fig2b_path, bundle="formal_fig2b_plot_data", seed="aggregate"
    )
    k1_candidate = rates.loc[rates["prefix_k"].eq(1)].copy()
    formal_comparison = formal_fig2b.merge(
        k1_candidate,
        on=["network_seed", "outcome_type", "history_relation"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_formal", "_candidate"),
    )
    if len(formal_comparison) != 80 or not formal_comparison["_merge"].eq("both").all():
        raise ValueError("Candidate K1 behavior rows do not match the formal Fig.2b row set")
    formal_rate_max_abs_difference = float(
        np.abs(
            formal_comparison["value"] - formal_comparison["rate_percent"]
        ).max()
    )
    formal_eligible_mismatches = int(
        formal_comparison["eligible_anchors_formal"]
        .ne(formal_comparison["eligible_anchors_candidate"])
        .sum()
    )
    formal_history_row_mismatches = int(
        formal_comparison["history_rows_formal"]
        .ne(formal_comparison["history_rows_candidate"])
        .sum()
    )
    if (
        formal_rate_max_abs_difference > 1e-12
        or formal_eligible_mismatches
        or formal_history_row_mismatches
    ):
        raise ValueError("Candidate K1 behavior values differ from formal Fig.2b")
    for keys, part in rates.groupby(
        ["prefix_k", "outcome_type", "history_relation"], sort=True
    ):
        prefix_k, outcome, relation = keys
        ctx.add_descriptive(
            panel="b",
            endpoint=f"{outcome}_rate",
            condition=f"K{int(prefix_k)}_{relation}",
            values=part["rate_percent"],
            unit="percent",
            source=source_name,
        )

    pivot = rates.pivot(
        index=["network_seed", "outcome_type"],
        columns=["prefix_k", "history_relation"],
        values="rate_percent",
    )
    contrast_records: list[dict[str, Any]] = []
    for (network_seed, outcome), row in pivot.iterrows():
        k1_contrast = float(row[(1, "aligned")] - row[(1, "mismatched")])
        k5_contrast = float(row[(5, "aligned")] - row[(5, "mismatched")])
        k1_balanced = float(0.5 * (row[(1, "aligned")] + row[(1, "mismatched")]))
        k5_balanced = float(0.5 * (row[(5, "aligned")] + row[(5, "mismatched")]))
        contrast_records.append(
            {
                "network_seed": int(network_seed),
                "outcome_type": str(outcome),
                "K1_aligned_minus_mismatched": k1_contrast,
                "K5_aligned_minus_mismatched": k5_contrast,
                "depth_by_relation_interaction": k5_contrast - k1_contrast,
                "K1_relation_balanced_rate": k1_balanced,
                "K5_relation_balanced_rate": k5_balanced,
                "relation_balanced_K5_minus_K1": k5_balanced - k1_balanced,
            }
        )
    contrasts = pd.DataFrame(contrast_records).sort_values(
        ["network_seed", "outcome_type"], kind="stable"
    )
    contrasts.to_csv(
        ctx.data_dir / "fig4b_behavior_network_contrasts.csv", index=False, encoding="utf-8"
    )
    contrast_source = "data/fig4b_behavior_network_contrasts.csv"
    balanced_records: list[dict[str, Any]] = []
    for row in contrasts.itertuples(index=False):
        balanced_records.extend(
            [
                {
                    "network_seed": int(row.network_seed),
                    "outcome_type": str(row.outcome_type),
                    "prefix_k": 1,
                    "relation_balanced_rate_percent": float(row.K1_relation_balanced_rate),
                },
                {
                    "network_seed": int(row.network_seed),
                    "outcome_type": str(row.outcome_type),
                    "prefix_k": 5,
                    "relation_balanced_rate_percent": float(row.K5_relation_balanced_rate),
                },
            ]
        )
    balanced = pd.DataFrame(balanced_records).sort_values(
        ["network_seed", "outcome_type", "prefix_k"], kind="stable"
    )
    balanced.to_csv(
        ctx.data_dir / "fig4b_behavior_depth_network_rates.csv",
        index=False,
        encoding="utf-8",
    )
    balanced_source = "data/fig4b_behavior_depth_network_rates.csv"
    for keys, part in balanced.groupby(["outcome_type", "prefix_k"], sort=True):
        outcome, prefix_k = keys
        ctx.add_descriptive(
            panel="b",
            endpoint=f"{outcome}_relation_balanced_rate",
            condition=f"K{int(prefix_k)}",
            values=part["relation_balanced_rate_percent"],
            unit="percent",
            source=balanced_source,
        )
    for outcome, part in contrasts.groupby("outcome_type", sort=True):
        k1_alt = "greater" if outcome == "rescue" else "less"
        ctx.add_inference(
            panel="b",
            claim_id="one_step_alignment_signature",
            endpoint=f"K1_{outcome}_aligned_minus_mismatched",
            values=part["K1_aligned_minus_mismatched"],
            null=0.0,
            alternative=k1_alt,
            family="fig4b_k1_signature_2",
            unit="percent",
            source=contrast_source,
        )
        ctx.add_inference(
            panel="b",
            claim_id="deep_history_alignment_contrast",
            endpoint=f"K5_{outcome}_aligned_minus_mismatched",
            values=part["K5_aligned_minus_mismatched"],
            null=0.0,
            alternative="two-sided",
            family="fig4b_k5_contrasts_2",
            unit="percent",
            source=contrast_source,
        )
        ctx.add_inference(
            panel="b",
            claim_id="depth_changes_alignment_signature",
            endpoint=f"{outcome}_depth_by_relation_interaction_K5_minus_K1",
            values=part["depth_by_relation_interaction"],
            null=0.0,
            alternative="two-sided",
            family="fig4b_depth_interaction_2",
            unit="percent",
            source=contrast_source,
        )
        ctx.add_inference(
            panel="b",
            claim_id="deep_history_changes_behavioral_outcomes",
            endpoint=f"{outcome}_relation_balanced_K5_minus_K1",
            values=part["relation_balanced_K5_minus_K1"],
            null=0.0,
            alternative="two-sided",
            family="fig4b_balanced_depth_shift_2",
            unit="percent",
            source=contrast_source,
        )

    eligible = rates.pivot_table(
        index=["network_seed", "outcome_type", "history_relation"],
        columns="prefix_k",
        values="eligible_anchors",
        aggfunc="first",
    )
    eligible_mismatch_count = int(eligible[1].ne(eligible[5]).sum())
    if eligible_mismatch_count:
        raise ValueError("K1/K5 eligible-anchor counts differ")
    audit = {
        "n_networks": int(rates["network_seed"].nunique()),
        "n_k_specific_s0_anchors": int(len(s0_unique)),
        "n_unique_network_anchor_pairs": int(len(s0_unique) // 2),
        "s0_identity_mismatch_counts": identity_mismatch,
        "eligible_anchor_mismatch_rows": eligible_mismatch_count,
        "minimum_rescue_eligible_anchors": int(
            rates.loc[rates["outcome_type"].eq("rescue"), "eligible_anchors"].min()
        ),
        "minimum_loss_eligible_anchors": int(
            rates.loc[rates["outcome_type"].eq("loss"), "eligible_anchors"].min()
        ),
        "formal_fig2b_k1_rows_compared": int(len(formal_comparison)),
        "formal_fig2b_rate_max_abs_difference": formal_rate_max_abs_difference,
        "formal_fig2b_eligible_anchor_mismatches": formal_eligible_mismatches,
        "formal_fig2b_history_row_mismatches": formal_history_row_mismatches,
    }
    ctx.logs.append(f"behavior audit {audit}")
    return rates, contrasts, balanced, audit


def _analyze_event(ctx: CandidateAnalysis) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = REPO_ROOT / "results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory"
    paths = _require_seed_coverage(
        base.glob("seed_*/data/metrics/fixed_b_event_gamma_cell_metrics.csv"),
        "fixed-B event residual",
    )
    frames: list[pd.DataFrame] = []
    for path in paths:
        seed = _seed_from_path(path)
        frame = ctx.load_csv(path, bundle="fixed_b_event_residual", seed=seed)
        if not frame["network_seed"].eq(seed).all():
            raise ValueError(f"{path}: embedded network_seed does not match path")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    valid = pd.to_numeric(data["valid"], errors="raise").eq(1)
    focus = data.loc[data["prefix_k"].eq(5) & valid].copy()
    network = (
        focus.groupby("network_seed", as_index=False, sort=True)
        .agg(
            changed_events=("changed_coordinate_gamma_mean_abs", "mean"),
            matched_random=("matched_random_gamma_mean_abs", "mean"),
            n_valid_rows=("valid", "size"),
        )
    )
    network["changed_minus_random"] = network["changed_events"] - network["matched_random"]
    if tuple(network["network_seed"].astype(int)) != EXPECTED_SEEDS:
        raise ValueError("Event table does not cover all 20 networks")
    if not network["n_valid_rows"].eq(500).all():
        raise ValueError("Event table does not contain 500 valid rows per network at K5")
    network.to_csv(
        ctx.data_dir / "fig4d_k5_event_network_metrics.csv", index=False, encoding="utf-8"
    )
    source_name = "data/fig4d_k5_event_network_metrics.csv"
    for condition, column in (
        ("changed_events", "changed_events"),
        ("matched_random", "matched_random"),
        ("changed_minus_random", "changed_minus_random"),
    ):
        ctx.add_descriptive(
            panel="d",
            endpoint="K5_residual_magnitude",
            condition=condition,
            values=network[column],
            unit="mean_absolute_residual",
            source=source_name,
        )
    ctx.add_inference(
        panel="d",
        claim_id="deep_history_residual_tracks_changed_events",
        endpoint="K5_changed_events_minus_matched_random",
        values=network["changed_minus_random"],
        null=0.0,
        alternative="greater",
        family="fig4d_event_1",
        unit="mean_absolute_residual",
        source=source_name,
    )
    audit = {
        "n_networks": int(len(network)),
        "valid_rows_per_network_min": int(network["n_valid_rows"].min()),
        "valid_rows_per_network_max": int(network["n_valid_rows"].max()),
        "n_networks_changed_above_random": int(network["changed_minus_random"].gt(0).sum()),
    }
    ctx.logs.append(f"event audit {audit}")
    return network, audit


def _load_authoritative_fixed_b(
    ctx: CandidateAnalysis,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    aggregate_dir = (
        REPO_ROOT
        / "results/paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/aggregate"
    )
    scalar_path = aggregate_dir / "fixed_b_confirmatory_network_scalars.csv"
    inference_path = aggregate_dir / "fixed_b_confirmatory_inference.csv"
    scalars = ctx.load_csv(scalar_path, bundle="fixed_b_confirmatory_scalars", seed="aggregate")
    supplied = ctx.load_csv(
        inference_path, bundle="fixed_b_confirmatory_inference", seed="aggregate"
    )
    endpoints = [
        "same_B_common_update_cosine",
        "processing_residual_gamma_energy_fraction",
        "full_trace_event_gamma_enrichment",
        "layer1_only_layer2_update_donor_transfer",
        "layer1_only_early_class_score_donor_transfer",
    ]
    k5 = scalars.loc[
        scalars["prefix_k"].eq(5) & scalars["endpoint"].isin(endpoints)
    ].copy()
    counts = k5.groupby("endpoint")["network_seed"].nunique()
    if not counts.reindex(endpoints).eq(20).all():
        raise ValueError(f"K5 fixed-B scalar coverage failed: {counts.to_dict()}")
    k5.to_csv(
        ctx.data_dir / "fig4c_e_k5_fixed_b_network_scalars.csv", index=False, encoding="utf-8"
    )
    source_name = "data/fig4c_e_k5_fixed_b_network_scalars.csv"
    for endpoint, part in k5.groupby("endpoint", sort=True):
        panel = "c" if endpoint in {
            "same_B_common_update_cosine",
            "processing_residual_gamma_energy_fraction",
        } else ("d" if endpoint == "full_trace_event_gamma_enrichment" else "e")
        ctx.add_descriptive(
            panel=panel,
            endpoint=endpoint,
            condition="K5",
            values=part["value"],
            unit="index",
            source=source_name,
        )
    authoritative = supplied.loc[
        supplied["prefix_k"].eq(5) & supplied["endpoint"].isin(endpoints)
    ].copy()
    if len(authoritative) != len(endpoints):
        raise ValueError("Supplied K5 fixed-B inference is incomplete")
    merged_mean = k5.groupby("endpoint", as_index=False)["value"].mean().merge(
        authoritative[["endpoint", "mean"]], on="endpoint", validate="one_to_one"
    )
    if float(np.abs(merged_mean["value"] - merged_mean["mean"]).max()) > 1e-12:
        raise ValueError("Supplied K5 inference means do not match network scalars")
    panel_map = {
        "same_B_common_update_cosine": "c",
        "processing_residual_gamma_energy_fraction": "c",
        "full_trace_event_gamma_enrichment": "d",
        "layer1_only_layer2_update_donor_transfer": "e",
        "layer1_only_early_class_score_donor_transfer": "e",
    }
    rows: list[dict[str, Any]] = []
    for row in authoritative.itertuples(index=False):
        adjusted = pd.to_numeric(pd.Series([row.holm_adjusted_p]), errors="coerce").iloc[0]
        rows.append(
            {
                "panel": panel_map[str(row.endpoint)],
                "claim_id": "deep_history_fixed_b_mechanism_persists",
                "endpoint": str(row.endpoint),
                "unit": "index",
                "n_networks": int(row.n_networks),
                "mean": float(row.mean),
                "sd": float("nan"),
                "sem": float("nan"),
                "ci95_low": float(row.ci95_low),
                "ci95_high": float(row.ci95_high),
                "null_value": float(row.threshold),
                "effect_vs_null": float(row.mean) - float(row.threshold),
                "alternative": "greater",
                "n_above_null": int(round(float(row.fraction_meeting_threshold) * int(row.n_networks))),
                "n_below_null": int(
                    int(row.n_networks)
                    - round(float(row.fraction_meeting_threshold) * int(row.n_networks))
                ),
                "p_value": float(row.p_one_sided),
                "p_holm_family": float(adjusted) if np.isfinite(adjusted) else float("nan"),
                "p_holm_all_new": float("nan"),
                "correction_family": str(row.family),
                "inference_origin": "supplied_confirmatory_inference",
                "method": (
                    "supplied fixed-B confirmatory bootstrap CI and exact network-level "
                    "sign-flip inference; original multiplicity retained"
                ),
                "source_file": _relative(inference_path),
            }
        )
    authoritative_table = pd.DataFrame(rows)
    audit = {
        "n_networks": int(k5["network_seed"].nunique()),
        "n_endpoints": int(k5["endpoint"].nunique()),
        "all_network_values_meet_endpoint_threshold": bool(
            (k5["value"] >= k5["threshold"]).all()
        ),
        "supplied_mean_max_abs_difference": float(
            np.abs(merged_mean["value"] - merged_mean["mean"]).max()
        ),
    }
    ctx.logs.append(f"fixed-B authoritative audit {audit}")
    return k5, authoritative_table, audit


def _finalize_inference(ctx: CandidateAnalysis) -> pd.DataFrame:
    new = pd.DataFrame(ctx.new_inference_rows)
    for _, indices in new.groupby("correction_family", sort=True).groups.items():
        idx = list(indices)
        new.loc[idx, "p_holm_family"] = _holm_adjust(
            new.loc[idx, "p_value"].to_numpy(dtype=np.float64)
        )
    new["p_holm_all_new"] = _holm_adjust(new["p_value"].to_numpy(dtype=np.float64))
    return new.sort_values(["panel", "correction_family", "endpoint"], kind="stable")


def _find(table: pd.DataFrame, endpoint: str) -> pd.Series:
    rows = table.loc[table["endpoint"].eq(endpoint)]
    if len(rows) != 1:
        raise ValueError(f"Expected one inference row for {endpoint}, got {len(rows)}")
    return rows.iloc[0]


def _build_verdict(
    *,
    new: pd.DataFrame,
    authoritative: pd.DataFrame,
    progressive_audit: dict[str, Any],
    behavior_audit: dict[str, Any],
    event_audit: dict[str, Any],
) -> dict[str, Any]:
    recurrence_rows = new.loc[new["correction_family"].eq("fig4a_recurrence_11")]
    recurrence_pass = bool(
        len(recurrence_rows) == 11
        and recurrence_rows["effect_vs_null"].gt(0).all()
        and recurrence_rows["p_holm_all_new"].lt(0.05).all()
        and progressive_audit["n_positive_network_stage_rows"]
        == progressive_audit["n_network_stage_rows"]
    )
    rescue_depth = _find(new, "rescue_relation_balanced_K5_minus_K1")
    loss_depth = _find(new, "loss_relation_balanced_K5_minus_K1")
    behavior_pass = bool(
        rescue_depth["mean"] < 0
        and rescue_depth["p_holm_all_new"] < 0.05
        and loss_depth["mean"] > 0
        and loss_depth["p_holm_all_new"] < 0.05
        and not any(behavior_audit["s0_identity_mismatch_counts"].values())
    )
    common = _find(authoritative, "same_B_common_update_cosine")
    residual = _find(authoritative, "processing_residual_gamma_energy_fraction")
    state_component_pass = bool(
        common["effect_vs_null"] > 0
        and common["p_value"] < 0.05
        and common["n_above_null"] == 20
        and residual["effect_vs_null"] > 0
        and residual["p_holm_family"] < 0.05
        and residual["n_above_null"] == 20
    )
    event = _find(new, "K5_changed_events_minus_matched_random")
    event_pass = bool(
        event["mean"] > 0
        and event["p_holm_all_new"] < 0.05
        and event_audit["n_networks_changed_above_random"] == 20
    )
    l2_donor = _find(authoritative, "layer1_only_layer2_update_donor_transfer")
    early_donor = _find(
        authoritative, "layer1_only_early_class_score_donor_transfer"
    )
    donor_pass = bool(
        l2_donor["mean"] > 0
        and l2_donor["p_holm_family"] < 0.05
        and l2_donor["n_above_null"] == 20
        and early_donor["mean"] > 0
        and early_donor["p_holm_family"] < 0.05
        and early_donor["n_above_null"] == 20
    )
    k5_rescue = _find(new, "K5_rescue_aligned_minus_mismatched")
    k5_loss = _find(new, "K5_loss_aligned_minus_mismatched")
    gates = {
        "stagewise_recurrence_beyond_passive": recurrence_pass,
        "deep_history_shifts_opportunity_conditioned_outcomes": behavior_pass,
        "k5_common_and_history_components_persist": state_component_pass,
        "k5_residual_tracks_changed_events": event_pass,
        "k5_downstream_donor_transfer_persists": donor_pass,
    }
    return {
        "overall_candidate_conclusion_supported": bool(all(gates.values())),
        "gates": gates,
        "supported_claim": (
            "Successive inputs repeatedly displace the inherited STSP state beyond "
            "equal-time passive evolution. Relative to K1, accumulated K5 history "
            "reduces relation-balanced rescue and increases relation-balanced loss, while "
            "the same-B common component, history residual, changed-event association, "
            "and L1-to-downstream donor transfer remain detectable across 20 networks."
        ),
        "unsupported_or_overstated_claims": [
            (
                "Do not claim a statistically confirmed K5 rescue reversal: its two-sided "
                f"Holm-adjusted p is {float(k5_rescue['p_holm_all_new']):.6g}."
            ),
            (
                "Do not claim equivalence or exact disappearance of the K5 loss alignment "
                f"contrast: its two-sided Holm-adjusted p is {float(k5_loss['p_holm_all_new']):.6g}, "
                "and no equivalence margin was prespecified."
            ),
            "Do not infer full-sequence organization or long-term memory from these endpoints.",
            "Do not claim stagewise behavioral or donor-swap replication in the progressive protocol.",
        ],
    }


def _write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    verdict = summary["verdict"]
    lines = [
        "# Fig.4 candidate statistics (KEEP until panel decision)",
        "",
        "This is a plot-only/network-level candidate reanalysis. It does not rerun training,",
        "simulation, or forward replay, and it does not modify formal Fig.2/Fig.4 artifacts.",
        "",
        "## Verdict",
        "",
        f"Overall supported: `{verdict['overall_candidate_conclusion_supported']}`",
        "",
        verdict["supported_claim"],
        "",
        "## Candidate Fig.4b display",
        "",
        "Display only relation-balanced K1 versus K5 rescue and loss rates. Rescue and loss",
        "retain separate opportunity denominators. Do not display the two K5 aligned-minus-",
        "mismatched contrasts; keep the significant depth-by-relation interactions in the",
        "statistics/source-data record only.",
        "",
        "## Statistical policy",
        "",
        "- Independent unit: trained network (`n = 20`, seeds 1000-1019).",
        "- Candidate-derived CIs: 20,000-draw percentile bootstrap of network means.",
        "- Candidate-derived tests: exact sign-flip over network values.",
        "- Holm adjustment is reported within each prespecified family and across all new tests.",
        "- Fixed-B K5 mechanism endpoints retain their supplied confirmatory inference.",
        "",
        "## Boundaries",
        "",
    ]
    lines.extend(f"- {item}" for item in verdict["unsupported_or_overstated_claims"])
    lines.extend(
        [
            "",
            "The directory name contains `KEEP` because these candidate statistics should remain",
            "available until the user either promotes them into the formal Fig.4 bundle or rejects",
            "the candidate design.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path) -> dict[str, Any]:
    ctx = CandidateAnalysis(output_dir)
    progressive, progressive_audit = _analyze_progressive(ctx)
    rates, behavior_contrasts, behavior_balanced, behavior_audit = _analyze_behavior(ctx)
    event, event_audit = _analyze_event(ctx)
    k5_scalars, authoritative, fixed_b_audit = _load_authoritative_fixed_b(ctx)
    new_inference = _finalize_inference(ctx)
    descriptive = pd.DataFrame(ctx.descriptive_rows).sort_values(
        ["panel", "endpoint", "condition"], kind="stable"
    )
    new_inference.to_csv(
        ctx.metrics_dir / "candidate_inference_new.csv", index=False, encoding="utf-8"
    )
    authoritative.to_csv(
        ctx.metrics_dir / "authoritative_k5_inference.csv", index=False, encoding="utf-8"
    )
    combined = pd.concat([new_inference, authoritative], ignore_index=True, sort=False).sort_values(
        ["panel", "inference_origin", "endpoint"], kind="stable"
    )
    combined.to_csv(
        ctx.metrics_dir / "fig4_candidate_inference.csv", index=False, encoding="utf-8"
    )
    descriptive.to_csv(
        ctx.metrics_dir / "fig4_candidate_descriptive.csv", index=False, encoding="utf-8"
    )
    source_manifest = pd.DataFrame(ctx.source_records).sort_values(
        ["bundle", "network_seed", "relative_path"], kind="stable"
    )
    source_manifest.to_csv(
        ctx.meta_dir / "source_manifest.csv", index=False, encoding="utf-8"
    )
    parent_hashes_unchanged = all(
        _sha256(REPO_ROOT / row.relative_path) == row.sha256
        for row in source_manifest.itertuples(index=False)
    )
    verdict = _build_verdict(
        new=new_inference,
        authoritative=authoritative,
        progressive_audit=progressive_audit,
        behavior_audit=behavior_audit,
        event_audit=event_audit,
    )
    summary = {
        "experiment_id": "fig4_candidate_statistics_20260731",
        "status": "completed",
        "analysis_role": "candidate_plot_only_secondary_reanalysis",
        "network_seeds": list(EXPECTED_SEEDS),
        "n_networks": len(EXPECTED_SEEDS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "random_seed": RANDOM_SEED,
        "audits": {
            "progressive": progressive_audit,
            "behavior": behavior_audit,
            "event": event_audit,
            "fixed_b_authoritative": fixed_b_audit,
        },
        "row_counts": {
            "progressive_network_stage": int(len(progressive)),
            "behavior_network_rates": int(len(rates)),
            "behavior_network_contrasts": int(len(behavior_contrasts)),
            "behavior_depth_network_rates": int(len(behavior_balanced)),
            "event_network_metrics": int(len(event)),
            "k5_fixed_b_network_scalars": int(len(k5_scalars)),
            "new_inference": int(len(new_inference)),
            "authoritative_inference": int(len(authoritative)),
            "descriptive": int(len(descriptive)),
        },
        "parent_hashes_unchanged": bool(parent_hashes_unchanged),
        "display_contract_candidate": {
            "fig4b_display": (
                "relation-balanced K1 versus K5 rescue and loss rates; each outcome retains "
                "its own opportunity denominator"
            ),
            "excluded_from_artwork": [
                "K5_rescue_aligned_minus_mismatched",
                "K5_loss_aligned_minus_mismatched",
            ],
            "statistics_only_not_artwork": [
                "rescue_depth_by_relation_interaction_K5_minus_K1",
                "loss_depth_by_relation_interaction_K5_minus_K1",
            ],
        },
        "verdict": verdict,
    }
    run_config = {
        "source_root": "results/paper_figure_multi_seed and results/multi_seed_rollout",
        "output_dir": _relative(output_dir),
        "network_seeds": list(EXPECTED_SEEDS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "random_seed": RANDOM_SEED,
        "simulation_or_forward_replay": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_native(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(_native(run_config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    ctx.logs.append(f"parent_hashes_unchanged={parent_hashes_unchanged}")
    ctx.logs.append(f"verdict={verdict}")
    (ctx.log_dir / "run.log").write_text("\n".join(ctx.logs) + "\n", encoding="utf-8")
    _write_readme(output_dir, summary)
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifact_manifest = {
        "experiment_id": summary["experiment_id"],
        "files": [
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        ],
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results/paper_figure_multi_seed/fig4_accumulated_history_statistics"
        ),
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    summary = run(output_dir.resolve())
    print(json.dumps(_native(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
