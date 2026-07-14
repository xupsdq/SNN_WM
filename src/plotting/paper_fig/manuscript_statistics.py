from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scipy import optimize, stats
except Exception:  # pragma: no cover - scipy is expected in torch_env.
    optimize = None
    stats = None


EXPECTED_SEEDS = tuple(range(1000, 1020))
MIN_NETWORKS = 3


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Return trapezoidal AUC across NumPy versions."""
    fn = getattr(np, "trapezoid", np.trapz)
    return float(fn(y, x))

LONG_FIELDS = [
    "task_id",
    "fig",
    "panel",
    "metric",
    "group",
    "condition_a",
    "condition_b",
    "n_networks",
    "n_observations",
    "mean",
    "sd",
    "sem",
    "ci95_low",
    "ci95_high",
    "effect",
    "effect_ci95_low",
    "effect_ci95_high",
    "statistic",
    "p_value",
    "p_value_fdr",
    "correction_family",
    "method",
    "source_file",
    "caveat",
]

AUDIT_FIELDS = [
    "task_id",
    "status",
    "expected_seeds",
    "usable_seeds",
    "missing_seeds",
    "source_files",
    "required_columns",
    "row_count_check",
    "calculable",
    "reason",
]


@dataclass(frozen=True)
class SourceSpec:
    path: str
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    fig: str
    panel: str
    claim: str
    sources: tuple[SourceSpec, ...]
    calculator: str
    how: str
    caveat: str = ""


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        "Q01",
        "fig1",
        "B",
        "baseline classification recall remains above 90%.",
        (
            SourceSpec(
                "data/metrics/panel_b_baseline_metrics_by_network.csv",
                ("network_seed", "overall_recall", "n_trials", "n_correct"),
            ),
        ),
        "q01",
        "Seed-level overall_recall; one-sample t-test vs 0.90, plus CI.",
    ),
    TaskSpec(
        "Q02",
        "fig1",
        "C",
        "delay-period STSP states decode sample identity above chance.",
        (
            SourceSpec(
                "data/metrics/panel_c_delay_decode_metrics.csv",
                ("network_seed", "layer", "delay_ms", "acc", "macro_f1", "chance", "n_test"),
            ),
        ),
        "q02",
        "Seed-level held-out acc/macro-F1 by layer and delay; one-sample tests vs chance.",
        "No label-shuffle null distribution is present; chance-level tests are computable.",
    ),
    TaskSpec(
        "Q03",
        "fig1",
        "D,E",
        "donor shuffle shifts attribution and dynamic/static controls change probe disruption.",
        (
            SourceSpec(
                "data/metrics/panel_d_condition_metrics.csv",
                (
                    "network_seed",
                    "condition",
                    "error_rate",
                    "sample_attribution_rate",
                    "donor_attribution_rate",
                    "probe_attribution_rate",
                    "n_trials",
                ),
            ),
            SourceSpec(
                "data/metrics/panel_e_attribution_metrics.csv",
                (
                    "network_seed",
                    "condition",
                    "original_sample_attribution",
                    "donor_sample_attribution",
                    "donor_shift_gain_vs_dynamic",
                    "original_drop_vs_dynamic",
                ),
            ),
        ),
        "q03",
        "Seed-level paired contrasts across dynamic_intact/static_frozen/ux_trial_shuffle.",
    ),
    TaskSpec(
        "Q04",
        "fig2",
        "B",
        "the mixed two-item state remains similar to both constituent references.",
        (
            SourceSpec(
                "data/metrics/panel_b_dual_retention_metrics.csv",
                ("network_seed", "pair_id", "layer", "state_variable", "sim_to_A", "sim_to_B", "sim_to_A_minus_B"),
            ),
        ),
        "q04",
        "Pair-level values are averaged within seed, layer, and state variable; similarities are tested vs 0.",
    ),
    TaskSpec(
        "Q05",
        "fig2",
        "C",
        "true experienced pairs exceed shuffled-pair controls.",
        (
            SourceSpec(
                "data/metrics/panel_c_pair_specificity_metrics.csv",
                ("network_seed", "pair_id", "layer", "state_variable", "true_minus_shuffled"),
            ),
        ),
        "q05",
        "Pair-level true_minus_shuffled is averaged within seed/layer/state variable and tested vs 0.",
    ),
    TaskSpec(
        "Q06",
        "fig2",
        "D",
        "WPRI and beyond-linear residual indices are above zero.",
        (
            SourceSpec(
                "data/metrics/panel_d_pair_level_organization_metrics.csv",
                ("network_seed", "pair_id", "layer", "state_variable", "WPRI"),
            ),
            SourceSpec(
                "data/metrics/panel_d_linear_residual_pair_specificity_metrics.csv",
                ("network_seed", "pair_id", "layer", "state_variable", "beyond_linear_pair_index"),
            ),
        ),
        "q06",
        "Both indices are averaged within seed/layer/state variable and tested vs 0.",
    ),
    TaskSpec(
        "Q07",
        "fig2",
        "E",
        "neutral ping favors pair-member readout under the fused state.",
        (
            SourceSpec(
                "data/metrics/panel_e_neutral_ping_metrics.csv",
                ("network_seed", "state_condition", "P_pair", "P_A", "P_B", "P_other", "P_silent"),
            ),
        ),
        "q07",
        "Seed-level paired contrasts of P_pair for S_AB vs S0/S_A/S_B.",
    ),
    TaskSpec(
        "Q08",
        "fig2",
        "F",
        "fused state improves partial-cue target recovery across dropout levels.",
        (
            SourceSpec(
                "data/metrics/panel_f_partial_cue_metrics.csv",
                ("network_seed", "state_condition", "target_item", "keep_prob", "P_target"),
            ),
        ),
        "q08",
        "AUC over keep_prob is computed within seed/state/target, then paired S_AB-control contrasts are tested.",
    ),
    TaskSpec(
        "Q09",
        "fig3",
        "A",
        "separable item-like traces approach a bounded regime as sequence length increases.",
        (
            SourceSpec(
                "data/metrics/panel_b_morphology_serial_profile.csv",
                ("network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms", "N_eff", "N_eff_fraction"),
            ),
        ),
        "q09",
        "N_eff is deduplicated to sequence-level rows; saturating and linear fits are compared within seed.",
    ),
    TaskSpec(
        "Q10",
        "fig3",
        "B",
        "multi-input state preserves recovery benefit over single-item memory.",
        (
            SourceSpec(
                "data/metrics/panel_d_item_functional_gain.csv",
                (
                    "network_seed",
                    "seq_len",
                    "delay_ms",
                    "target_position",
                    "P_target_sequence_state",
                    "P_target_single_item_memory",
                    "G_i",
                ),
            ),
        ),
        "q10",
        "G_i is averaged within seed/seq_len/delay and tested vs 0.",
    ),
    TaskSpec(
        "Q11",
        "fig3",
        "C",
        "recovery enhancement is content-specific versus mismatched and unseen cues.",
        (
            SourceSpec(
                "data/metrics/panel_c_cue_specificity_metrics.csv",
                ("network_seed", "cue_type", "state_condition", "memory_condition", "target_position", "P_target"),
            ),
        ),
        "q11",
        "Within sequence_state/S_final rows, matched, same-class foil, and unseen cues form three predeclared paired contrasts; BH-FDR is restricted to that Fig.3C family.",
        "BH-FDR is restricted to the three predeclared Q11 Fig.3C cue contrasts.",
    ),
    TaskSpec(
        "Q12",
        "fig3",
        "D",
        "recoverable-item fraction is reproducible across sequence sizes.",
        (
            SourceSpec(
                "data/metrics/panel_d_functional_boundary_metrics.csv",
                ("network_seed", "seq_len", "delay_ms", "rescued_fraction", "accessible_item_count"),
            ),
        ),
        "q12",
        "rescued_fraction is averaged within seed/seq_len/delay and tested vs 0.",
    ),
    TaskSpec(
        "Q13",
        "fig3",
        "E",
        "functional rescue depends jointly on sequence length and delay.",
        (
            SourceSpec(
                "data/metrics/panel_f_boundary_summary.csv",
                ("network_seed", "seq_len", "delay_ms", "rescued_fraction", "functional_retention_index"),
            ),
        ),
        "q13",
        "Per-seed OLS interaction coefficient from rescued_fraction ~ seq_len * delay_ms is tested vs 0.",
    ),
    TaskSpec(
        "Q14",
        "fig4",
        "B",
        "probe bias increases with visual similarity.",
        (
            SourceSpec(
                "data/metrics/panel_b_similarity_entry_metrics.csv",
                ("network_seed", "pixel_similarity", "acc_drop", "b_vec"),
            ),
        ),
        "q14",
        "Per-seed OLS slope of acc_drop against pixel_similarity is tested vs 0.",
    ),
    TaskSpec(
        "Q15",
        "fig4",
        "C",
        "within the highest-similarity bin, the natural high-overlap split shows a descriptive accuracy-drop difference.",
        (
            SourceSpec(
                "data/metrics/panel_c_high_similarity_overlap_accuracy_drop_contrast.csv",
                ("network_seed", "high_minus_low_acc_drop", "n_pairs_high", "n_pairs_low"),
            ),
        ),
        "q15",
        "The precomputed natural high-minus-low overlap accuracy-drop contrast is tested across networks; the endpoint is descriptive and non-causal.",
        "Descriptive, non-causal endpoint; report the raw two-sided P value and confidence interval without multiplicity adjustment.",
    ),
    TaskSpec(
        "Q16",
        "fig4",
        "D",
        "resetting overlap-aligned Layer 1 STSP removes more of the dynamic accuracy drop than the predeclared controls.",
        (
            SourceSpec(
                "data/metrics/panel_d_l1_stsp_overlap_perturbation_contrast.csv",
                ("network_seed", "dynamic_minus_overlap_reset", "random_reset_minus_overlap_reset"),
            ),
        ),
        "q16",
        "Two nonredundant precomputed seed-level Layer 1 reset contrasts are tested vs 0 and form one BH-FDR family.",
    ),
    TaskSpec(
        "Q17",
        "fig4",
        "E",
        "overlap-preserved probe trajectory is more dynamic-like than controls.",
        (
            SourceSpec(
                "data/metrics/panel_e_decision_spike_displacement.csv",
                ("network_seed", "pair_id", "condition", "mean_DPI_L3"),
            ),
        ),
        "q17",
        "mean_DPI_L3 is averaged within seed/condition and paired across perturbation conditions.",
    ),
    TaskSpec(
        "Q18",
        "fig4",
        "F",
        "removal or replacement shifts decision-space responses toward static-like regimes.",
        (
            SourceSpec(
                "data/metrics/supp_decision_deflection_metrics.csv",
                ("network_seed", "pair_id", "condition", "decision_deflection_score", "dynamic_like_recovery"),
            ),
        ),
        "q18",
        "Decision deflection scores are averaged within seed/condition and paired across perturbation conditions.",
    ),
    TaskSpec(
        "Q19",
        "fig5",
        "A",
        "overlap-dominant sites carry stronger retained STSP support than controls.",
        (
            SourceSpec(
                "data/metrics/panel_a_preprobe_support_metrics.csv",
                ("network_seed", "trial_id", "unit_group", "mean_support", "total_support"),
            ),
        ),
        "q19",
        "Unit-group support is averaged within seed/group and paired against controls.",
    ),
    TaskSpec(
        "Q20",
        "fig5",
        "B",
        "overlap-dominant units show more advance/recruit transitions.",
        (
            SourceSpec(
                "data/metrics/panel_b_transition_summary_by_group.csv",
                ("network_seed", "trial_id", "unit_group", "P_advance", "P_recruit", "P_advance_plus_recruit"),
            ),
        ),
        "q20",
        "Transition proportions are averaged within seed/group and paired against controls.",
    ),
    TaskSpec(
        "Q21",
        "fig5",
        "C",
        "selected dynamic winners show a larger full-pre dynamic-minus-static delta-V than their paired losers.",
        (
            SourceSpec(
                "data/metrics/panel_c_winner_loser_network_summary.csv",
                (
                    "network_seed",
                    "winner_minus_loser_full_pre_delta_v_mean",
                    "winner_minus_loser_late_pre_delta_v_mean",
                    "n_trials_eligible",
                    "n_events_eligible",
                ),
            ),
        ),
        "q21",
        "Events are averaged within trial and trials within network; the predeclared -8..-1 ms network contrast is tested vs 0.",
        "Only the full-pre -8..-1 ms endpoint is tested; the -4..-1 ms late-pre window is descriptive only.",
    ),
    TaskSpec(
        "Q22",
        "fig5",
        "D",
        "prior-updated Layer2 sites have higher probe update probability, especially under dynamic probe processing.",
        (
            SourceSpec(
                "data/metrics/panel_postprobe_l2_reupdate_history_composition.csv",
                (
                    "network_seed",
                    "condition",
                    "history_status",
                    "update_probability_given_history",
                    "dynamic_conditional_prior_minus_nonprior",
                    "static_conditional_prior_minus_nonprior",
                    "conditional_difference_in_differences",
                ),
            ),
        ),
        "q22d",
        "Seed-level P(update|prior-updated) and P(update|not-prior-updated) are contrasted within Dynamic and Static opportunity conditions; the dynamic-minus-static difference-in-differences is tested across networks.",
        "Static values are frozen-STSP update opportunities, not actual STSP mutation events.",
    ),
    TaskSpec(
        "Q22E",
        "fig5",
        "E",
        "attenuating and resetting STSP reduce advance/recruit transitions.",
        (
            SourceSpec(
                "data/metrics/panel_d_l1_stsp_perturbation_unit_transitions.csv",
                (
                    "network_seed",
                    "trial_id",
                    "condition",
                    "unit_group",
                    "included_in_main",
                    "first_spike_static",
                    "first_spike_condition",
                ),
            ),
        ),
        "q22e",
        "First-50-ms P(advance OR recruit) is recomputed per condition and network from unit transitions; Dynamic-minus-Attenuate and Dynamic-minus-Reset form one BH-FDR family.",
    ),
    TaskSpec(
        "Q23",
        "fig6",
        "A",
        "high-support pings bias readout composition toward recent items.",
        (
            SourceSpec(
                "data/metrics/panel_b_region_ping_readout_bias.csv",
                ("network_seed", "sequence_id", "entry_condition", "old_mass", "middle_mass", "recent_mass"),
            ),
        ),
        "q23",
        "recent_mass and positional center-of-mass are paired for peak vs valley/random conditions.",
    ),
    TaskSpec(
        "Q24",
        "fig6",
        "B",
        "direct ping Layer 1 spike probability increases with STSP score.",
        (
            SourceSpec(
                "data/metrics/panel_c_global_ping_score_spike_prediction.csv",
                ("network_seed", "sequence_id", "score_quantile_bin", "mean_score", "spike_probability"),
            ),
        ),
        "q24",
        "Per-seed quantile trend slope of spike_probability against mean_score is tested vs 0.",
        "Continuous per-site logistic raw data is not present; quantile trend is computable.",
    ),
    TaskSpec(
        "Q25",
        "fig6",
        "C",
        "weak-cue dynamic-over-baseline firing increase follows STSP score.",
        (
            SourceSpec(
                "data/metrics/panel_d_real_probe_score_spike_deflection.csv",
                ("network_seed", "sequence_id", "probe_id", "early_window_ms", "score_quantile_bin", "mean_score", "delta_spike_probability"),
            ),
        ),
        "q25",
        "Per-seed score-quantile trend slopes are tested by early response window.",
        "Continuous per-site logistic raw data is not present; quantile trend is computable.",
    ),
    TaskSpec(
        "Q26",
        "fig6",
        "D",
        "removing high-STSP overlap sites reduces recruitment more than matched controls.",
        (
            SourceSpec(
                "data/metrics/supp_s11f_high_stsp_ablation_paired_difference.csv",
                ("network_seed", "sequence_id", "probe_id", "value", "high_stsp_overlap", "matched_removal"),
            ),
        ),
        "q26",
        "Sequence/probe paired differences are averaged within network before the two-sided network-level test and t-based 95% CI.",
    ),
    TaskSpec(
        "Q27",
        "fig6",
        "E",
        "high-STSP recruitment gain is gated by probe overlap.",
        (
            SourceSpec(
                "data/metrics/panel_e_overlap_gated_stsp_interaction.csv",
                ("network_seed", "early_window_ms", "interaction_delta", "stsp_effect_with_overlap", "stsp_effect_without_overlap"),
            ),
        ),
        "q27",
        "Seed-level interaction_delta is tested vs 0 at 5/10/15/20 ms; the four windows form one BH-FDR family.",
    ),
)


FIGURE_ROOTS = {
    "fig1": ("fig1_functional_stsp_substrate", "fig1_functional_stsp_substrate"),
    "fig2": ("fig2_pair_fused_stsp_state", "fig2_pair_fused_stsp_state"),
    "fig3": ("fig3_multiitem_peak_landscape",),
    "fig4": ("fig4_overlap_reentry",),
    "fig5": ("fig5_local_support_competition",),
    "fig6": ("fig6_peak_amplified_reentry",),
}


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.12g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _seed_text(seeds: Sequence[int]) -> str:
    if not seeds:
        return ""
    if list(seeds) == list(range(int(seeds[0]), int(seeds[-1]) + 1)):
        return f"{int(seeds[0])}..{int(seeds[-1])}"
    return ",".join(str(int(seed)) for seed in seeds)


def _figure_root(root: Path, fig: str) -> Path:
    rel = FIGURE_ROOTS[fig]
    path = Path(root)
    for part in rel:
        path = path / part
    return path


def _seed_dirs(root: Path, fig: str, expected_seeds: Sequence[int]) -> dict[int, Path]:
    fig_root = _figure_root(root, fig)
    return {int(seed): fig_root / f"seed_{int(seed)}" for seed in expected_seeds}


def _read_csvs(root: Path, task: TaskSpec, rel_path: str, expected_seeds: Sequence[int]) -> tuple[pd.DataFrame, list[int]]:
    frames: list[pd.DataFrame] = []
    present: list[int] = []
    for seed, seed_dir in _seed_dirs(root, task.fig, expected_seeds).items():
        path = seed_dir / rel_path
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if "network_seed" not in frame.columns:
            frame["network_seed"] = int(seed)
        frame["_source_file"] = rel_path
        frames.append(frame)
        present.append(int(seed))
    if not frames:
        return pd.DataFrame(), present
    return pd.concat(frames, ignore_index=True), present


def _audit_task(root: Path, task: TaskSpec, expected_seeds: Sequence[int]) -> dict[str, str]:
    source_parts: list[str] = []
    required_parts: list[str] = []
    row_parts: list[str] = []
    usable_sets: list[set[int]] = []
    reasons: list[str] = []
    for source in task.sources:
        present: set[int] = set()
        row_counts: list[int] = []
        missing_cols: dict[int, list[str]] = {}
        for seed, seed_dir in _seed_dirs(root, task.fig, expected_seeds).items():
            path = seed_dir / source.path
            if not path.is_file():
                continue
            present.add(int(seed))
            try:
                header = pd.read_csv(path, nrows=0)
                missing = [col for col in source.required_columns if col not in header.columns]
                if missing:
                    missing_cols[int(seed)] = missing
                row_counts.append(sum(1 for _ in path.open("rb")) - 1)
            except Exception as exc:
                missing_cols[int(seed)] = [f"read_error:{exc}"]
        usable_sets.append(present.difference(missing_cols))
        source_parts.append(source.path)
        required_parts.append(f"{source.path}: {','.join(source.required_columns)}")
        unique_counts = sorted(set(row_counts))
        row_text = f"{source.path}: {len(present)}/{len(expected_seeds)}"
        if unique_counts:
            row_text += f" rows={unique_counts[:8]}"
        row_parts.append(row_text)
        if missing_cols:
            reasons.append(f"{source.path} missing columns/read errors in seeds {sorted(missing_cols)}")

    usable = set(expected_seeds) if not usable_sets else set.intersection(*usable_sets)
    missing = sorted(set(expected_seeds).difference(usable))
    if len(usable) == len(expected_seeds) and not reasons:
        status = "ok"
        reason = "All required sources are present for expected seeds."
    elif len(usable) >= MIN_NETWORKS:
        status = "partial"
        reason = "Computable with missing seeds/sources: " + ",".join(str(seed) for seed in missing)
        if reasons:
            reason += "; " + "; ".join(reasons)
    else:
        status = "unavailable"
        reason = "Not enough usable network seeds."
        if reasons:
            reason += " " + "; ".join(reasons)

    return {
        "task_id": task.task_id,
        "status": status,
        "expected_seeds": _seed_text(expected_seeds),
        "usable_seeds": _seed_text(sorted(usable)),
        "missing_seeds": ",".join(str(seed) for seed in missing),
        "source_files": "; ".join(source_parts),
        "required_columns": " | ".join(required_parts),
        "row_count_check": " | ".join(row_parts),
        "calculable": "true" if len(usable) >= MIN_NETWORKS else "false",
        "reason": reason,
    }


def _values(values: Iterable[Any]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    arr = series.to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _mean_ci(values: Iterable[Any]) -> dict[str, float]:
    arr = _values(values)
    n = int(arr.size)
    if n == 0:
        return {"mean": math.nan, "sd": math.nan, "sem": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = float(sd / math.sqrt(n)) if n > 1 else 0.0
    if n > 1 and sem > 0:
        tcrit = float(stats.t.ppf(0.975, n - 1)) if stats is not None else 1.96
        low = mean - tcrit * sem
        high = mean + tcrit * sem
    else:
        low = mean
        high = mean
    return {"mean": mean, "sd": sd, "sem": sem, "ci95_low": low, "ci95_high": high}


def _one_sample_test(values: Iterable[Any], reference: float = 0.0) -> tuple[str, float, float, float]:
    arr = _values(values)
    if stats is None or arr.size < 3:
        return ("one_sample_t_skipped", math.nan, math.nan, math.nan)
    result = stats.ttest_1samp(arr, popmean=float(reference), nan_policy="omit")
    sd = float(np.std(arr, ddof=1))
    effect_size = float((np.mean(arr) - reference) / sd) if sd > 0 else math.nan
    return ("one_sample_t", float(result.statistic), float(result.pvalue), effect_size)


def _paired_test(values_a: Iterable[Any], values_b: Iterable[Any]) -> tuple[str, float, float, float]:
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]
    if stats is None or a.size < 3:
        return ("paired_t_skipped", math.nan, math.nan, math.nan)
    diff = a - b
    result = stats.ttest_rel(a, b, nan_policy="omit")
    sd = float(np.std(diff, ddof=1))
    effect_size = float(np.mean(diff) / sd) if sd > 0 else math.nan
    return ("paired_t", float(result.statistic), float(result.pvalue), effect_size)


def _row(
    task: TaskSpec,
    *,
    metric: str,
    group: str = "",
    condition_a: str = "",
    condition_b: str = "",
    values: Iterable[Any] | None = None,
    effect_values: Iterable[Any] | None = None,
    reference: float | None = None,
    method: str = "describe",
    statistic: float = math.nan,
    p_value: float = math.nan,
    source_file: str | None = None,
    caveat: str | None = None,
) -> dict[str, str]:
    vals = _values(values if values is not None else [])
    eff = _values(effect_values if effect_values is not None else vals)
    desc = _mean_ci(vals)
    eff_desc = _mean_ci(eff)
    effect = eff_desc["mean"] if effect_values is not None else (
        desc["mean"] - float(reference) if reference is not None and math.isfinite(desc["mean"]) else math.nan
    )
    return {
        "task_id": task.task_id,
        "fig": task.fig,
        "panel": task.panel,
        "metric": metric,
        "group": group,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "n_networks": str(int(vals.size)),
        "n_observations": str(int(vals.size)),
        "mean": _fmt(desc["mean"]),
        "sd": _fmt(desc["sd"]),
        "sem": _fmt(desc["sem"]),
        "ci95_low": _fmt(desc["ci95_low"]),
        "ci95_high": _fmt(desc["ci95_high"]),
        "effect": _fmt(effect),
        "effect_ci95_low": _fmt(eff_desc["ci95_low"] if effect_values is not None else math.nan),
        "effect_ci95_high": _fmt(eff_desc["ci95_high"] if effect_values is not None else math.nan),
        "statistic": _fmt(statistic),
        "p_value": _fmt(p_value),
        "p_value_fdr": "",
        "correction_family": "",
        "method": method,
        "source_file": source_file or "; ".join(source.path for source in task.sources),
        "caveat": caveat if caveat is not None else task.caveat,
    }


def _group_label(values: Mapping[str, Any]) -> str:
    return ";".join(f"{key}={value}" for key, value in values.items() if str(value) != "")


def _seed_aggregate(df: pd.DataFrame, value_col: str, group_cols: Sequence[str] = ()) -> pd.DataFrame:
    cols = ["network_seed", *group_cols, value_col]
    work = df.loc[:, [col for col in cols if col in df.columns]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    return work.groupby(["network_seed", *group_cols], dropna=False, as_index=False)[value_col].mean()


def _one_sample_rows(
    task: TaskSpec,
    df: pd.DataFrame,
    value_col: str,
    *,
    metric: str,
    group_cols: Sequence[str] = (),
    reference: float = 0.0,
    source_file: str | None = None,
) -> list[dict[str, str]]:
    agg = _seed_aggregate(df, value_col, group_cols)
    grouped = [((), agg)] if not group_cols else agg.groupby(list(group_cols), dropna=False)
    rows: list[dict[str, str]] = []
    for key, sub in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        group = _group_label({col: key_tuple[i] for i, col in enumerate(group_cols)})
        values = _values(sub[value_col])
        method, statistic, p_value, _effect_size = _one_sample_test(values, reference)
        rows.append(
            _row(
                task,
                metric=metric,
                group=group,
                condition_a=f"mean-{reference:g}",
                values=values,
                reference=reference,
                method=method,
                statistic=statistic,
                p_value=p_value,
                source_file=source_file,
            )
        )
    return rows


def _paired_rows(
    task: TaskSpec,
    df: pd.DataFrame,
    value_col: str,
    *,
    factor_col: str,
    pairs: Sequence[tuple[str, str]],
    metric: str,
    group_cols: Sequence[str] = (),
    source_file: str | None = None,
) -> list[dict[str, str]]:
    agg = _seed_aggregate(df, value_col, [*group_cols, factor_col])
    index_cols = ["network_seed", *group_cols]
    rows: list[dict[str, str]] = []
    if agg.empty:
        return rows
    pivot = agg.pivot_table(index=index_cols, columns=factor_col, values=value_col, aggfunc="mean")
    for condition_a, condition_b in pairs:
        if condition_a not in pivot.columns or condition_b not in pivot.columns:
            continue
        compare = pivot[[condition_a, condition_b]].dropna()
        if not group_cols:
            grouped = [((), compare)]
        else:
            level: int | list[int] = 1 if len(group_cols) == 1 else list(range(1, 1 + len(group_cols)))
            grouped = compare.groupby(level=level, dropna=False)
        for key, sub in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            group = _group_label({col: key_tuple[i] for i, col in enumerate(group_cols)})
            a = sub[condition_a].to_numpy(dtype=float)
            b = sub[condition_b].to_numpy(dtype=float)
            diff = a - b
            method, statistic, p_value, _effect_size = _paired_test(a, b)
            rows.append(
                _row(
                    task,
                    metric=metric,
                    group=group,
                    condition_a=condition_a,
                    condition_b=condition_b,
                    values=diff,
                    effect_values=diff,
                    method=method,
                    statistic=statistic,
                    p_value=p_value,
                    source_file=source_file,
                )
            )
    return rows


def _column_contrast_rows(
    task: TaskSpec,
    df: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    *,
    source_file: str | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for col, metric in columns:
        agg = _seed_aggregate(df, col)
        values = _values(agg[col])
        method, statistic, p_value, _effect_size = _one_sample_test(values, 0.0)
        rows.append(
            _row(
                task,
                metric=metric,
                condition_a=f"{col}-0",
                values=values,
                reference=0.0,
                method=method,
                statistic=statistic,
                p_value=p_value,
                source_file=source_file,
            )
        )
    return rows


def _trend_slope_rows(
    task: TaskSpec,
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    metric: str,
    group_cols: Sequence[str] = (),
    source_file: str | None = None,
) -> list[dict[str, str]]:
    work = df.loc[:, [col for col in ["network_seed", *group_cols, x_col, y_col] if col in df.columns]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[x_col, y_col])
    agg = work.groupby(["network_seed", *group_cols, x_col], dropna=False, as_index=False)[y_col].mean()
    slope_rows: list[dict[str, Any]] = []
    for key, sub in agg.groupby(["network_seed", *group_cols], dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        seed = key_tuple[0]
        group_values = {col: key_tuple[i + 1] for i, col in enumerate(group_cols)}
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        if len(np.unique(x)) < 2:
            continue
        slope = float(np.polyfit(x, y, 1)[0])
        slope_rows.append({"network_seed": seed, **group_values, "slope": slope})
    if not slope_rows:
        return []
    return _one_sample_rows(
        task,
        pd.DataFrame(slope_rows),
        "slope",
        metric=metric,
        group_cols=group_cols,
        reference=0.0,
        source_file=source_file,
    )


def _auc_rows(
    task: TaskSpec,
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    factor_col: str,
    pairs: Sequence[tuple[str, str]],
    group_cols: Sequence[str] = (),
    source_file: str | None = None,
) -> list[dict[str, str]]:
    work = df.loc[:, [col for col in ["network_seed", *group_cols, factor_col, x_col, y_col] if col in df.columns]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[x_col, y_col])
    auc_records: list[dict[str, Any]] = []
    for key, sub in work.groupby(["network_seed", *group_cols, factor_col], dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        seed = key_tuple[0]
        group_values = {col: key_tuple[i + 1] for i, col in enumerate(group_cols)}
        condition = key_tuple[1 + len(group_cols)]
        ordered = sub.sort_values(x_col)
        x = ordered[x_col].to_numpy(dtype=float)
        y = ordered[y_col].to_numpy(dtype=float)
        if len(np.unique(x)) < 2:
            continue
        auc_records.append({"network_seed": seed, **group_values, factor_col: condition, "auc": _trapezoid(y, x)})
    if not auc_records:
        return []
    return _paired_rows(
        task,
        pd.DataFrame(auc_records),
        "auc",
        factor_col=factor_col,
        pairs=pairs,
        metric=f"{y_col}_auc",
        group_cols=group_cols,
        source_file=source_file,
    )


def _calc_q01(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _one_sample_rows(task, df, "overall_recall", metric="overall_recall_vs_0.90", reference=0.90)


def _calc_q02(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    rows = _one_sample_rows(task, df, "acc", metric="heldout_accuracy_vs_chance", group_cols=("layer", "delay_ms"), reference=0.10)
    rows.extend(_one_sample_rows(task, df, "macro_f1", metric="macro_f1_vs_chance", group_cols=("layer", "delay_ms"), reference=0.10))
    return rows


def _calc_q03(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    cond_df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    attr_df, _present = _read_csvs(root, task, task.sources[1].path, seeds)
    rows: list[dict[str, str]] = []
    rows.extend(
        _paired_rows(
            task,
            cond_df,
            "error_rate",
            factor_col="condition",
            pairs=(("dynamic_intact", "static_frozen"), ("ux_trial_shuffle", "dynamic_intact")),
            metric="probe_error_rate_contrast",
            source_file=task.sources[0].path,
        )
    )
    rows.extend(
        _paired_rows(
            task,
            attr_df,
            "donor_sample_attribution",
            factor_col="condition",
            pairs=(("ux_trial_shuffle", "dynamic_intact"),),
            metric="donor_attribution_gain",
            source_file=task.sources[1].path,
        )
    )
    rows.extend(
        _paired_rows(
            task,
            attr_df,
            "original_sample_attribution",
            factor_col="condition",
            pairs=(("dynamic_intact", "ux_trial_shuffle"),),
            metric="original_attribution_drop",
            source_file=task.sources[1].path,
        )
    )
    return rows


def _calc_q04(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    rows = _one_sample_rows(task, df, "sim_to_A", metric="sim_to_A_vs_zero", group_cols=("layer", "state_variable"))
    rows.extend(_one_sample_rows(task, df, "sim_to_B", metric="sim_to_B_vs_zero", group_cols=("layer", "state_variable")))
    rows.extend(_one_sample_rows(task, df, "sim_to_A_minus_B", metric="sim_to_A_minus_B_balance", group_cols=("layer", "state_variable")))
    return rows


def _calc_q05(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _one_sample_rows(task, df, "true_minus_shuffled", metric="true_minus_shuffled_pair_specificity", group_cols=("layer", "state_variable"))


def _calc_q06(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    wpri, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    residual, _present = _read_csvs(root, task, task.sources[1].path, seeds)
    rows = _one_sample_rows(task, wpri, "WPRI", metric="WPRI_vs_zero", group_cols=("layer", "state_variable"), source_file=task.sources[0].path)
    rows.extend(
        _one_sample_rows(
            task,
            residual,
            "beyond_linear_pair_index",
            metric="beyond_linear_pair_index_vs_zero",
            group_cols=("layer", "state_variable"),
            source_file=task.sources[1].path,
        )
    )
    return rows


def _calc_q07(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _paired_rows(
        task,
        df,
        "P_pair",
        factor_col="state_condition",
        pairs=(("S_AB", "S0"), ("S_AB", "S_A"), ("S_AB", "S_B")),
        metric="pair_member_mass",
    )


def _calc_q08(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _auc_rows(
        task,
        df,
        x_col="keep_prob",
        y_col="P_target",
        factor_col="state_condition",
        pairs=(("S_AB", "S0"), ("S_AB", "S_A"), ("S_AB", "S_B")),
        group_cols=("target_item",),
    )


def _saturating_model(x: np.ndarray, asymptote: float, rate: float) -> np.ndarray:
    return asymptote * (1.0 - np.exp(-rate * x))


def _calc_q09(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    dedup = df.drop_duplicates(["network_seed", "condition_id", "sequence_id", "seq_len", "delay_ms"])
    records: list[dict[str, Any]] = []
    for seed, sub in dedup.groupby("network_seed"):
        curve = sub.groupby("seq_len", as_index=False)["N_eff"].mean().sort_values("seq_len")
        x = curve["seq_len"].to_numpy(dtype=float)
        y = curve["N_eff"].to_numpy(dtype=float)
        if len(x) < 3:
            continue
        linear_pred = np.polyval(np.polyfit(x, y, 1), x)
        linear_sse = float(np.sum((y - linear_pred) ** 2))
        sat_sse = math.nan
        asymptote = math.nan
        if optimize is not None:
            try:
                params, _cov = optimize.curve_fit(
                    _saturating_model,
                    x,
                    y,
                    p0=(max(float(np.max(y)), 1e-6), 0.2),
                    bounds=([0.0, 0.0], [np.inf, np.inf]),
                    maxfev=10000,
                )
                asymptote = float(params[0])
                sat_pred = _saturating_model(x, float(params[0]), float(params[1]))
                sat_sse = float(np.sum((y - sat_pred) ** 2))
            except Exception:
                pass
        records.append({"network_seed": seed, "linear_minus_saturating_sse": linear_sse - sat_sse, "saturating_asymptote": asymptote})
    result = pd.DataFrame(records)
    rows = _one_sample_rows(task, result, "linear_minus_saturating_sse", metric="linear_minus_saturating_sse_vs_zero")
    rows.extend(_one_sample_rows(task, result, "saturating_asymptote", metric="saturating_asymptote", reference=0.0))
    return rows


def _calc_q10(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    rows = _one_sample_rows(task, df, "G_i", metric="recovery_gain_G_i", group_cols=("seq_len", "delay_ms"))
    rows.extend(_one_sample_rows(task, df, "G_i", metric="recovery_gain_G_i_overall"))
    return rows


def _calc_q11(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    work = df[
        df["state_condition"].astype(str).eq("S_final")
        & df["memory_condition"].astype(str).eq("sequence_state")
    ].copy()
    rows = _paired_rows(
        task,
        work,
        "P_target",
        factor_col="cue_type",
        pairs=(("matched", "mismatched"), ("matched", "unseen")),
        metric="matched_cue_specificity",
    )
    rows.extend(
        _paired_rows(
            task,
            work,
            "P_target",
            factor_col="cue_type",
            pairs=(("mismatched", "unseen"),),
            metric="same_label_foil_vs_unseen_cue_access",
        )
    )
    return rows


def _calc_q12(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    rows = _one_sample_rows(task, df, "rescued_fraction", metric="rescued_fraction_vs_zero", group_cols=("seq_len", "delay_ms"))
    rows.extend(_one_sample_rows(task, df, "rescued_fraction", metric="rescued_fraction_overall"))
    return rows


def _interaction_coefs(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for seed, sub in df.groupby("network_seed"):
        work = sub.loc[:, ["seq_len", "delay_ms", y_col]].dropna().copy()
        if len(work) < 4:
            continue
        k = (pd.to_numeric(work["seq_len"], errors="coerce").to_numpy(dtype=float) - work["seq_len"].mean()) / max(float(work["seq_len"].std(ddof=0)), 1.0)
        d = (pd.to_numeric(work["delay_ms"], errors="coerce").to_numpy(dtype=float) - work["delay_ms"].mean()) / max(float(work["delay_ms"].std(ddof=0)), 1.0)
        y = pd.to_numeric(work[y_col], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(k) & np.isfinite(d) & np.isfinite(y)
        if keep.sum() < 4:
            continue
        xmat = np.column_stack([np.ones(keep.sum()), k[keep], d[keep], k[keep] * d[keep]])
        beta, *_ = np.linalg.lstsq(xmat, y[keep], rcond=None)
        records.append({"network_seed": seed, "interaction_beta": float(beta[3])})
    return pd.DataFrame(records)


def _calc_q13(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    beta = _interaction_coefs(df, "rescued_fraction")
    rows = _one_sample_rows(task, beta, "interaction_beta", metric="rescued_fraction_seq_len_x_delay_interaction")
    rows.extend(_one_sample_rows(task, df, "rescued_fraction", metric="rescued_fraction_cell_estimate", group_cols=("seq_len", "delay_ms")))
    return rows


def _calc_q14(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _trend_slope_rows(task, df, x_col="pixel_similarity", y_col="acc_drop", metric="acc_drop_vs_pixel_similarity_slope")


def _calc_q15(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _column_contrast_rows(
        task,
        df,
        (("high_minus_low_acc_drop", "high_minus_low_overlap_accuracy_drop"),),
    )


def _calc_q16(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _column_contrast_rows(
        task,
        df,
        (
            ("dynamic_minus_overlap_reset", "dynamic_minus_overlap_reset_accuracy_drop"),
            ("random_reset_minus_overlap_reset", "random_matched_reset_minus_overlap_reset_accuracy_drop"),
        ),
    )


def _calc_q17(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _paired_rows(
        task,
        df,
        "mean_DPI_L3",
        factor_col="condition",
        pairs=(
            ("sample_keep_overlap_only_dynamic", "sample_keep_nonoverlap_only_dynamic"),
            ("sample_keep_overlap_only_dynamic", "sample_random_matched_dynamic"),
            ("sample_keep_overlap_only_dynamic", "full_static"),
        ),
        metric="mean_DPI_L3_condition_contrast",
    )


def _calc_q18(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _paired_rows(
        task,
        df,
        "decision_deflection_score",
        factor_col="condition",
        pairs=(
            ("sample_keep_overlap_only_dynamic", "sample_keep_nonoverlap_only_dynamic"),
            ("sample_keep_overlap_only_dynamic", "sample_random_matched_dynamic"),
            ("full_dynamic", "sample_keep_overlap_only_dynamic"),
        ),
        metric="decision_deflection_score_condition_contrast",
    )


def _calc_q19(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _paired_rows(
        task,
        df,
        "mean_support",
        factor_col="unit_group",
        pairs=(
            ("overlap_dominant", "probe_only_dominant"),
            ("overlap_dominant", "balanced"),
            ("overlap_dominant", "random_matched"),
        ),
        metric="preprobe_mean_support",
    )


def _calc_q20(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    rows: list[dict[str, str]] = []
    for metric_col in ("P_advance", "P_recruit", "P_advance_plus_recruit"):
        rows.extend(
            _paired_rows(
                task,
                df,
                metric_col,
                factor_col="unit_group",
                pairs=(
                    ("overlap_dominant", "probe_only_dominant"),
                    ("overlap_dominant", "balanced"),
                    ("overlap_dominant", "random_matched"),
                ),
                metric=metric_col,
            )
        )
    return rows


def _calc_q21(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    rows = _one_sample_rows(
        task,
        df,
        "winner_minus_loser_full_pre_delta_v_mean",
        metric="winner_minus_loser_full_pre_delta_v_mean",
    )
    late = _seed_aggregate(df, "winner_minus_loser_late_pre_delta_v_mean")
    rows.append(
        _row(
            task,
            metric="winner_minus_loser_late_pre_delta_v_mean",
            values=late["winner_minus_loser_late_pre_delta_v_mean"],
            method="describe",
            caveat="Descriptive -4..-1 ms window; no hypothesis test or multiplicity adjustment.",
        )
    )
    return rows


def _calc_q22d(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    if df.empty:
        return []
    work = df.copy()
    for col in (
        "update_probability_given_history",
        "dynamic_conditional_prior_minus_nonprior",
        "static_conditional_prior_minus_nonprior",
        "conditional_difference_in_differences",
    ):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    by_history = work.pivot_table(
        index=["network_seed", "condition"],
        columns="history_status",
        values="update_probability_given_history",
        aggfunc="mean",
    ).reset_index()
    rows: list[dict[str, str]] = []
    if {"prior_updated", "not_prior_updated"}.issubset(set(by_history.columns)):
        by_history["prior_minus_nonprior"] = by_history["prior_updated"] - by_history["not_prior_updated"]
        wide = by_history.pivot_table(
            index="network_seed",
            columns="condition",
            values="prior_minus_nonprior",
            aggfunc="mean",
        ).reset_index()
        seed_support = pd.DataFrame({"network_seed": wide["network_seed"]})
        if "dynamic_intact" in wide.columns:
            seed_support["dynamic_update_probability_prior_minus_nonprior"] = wide["dynamic_intact"]
        if "static_opportunity" in wide.columns:
            seed_support["static_opportunity_probability_prior_minus_nonprior"] = wide["static_opportunity"]
        if {"dynamic_intact", "static_opportunity"}.issubset(set(wide.columns)):
            seed_support["conditional_difference_in_differences"] = wide["dynamic_intact"] - wide["static_opportunity"]
    else:
        seed_support = work.groupby("network_seed", as_index=False)[
            [
                "dynamic_conditional_prior_minus_nonprior",
                "static_conditional_prior_minus_nonprior",
                "conditional_difference_in_differences",
            ]
        ].mean()
        seed_support["dynamic_update_probability_prior_minus_nonprior"] = seed_support["dynamic_conditional_prior_minus_nonprior"]
        seed_support["static_opportunity_probability_prior_minus_nonprior"] = seed_support["static_conditional_prior_minus_nonprior"]
    if not seed_support.empty:
        rows.extend(
            _column_contrast_rows(
                task,
                seed_support,
                (
                    ("dynamic_update_probability_prior_minus_nonprior", "dynamic_update_probability_prior_minus_nonprior"),
                    ("conditional_difference_in_differences", "conditional_difference_in_differences"),
                    ("static_opportunity_probability_prior_minus_nonprior", "static_opportunity_probability_prior_minus_nonprior"),
                ),
                source_file=task.sources[0].path,
            )
        )
    return rows


def _calc_q22e(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    records: list[dict[str, Any]] = []
    source = task.sources[0]
    required = list(source.required_columns)
    for seed, seed_dir in _seed_dirs(root, task.fig, seeds).items():
        path = seed_dir / source.path
        if not path.is_file():
            continue
        frame = pd.read_csv(path, usecols=required)
        use = frame[
            frame["condition"].astype(str).isin(("dynamic_intact", "attenuate_l1_stsp", "reset_l1_stsp"))
            & frame["included_in_main"].astype(str).str.lower().isin(("true", "1", "yes"))
        ].copy()
        if use.empty:
            continue
        config_path = seed_dir / "run_config.json"
        dt_seconds = 0.001
        if config_path.is_file():
            try:
                dt_seconds = float(json.loads(config_path.read_text(encoding="utf-8")).get("dt", dt_seconds))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        window_steps = max(1, int(round(50.0 / max(dt_seconds * 1000.0, 1e-12))))
        static = pd.to_numeric(use["first_spike_static"], errors="coerce").fillna(-1).astype(int)
        condition = pd.to_numeric(use["first_spike_condition"], errors="coerce").fillna(-1).astype(int)
        static = static.where((static >= 0) & (static < window_steps), -1)
        condition = condition.where((condition >= 0) & (condition < window_steps), -1)
        use["advance_or_recruit"] = (
            ((condition >= 0) & (static < 0))
            | ((condition >= 0) & (static >= 0) & (condition < static))
        ).astype(float)
        values = use.groupby("condition", sort=False)["advance_or_recruit"].mean()
        if not {"dynamic_intact", "attenuate_l1_stsp", "reset_l1_stsp"}.issubset(values.index):
            continue
        records.append(
            {
                "network_seed": int(seed),
                "dynamic_minus_attenuate_P_advance_or_recruit_50ms": float(values["dynamic_intact"] - values["attenuate_l1_stsp"]),
                "dynamic_minus_reset_P_advance_or_recruit_50ms": float(values["dynamic_intact"] - values["reset_l1_stsp"]),
            }
        )
    df = pd.DataFrame(records)
    return _column_contrast_rows(
        task,
        df,
        (
            ("dynamic_minus_attenuate_P_advance_or_recruit_50ms", "dynamic_minus_attenuate_P_advance_or_recruit_50ms"),
            ("dynamic_minus_reset_P_advance_or_recruit_50ms", "dynamic_minus_reset_P_advance_or_recruit_50ms"),
        ),
    )


def _calc_q23(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    mass = df[["old_mass", "middle_mass", "recent_mass"]].sum(axis=1).replace(0, np.nan)
    df = df.copy()
    df["positional_com"] = (df["old_mass"] * 1.0 + df["middle_mass"] * 2.0 + df["recent_mass"] * 3.0) / mass
    rows = _paired_rows(
        task,
        df,
        "recent_mass",
        factor_col="entry_condition",
        pairs=(("peak", "valley"), ("peak", "random")),
        metric="recent_mass",
    )
    rows.extend(
        _paired_rows(
            task,
            df,
            "positional_com",
            factor_col="entry_condition",
            pairs=(("peak", "valley"), ("peak", "random")),
            metric="positional_center_of_mass",
        )
    )
    return rows


def _calc_q24(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _trend_slope_rows(task, df, x_col="mean_score", y_col="spike_probability", metric="spike_probability_vs_STSP_score_slope")


def _calc_q25(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _trend_slope_rows(
        task,
        df,
        x_col="mean_score",
        y_col="delta_spike_probability",
        metric="delta_spike_probability_vs_STSP_score_slope",
        group_cols=("early_window_ms",),
    )


def _calc_q26(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _one_sample_rows(task, df, "value", metric="high_stsp_overlap_minus_matched_loss")


def _calc_q27(root: Path, task: TaskSpec, seeds: Sequence[int]) -> list[dict[str, str]]:
    df, _present = _read_csvs(root, task, task.sources[0].path, seeds)
    return _one_sample_rows(task, df, "interaction_delta", metric="interaction_delta", group_cols=("early_window_ms",))


CALCULATORS: dict[str, Callable[[Path, TaskSpec, Sequence[int]], list[dict[str, str]]]] = {
    "q01": _calc_q01,
    "q02": _calc_q02,
    "q03": _calc_q03,
    "q04": _calc_q04,
    "q05": _calc_q05,
    "q06": _calc_q06,
    "q07": _calc_q07,
    "q08": _calc_q08,
    "q09": _calc_q09,
    "q10": _calc_q10,
    "q11": _calc_q11,
    "q12": _calc_q12,
    "q13": _calc_q13,
    "q14": _calc_q14,
    "q15": _calc_q15,
    "q16": _calc_q16,
    "q17": _calc_q17,
    "q18": _calc_q18,
    "q19": _calc_q19,
    "q20": _calc_q20,
    "q21": _calc_q21,
    "q22d": _calc_q22d,
    "q22e": _calc_q22e,
    "q23": _calc_q23,
    "q24": _calc_q24,
    "q25": _calc_q25,
    "q26": _calc_q26,
    "q27": _calc_q27,
}


def _apply_bh_fdr(
    rows: list[dict[str, str]],
    *,
    predicate: Callable[[dict[str, str]], bool] | None = None,
) -> None:
    indexed: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        if predicate is not None and not predicate(row):
            continue
        try:
            p_value = float(row.get("p_value") or "nan")
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
    for reverse_index, (_row_index, p_value) in enumerate(reversed(ordered), start=1):
        rank = m - reverse_index + 1
        running = min(running, p_value * m / rank)
        adjusted[rank - 1] = min(running, 1.0)
    for (row_index, _p), p_adj in zip(ordered, adjusted):
        rows[row_index]["p_value_fdr"] = _fmt(p_adj)


def _apply_fdr(rows: list[dict[str, str]]) -> None:
    """Apply only the predeclared global and endpoint-specific families."""
    for row in rows:
        row["p_value_fdr"] = ""
        row["correction_family"] = "global_manuscript_remaining"
    _apply_bh_fdr(rows)
    families = {
        "Q11": "Q11_cue_specificity_3",
        "Q16": "Q16_l1_reset_planned_2",
        "Q21": "Q21_full_pre_primary_1",
        "Q22E": "Q22E_advance_or_recruit_planned_2",
        "Q27": "Q27_interaction_windows_4",
    }
    for task_id, family in families.items():
        _apply_bh_fdr(rows, predicate=lambda row, task_id=task_id: row.get("task_id") == task_id and row.get("method") != "describe")
        for row in rows:
            if row.get("task_id") == task_id and row.get("method") != "describe":
                row["correction_family"] = family
    for row in rows:
        if row.get("task_id") == "Q15":
            row["p_value_fdr"] = ""
            row["correction_family"] = "none_descriptive_unadjusted"
        elif row.get("method") == "describe":
            row["p_value_fdr"] = ""
            row["correction_family"] = "not_applicable_descriptive"


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json_summary(
    path: Path,
    *,
    paper_fig_root: Path,
    audit_rows: Sequence[Mapping[str, str]],
    stat_rows: Sequence[Mapping[str, str]],
) -> None:
    figure_counts = pd.Series([row.get("fig", "") for row in stat_rows]).value_counts().to_dict()
    status_counts = pd.Series([row.get("status", "") for row in audit_rows]).value_counts().to_dict()
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_fig_root": str(paper_fig_root),
        "row_count": int(len(stat_rows)),
        "figure_counts": {str(key): int(value) for key, value in figure_counts.items() if str(key)},
        "audit_status_counts": {str(key): int(value) for key, value in status_counts.items() if str(key)},
        "rows": [dict(row) for row in stat_rows],
        "audit": [dict(row) for row in audit_rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report_status(status: str) -> str:
    return {"ok": "可计算", "partial": "部分可计算", "unavailable": "不可计算"}.get(status, status)


def _write_report(path: Path, audit_rows: Sequence[Mapping[str, str]], stat_rows: Sequence[Mapping[str, str]]) -> None:
    by_task_stats: dict[str, list[Mapping[str, str]]] = {}
    for row in stat_rows:
        by_task_stats.setdefault(str(row["task_id"]), []).append(row)
    status_counts = pd.Series([row["status"] for row in audit_rows]).value_counts().to_dict()
    lines = [
        "# Manuscript Statistics Source Audit",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 汇总",
        "",
        f"- ok: {int(status_counts.get('ok', 0))}",
        f"- partial: {int(status_counts.get('partial', 0))}",
        f"- unavailable: {int(status_counts.get('unavailable', 0))}",
        "",
        "## Q01-Q27",
        "",
    ]
    task_map = {task.task_id: task for task in TASKS}
    for audit in audit_rows:
        task = task_map[str(audit["task_id"])]
        lines.append(f"### {task.task_id} {task.fig.upper()}{task.panel}")
        lines.append("")
        lines.append(f"- 结论：{task.claim}")
        lines.append(f"- 状态：{_report_status(str(audit['status']))}；usable seeds: {audit['usable_seeds'] or 'none'}")
        lines.append(f"- 来源位置：{audit['source_files']}")
        lines.append(f"- 如何计算：{task.how}")
        reason = audit.get("reason", "")
        if reason:
            lines.append(f"- 审计说明：{reason}")
        caveat = task.caveat
        if caveat:
            lines.append(f"- 主要 caveat：{caveat}")
        stats_for_task = by_task_stats.get(task.task_id, [])
        if stats_for_task:
            preview = stats_for_task[:6]
            lines.append("- 已生成统计行示例：")
            for row in preview:
                contrast = row.get("condition_a", "")
                if row.get("condition_b"):
                    contrast = f"{row.get('condition_a')} - {row.get('condition_b')}"
                lines.append(
                    f"  - {row.get('metric')} {row.get('group')}: n={row.get('n_networks')}, "
                    f"effect/mean={row.get('effect') or row.get('mean')}, p={row.get('p_value') or 'NA'}"
                    + (f", contrast={contrast}" if contrast else "")
                )
            if len(stats_for_task) > len(preview):
                lines.append(f"  - ... additional rows: {len(stats_for_task) - len(preview)}")
        lines.append("")
    partial = [row for row in audit_rows if row.get("status") != "ok"]
    lines.extend(["## Partial Or Unavailable", ""])
    if not partial:
        lines.append("- None")
    else:
        for row in partial:
            lines.append(f"- {row['task_id']}: {row['status']} - {row['reason']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_status(status: str) -> str:
    return {"ok": "computable", "partial": "partially computable", "unavailable": "unavailable"}.get(status, status)


def _write_report(path: Path, audit_rows: Sequence[Mapping[str, str]], stat_rows: Sequence[Mapping[str, str]]) -> None:
    by_task_stats: dict[str, list[Mapping[str, str]]] = {}
    for row in stat_rows:
        by_task_stats.setdefault(str(row["task_id"]), []).append(row)
    status_counts = pd.Series([row["status"] for row in audit_rows]).value_counts().to_dict()
    lines = [
        "# Manuscript Statistics Source Audit",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- ok: {int(status_counts.get('ok', 0))}",
        f"- partial: {int(status_counts.get('partial', 0))}",
        f"- unavailable: {int(status_counts.get('unavailable', 0))}",
        "",
        "## Q01-Q27",
        "",
    ]
    task_map = {task.task_id: task for task in TASKS}
    for audit in audit_rows:
        task = task_map[str(audit["task_id"])]
        lines.append(f"### {task.task_id} {task.fig.upper()}{task.panel}")
        lines.append("")
        lines.append(f"- Claim: {task.claim}")
        lines.append(f"- Status: {_report_status(str(audit['status']))}; usable seeds: {audit['usable_seeds'] or 'none'}")
        lines.append(f"- Source files: {audit['source_files']}")
        lines.append(f"- Calculation: {task.how}")
        reason = audit.get("reason", "")
        if reason:
            lines.append(f"- Audit note: {reason}")
        caveat = task.caveat
        if caveat:
            lines.append(f"- Caveat: {caveat}")
        stats_for_task = by_task_stats.get(task.task_id, [])
        if stats_for_task:
            preview = stats_for_task[:6]
            lines.append("- Generated statistic rows:")
            for row in preview:
                contrast = row.get("condition_a", "")
                if row.get("condition_b"):
                    contrast = f"{row.get('condition_a')} - {row.get('condition_b')}"
                lines.append(
                    f"  - {row.get('metric')} {row.get('group')}: n={row.get('n_networks')}, "
                    f"effect/mean={row.get('effect') or row.get('mean')}, p={row.get('p_value') or 'NA'}"
                    + (f", contrast={contrast}" if contrast else "")
                )
            if len(stats_for_task) > len(preview):
                lines.append(f"  - ... additional rows: {len(stats_for_task) - len(preview)}")
        lines.append("")
    partial = [row for row in audit_rows if row.get("status") != "ok"]
    lines.extend(["## Partial Or Unavailable", ""])
    if not partial:
        lines.append("- None")
    else:
        for row in partial:
            lines.append(f"- {row['task_id']}: {row['status']} - {row['reason']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_expected_seeds(text: str) -> tuple[int, ...]:
    text = str(text).strip()
    match = re.fullmatch(r"(\d+)\.\.(\d+)", text)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return tuple(range(start, end + 1))
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("--expected-seeds cannot be empty")
    return tuple(values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute manuscript statistics directly from paper-figure seed bundles.")
    parser.add_argument("--paper-fig-root", default="results/paper_figure_multi_seed")
    parser.add_argument("--output-dir", default="results/paper_figure_multi_seed/statistics")
    parser.add_argument("--expected-seeds", default="1000..1019")
    parser.add_argument("--dry-run", action="store_true", help="Write audit/report only; skip statistics calculations.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(args.paper_fig_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    expected_seeds = _parse_expected_seeds(args.expected_seeds)
    if not root.is_dir():
        raise FileNotFoundError(f"Paper-figure root does not exist: {root}")

    audit_rows = [_audit_task(root, task, expected_seeds) for task in TASKS]
    stat_rows: list[dict[str, str]] = []
    if not args.dry_run:
        for task, audit in zip(TASKS, audit_rows):
            if audit["calculable"] != "true":
                continue
            calculator = CALCULATORS[task.calculator]
            try:
                stat_rows.extend(calculator(root, task, expected_seeds))
            except Exception as exc:
                audit["status"] = "unavailable"
                audit["calculable"] = "false"
                audit["reason"] = f"Calculation failed: {exc}"

    _apply_fdr(stat_rows)
    _write_csv(output_dir / "manuscript_stats_audit.csv", audit_rows, AUDIT_FIELDS)
    if not args.dry_run:
        _write_csv(output_dir / "manuscript_stats_long.csv", stat_rows, LONG_FIELDS)
    else:
        _write_csv(output_dir / "manuscript_stats_long.csv", [], LONG_FIELDS)
    _write_json_summary(
        output_dir / "manuscript_stats_summary.json",
        paper_fig_root=root,
        audit_rows=audit_rows,
        stat_rows=stat_rows if not args.dry_run else [],
    )
    _write_report(output_dir / "manuscript_stats_report.md", audit_rows, stat_rows)
    print(f"Wrote {output_dir / 'manuscript_stats_audit.csv'}")
    print(f"Wrote {output_dir / 'manuscript_stats_long.csv'}")
    print(f"Wrote {output_dir / 'manuscript_stats_summary.json'}")
    print(f"Wrote {output_dir / 'manuscript_stats_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
