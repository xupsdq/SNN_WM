from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting.common.colors import get_plot_cmap, get_plot_color
from src.plotting.common.theme_tokens import COLOR_NEUTRAL, GRID_ALPHA_SOFT
from src.plotting.experiments._common import load_bundle_npz, main_for, optional_bundle_file, read_bundle_csv


PANEL_A_NPZ = "l1_panel_a_preprobe_gain_map.npz"
PANEL_C_NPZ = "l1_local_event_time_alignment.npz"
PANEL_D_CSV = "l1_local_causal_chain_events.csv"


def _require_arrays(payload: dict[str, np.ndarray], names: tuple[str, ...], source: str) -> None:
    missing = [name for name in names if name not in payload]
    if missing:
        raise KeyError(f"{source} missing required arrays: {', '.join(missing)}")


def _require_bundle_artifacts(input_dir: Path) -> None:
    required = (PANEL_A_NPZ, PANEL_C_NPZ, PANEL_D_CSV)
    missing = [name for name in required if optional_bundle_file(input_dir, name) is None]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            "DMS overlap UX support plot requires refreshed computation artifacts. "
            f"Missing: {joined}. Re-run the computation runner for dms_overlap_ux_support_mechanism_experiment."
        )


def _nanmean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array for trace summaries.")
    mean = np.nanmean(arr, axis=0)
    count = np.sum(np.isfinite(arr), axis=0).astype(np.float64)
    std = np.nanstd(arr, axis=0, ddof=1)
    err = np.divide(std, np.sqrt(count), out=np.zeros_like(std), where=count > 1.0)
    err[count <= 1.0] = 0.0
    return mean, err


def _sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def draw_panel_a_support_map_on_ax(ax: plt.Axes, payload: dict[str, np.ndarray]) -> None:
    """Draw the overlap/preprobe support map on an existing axes."""
    _require_arrays(payload, ("sample_mask", "probe_mask", "ux_map_pre_dynamic"), PANEL_A_NPZ)
    sample_mask = np.asarray(payload["sample_mask"], dtype=bool)
    probe_mask = np.asarray(payload["probe_mask"], dtype=bool)
    support = np.asarray(payload["ux_map_pre_dynamic"], dtype=np.float64)
    if support.ndim > 2:
        support = np.squeeze(support)
    if support.shape != sample_mask.shape or support.shape != probe_mask.shape:
        raise ValueError("Panel A masks and ux_map_pre_dynamic must have matching 2D shapes.")

    support_delta = support - float(np.nanmin(support))
    support_delta = np.clip(support_delta, 0.0, None)
    masked_support = np.ma.masked_where((~np.isfinite(support_delta)) | (support_delta <= 1e-12), support_delta)

    ax.set_facecolor("black")
    ax.imshow(np.zeros_like(support_delta), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    cmap = get_plot_cmap("stsp_support").copy()
    cmap.set_bad(alpha=0.0)
    ax.imshow(masked_support, cmap=cmap, interpolation="nearest", alpha=0.92)
    sample_color = get_plot_color("sample_only_region")
    probe_color = get_plot_color("probe_only_region")
    ax.contour(sample_mask.astype(float), levels=[0.5], colors=[sample_color], linewidths=1.8, linestyles="-")
    ax.contour(probe_mask.astype(float), levels=[0.5], colors=[probe_color], linewidths=2.1)
    legend = ax.legend(
        handles=[
            Line2D([0], [0], color=sample_color, linestyle="-", linewidth=1.8, label="sample"),
            Line2D([0], [0], color=probe_color, linestyle="-", linewidth=2.1, label="probe"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        borderaxespad=0.35,
    )
    for text in legend.get_texts():
        text.set_color("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_panel_a(payload: dict[str, np.ndarray]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.0, 4.2))
    draw_panel_a_support_map_on_ax(ax, payload)
    fig.tight_layout()
    return fig


def _transition_counts(df: pd.DataFrame, group: str) -> dict[str, float]:
    sub = df[df["unit_group"].astype(str) == group].copy()
    if sub.empty:
        raise ValueError(f"l1_firing_transition_summary.csv has no rows for unit_group={group!r}")
    if "aggregation_scope" in sub.columns and (sub["aggregation_scope"].astype(str) == "pooled").any():
        row = sub[sub["aggregation_scope"].astype(str) == "pooled"].iloc[0]
        n_units = float(row["n_units"])
        counts = {
            "advanced": float(row["n_advance"]),
            "recruited": float(row["n_recruit"]),
            "lost": float(row["n_loss"]),
        }
        counts["unchanged"] = float(row["n_unchanged"]) if "n_unchanged" in row.index else max(0.0, n_units - sum(counts.values()))
        return counts
    trial_rows = sub[sub["aggregation_scope"].astype(str) == "per_trial"] if "aggregation_scope" in sub.columns else sub
    counts = {
        "advanced": float(pd.to_numeric(trial_rows["n_advance"], errors="coerce").sum()),
        "recruited": float(pd.to_numeric(trial_rows["n_recruit"], errors="coerce").sum()),
        "lost": float(pd.to_numeric(trial_rows["n_loss"], errors="coerce").sum()),
    }
    if "n_unchanged" in trial_rows.columns:
        counts["unchanged"] = float(pd.to_numeric(trial_rows["n_unchanged"], errors="coerce").sum())
    else:
        n_units = float(pd.to_numeric(trial_rows["n_units"], errors="coerce").sum())
        counts["unchanged"] = max(0.0, n_units - sum(counts.values()))
    return counts


def draw_panel_b_early_probe_transitions_on_ax(ax: plt.Axes, df: pd.DataFrame, *, title: str | None = None) -> None:
    """Draw the early-probe transition stacked bars on an existing axes."""
    for column in ("unit_group", "n_units", "n_advance", "n_recruit", "n_loss"):
        if column not in df.columns:
            raise ValueError(f"l1_firing_transition_summary.csv missing required column: {column}")
    groups = [("overlap_dominant", "overlap-dominant"), ("probe_only_dominant", "probe-only-dominant")]
    segments = [
        ("advanced", "advanced", get_plot_color("dynamic")),
        ("recruited", "recruited", get_plot_color("probe_only_region")),
        ("lost", "lost", get_plot_color("non_overlap_control")),
    ]
    y_positions = np.array([1.0, 0.0])
    bar_height = 0.34
    visible_totals: list[float] = []
    for ypos, (group_key, group_label) in zip(y_positions, groups):
        counts = _transition_counts(df, group_key)
        total = max(sum(counts.values()), 1.0)
        left = 0.0
        for seg_key, seg_label, color in segments:
            width = 100.0 * counts[seg_key] / total
            ax.barh(ypos, width, left=left, height=bar_height, color=color, edgecolor="white", linewidth=0.8)
            if width >= 7.0:
                ax.text(left + width / 2.0, ypos, f"{seg_label}\n{width:.0f}%", ha="center", va="center", fontsize=8, color="white" if seg_key != "recruited" else COLOR_NEUTRAL)
            left += width
        visible_totals.append(left)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in groups])
    x_max = min(100.0, max(visible_totals, default=0.0) + 8.0)
    ax.set_xlim(0.0, max(20.0, x_max))
    ax.set_xlabel("Transition fraction (%)")
    if title:
        ax.set_title(title)
    ax.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=5.0, label=label)
            for _, label, color in segments
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
        handlelength=1.6,
    )
    ax.grid(axis="x", alpha=GRID_ALPHA_SOFT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_overlap_vs_probe_only_support_on_ax(ax: plt.Axes, df: pd.DataFrame, *, title: str | None = None) -> None:
    """Draw overlap-aligned versus probe-only support as a paired experiment-style bar plot."""
    required = ("support_region", "pre_probe_stsp_support")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"support comparison missing required columns: {', '.join(missing)}")
    groups = [("Overlap-aligned", get_plot_color("sample_probe_overlap")), ("Probe-only", get_plot_color("probe_only_region"))]
    x = np.arange(len(groups), dtype=float)
    means = []
    errors = []
    for group, _ in groups:
        values = pd.to_numeric(df.loc[df["support_region"].astype(str) == group, "pre_probe_stsp_support"], errors="coerce").dropna().to_numpy(dtype=float)
        means.append(float(values.mean()) if values.size else 0.0)
        errors.append(_sem(values) if values.size else 0.0)
    ax.bar(x, means, yerr=errors, color=[color for _, color in groups], edgecolor=COLOR_NEUTRAL, linewidth=0.7, alpha=0.82, capsize=3)
    if "seed" in df.columns:
        for _, part in df.groupby("seed", dropna=False):
            pts = []
            for group, _ in groups:
                vals = pd.to_numeric(part.loc[part["support_region"].astype(str) == group, "pre_probe_stsp_support"], errors="coerce").dropna()
                pts.append(float(vals.mean()) if not vals.empty else np.nan)
            if np.isfinite(pts).all():
                ax.plot(x, pts, color=COLOR_NEUTRAL, alpha=0.22, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in groups], rotation=18, ha="right")
    ax.set_ylabel("Pre-probe STSP support")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=GRID_ALPHA_SOFT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_panel_b(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    draw_panel_b_early_probe_transitions_on_ax(ax, df)
    fig.tight_layout()
    return fig


def draw_panel_c_winner_loser_event_chain_on_axes(axes: tuple[plt.Axes, plt.Axes] | list[plt.Axes], payload: dict[str, np.ndarray]) -> None:
    """Draw the two-axis winner/loser event-chain traces."""
    _require_arrays(payload, ("relative_time", "winner_delta_v_aligned", "loser_delta_v_aligned", "loser_inh_before_aligned"), PANEL_C_NPZ)
    rel = np.asarray(payload["relative_time"], dtype=np.float64)
    winner_mean, winner_err = _nanmean_sem(np.asarray(payload["winner_delta_v_aligned"], dtype=np.float64))
    loser_mean, loser_err = _nanmean_sem(np.asarray(payload["loser_delta_v_aligned"], dtype=np.float64))
    inh_mean, inh_err = _nanmean_sem(np.asarray(payload["loser_inh_before_aligned"], dtype=np.float64))
    winner_color = get_plot_color("dynamic")
    loser_color = get_plot_color("non_overlap_control")

    ax_top, ax_bottom = axes
    winner_line, = ax_top.plot(rel, 1000.0 * winner_mean, color=winner_color, linewidth=2.0, label="winner")
    ax_top.fill_between(rel, 1000.0 * (winner_mean - winner_err), 1000.0 * (winner_mean + winner_err), color=winner_color, alpha=0.18, linewidth=0)
    loser_line, = ax_top.plot(rel, 1000.0 * loser_mean, color=loser_color, linewidth=2.0, label="loser")
    ax_top.fill_between(rel, 1000.0 * (loser_mean - loser_err), 1000.0 * (loser_mean + loser_err), color=loser_color, alpha=0.16, linewidth=0)
    ax_bottom.plot(rel, 1000.0 * inh_mean, color=loser_color, linewidth=2.0)
    ax_bottom.fill_between(rel, 1000.0 * (inh_mean - inh_err), 1000.0 * (inh_mean + inh_err), color=loser_color, alpha=0.16, linewidth=0)

    for ax in axes:
        ax.axvline(0.0, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color=get_plot_color("other_residual"), linestyle=":", linewidth=0.9)
        ax.grid(alpha=GRID_ALPHA_SOFT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_top.set_ylabel("Delta effective V (mV)")
    ax_bottom.set_ylabel("Loser inhibition (mV)")
    ax_bottom.set_xlabel("Time from winner spike")
    ax_bottom.legend(handles=[winner_line, loser_line], loc="lower right", frameon=False, fontsize=9)
    if rel.size:
        for ax in axes:
            ax.margins(x=0.04)


def _plot_panel_c(payload: dict[str, np.ndarray]) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True, gridspec_kw={"hspace": 0.08})
    draw_panel_c_winner_loser_event_chain_on_axes(axes, payload)
    fig.subplots_adjust(left=0.14, right=0.95, bottom=0.13, top=0.96, hspace=0.10)
    return fig


def draw_panel_d_local_chain_occurrence_on_ax(ax: plt.Axes, df: pd.DataFrame, *, title: str | None = None) -> None:
    """Draw local causal-chain occurrence fractions on an existing axes."""
    metrics = [
        ("winner_pre_spike_boost", "winner\nboost"),
        ("loser_post_winner_suppressed", "loser\nsuppression"),
        ("full_chain_satisfied", "full\nchain"),
    ]
    for column, _ in metrics:
        if column not in df.columns:
            raise ValueError(f"{PANEL_D_CSV} missing required column: {column}")
    y = np.arange(len(metrics), dtype=np.float64)[::-1]
    color = get_plot_color("dynamic")
    for ypos, (column, label) in zip(y, metrics):
        trial_means = df.groupby("trial_id", sort=True)[column].mean() if "trial_id" in df.columns else pd.Series(pd.to_numeric(df[column], errors="coerce"))
        values = pd.to_numeric(trial_means, errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        mean = 100.0 * float(values.mean()) if values.size else 0.0
        err = 100.0 * _sem(values)
        point_color = get_plot_color("peak_region") if column == "full_chain_satisfied" else color
        ax.errorbar(mean, ypos, xerr=err, fmt="o", color=point_color, ecolor=point_color, elinewidth=1.6, capsize=3, markersize=7 if column == "full_chain_satisfied" else 6)
        ax.text(mean + 2.0, ypos, f"{mean:.0f}%", ha="left", va="center", fontsize=8, color=COLOR_NEUTRAL)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in metrics])
    for tick, (column, _) in zip(ax.get_yticklabels(), metrics):
        if column == "full_chain_satisfied":
            tick.set_fontweight("bold")
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Fraction of local events (%)")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", alpha=GRID_ALPHA_SOFT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_panel_d(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    draw_panel_d_local_chain_occurrence_on_ax(ax, df)
    fig.tight_layout()
    return fig


def plot_bundle(input_dir: Path):
    _require_bundle_artifacts(input_dir)
    panel_a_payload = load_bundle_npz(input_dir, PANEL_A_NPZ)
    df_firing = read_bundle_csv(input_dir, "l1_firing_transition_summary.csv")
    panel_c_payload = load_bundle_npz(input_dir, PANEL_C_NPZ)
    df_chain = read_bundle_csv(input_dir, PANEL_D_CSV)
    return {
        "fig4_panel_a_preprobe_stsp_support": _plot_panel_a(panel_a_payload),
        "fig4_panel_b_early_probe_transitions": _plot_panel_b(df_firing),
        "fig4_panel_c_winner_loser_event_chain": _plot_panel_c(panel_c_payload),
        "fig4_panel_d_local_chain_occurrence": _plot_panel_d(df_chain),
    }


if __name__ == "__main__":
    raise SystemExit(
        main_for(
            "dms_overlap_ux_support_mechanism_experiment",
            plot_bundle,
            title="DMS Overlap UX Support Mechanism",
        )
    )
