from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt

from figure_utils_common import save_figure_all_formats
from paper_plot_style import apply_paper_style
from src.plotting.fig2_silent_memory_effect import plot_memory_effect_vs_delay
from src.plotting.final_panels import (
    add_panel_label,
    draw_example_distractor_panel,
    draw_example_dms_panel,
    draw_example_ping_panel,
    draw_example_shuffle_panel,
    plot_baseline_panels,
    plot_causal_bias_only_panel,
    plot_causal_condition_panel,
    plot_causal_error_destination_panel,
    plot_distractor_accuracy_only_panel,
    plot_distractor_sensory_panel,
    plot_engram_panel,
    plot_external_input_overlay_panel,
    plot_input_similarity_dynamic_only_panel,
    plot_input_similarity_panel,
    plot_ping_first_fire_accuracy_panel,
    plot_representative_silent_memory_panel,
    plot_sample_survival_panel,
    plot_triplet_confidence_only,
    plot_triplet_summary_only,
    plot_ux_shuffle_accuracy_panel,
)
from src.plotting.results_registry import (
    PanelContext,
    default_dataset_root,
    default_model_path,
    default_results_root,
    load_required_csv,
    load_required_json,
)


def build_figure1(
    *,
    results_root: Path | str | None = None,
    model_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> plt.Figure:
    results_root = Path(default_results_root() if results_root is None else results_root).resolve()
    model_path = default_model_path() if model_path is None else model_path
    dataset_root = default_dataset_root() if dataset_root is None else dataset_root

    baseline_context = PanelContext("Fig1", "A/B", "baseline_processing")
    silent_context = PanelContext("Fig1", "C", "silent_memory_effect")
    df_confusion = load_required_csv(
        results_root,
        baseline_context,
        "metrics_confusion_matrix.csv",
        ["true_label", "pred_label", "count", "fraction_row_normalized"],
        model_path=model_path,
        dataset_root=dataset_root,
    )
    df_recall = load_required_csv(
        results_root,
        baseline_context,
        "metrics_class_recall.csv",
        ["label", "recall", "n_trials", "overall_accuracy"],
        model_path=model_path,
        dataset_root=dataset_root,
    )
    run_config = load_required_json(
        results_root,
        baseline_context,
        "run_config.json",
        ["num_classes", "metrics"],
        model_path=model_path,
        dataset_root=dataset_root,
    )
    df_memory = load_required_csv(
        results_root,
        silent_context,
        "metrics_memory_effect_vs_delay.csv",
        [
            "delay_ms",
            "stsp_mode",
            "memory_effect_mean",
            "memory_effect_ci95_lower",
            "memory_effect_ci95_upper",
            "fit_y",
            "fit_tau",
        ],
        model_path=model_path,
    )

    fig = plt.figure(figsize=(13.2, 10.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.42, wspace=0.38)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    plot_baseline_panels(
        ax_a,
        ax_b,
        df_confusion,
        df_recall,
        n_trials=int(run_config["metrics"]["n_trials"]),
        num_classes=int(run_config["num_classes"]),
    )
    plot_memory_effect_vs_delay(ax_c, df_memory)
    ax_a.set_box_aspect(1.0)
    ax_b.set_box_aspect(1.0)
    c_label_x = (ax_a.get_position().x0 - 0.14 * ax_a.get_position().width - ax_c.get_position().x0) / ax_c.get_position().width
    add_panel_label(ax_a, "A")
    add_panel_label(ax_b, "B")
    add_panel_label(ax_c, "C", x=c_label_x)
    _clear_all_axis_titles(fig)
    return fig


def build_figure2(
    *,
    results_root: Path | str | None = None,
    model_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> plt.Figure:
    del dataset_root
    results_root = Path(default_results_root() if results_root is None else results_root).resolve()
    model_path = default_model_path() if model_path is None else model_path
    silent_context = PanelContext("Fig2", "B", "silent_memory_effect")
    engram_context = PanelContext("Fig2", "C", "engram_decode")
    triplet_summary_context = PanelContext("Fig2", "D", "silent_substrate_triplet")
    triplet_conf_context = PanelContext("Fig2", "E", "silent_substrate_triplet")

    df_raster = load_required_csv(
        results_root,
        silent_context,
        "representative_raster_points.csv",
        ["trial_id", "stsp_mode", "delay_ms", "layer", "t_step", "neuron_index", "phase"],
        model_path=model_path,
    )
    df_population = load_required_csv(
        results_root,
        silent_context,
        "representative_population_rate.csv",
        [
            "trial_id",
            "stsp_mode",
            "delay_ms",
            "t_step",
            "time_ms",
            "phase",
            "pooled_spike_count",
            "pooled_rate_spikes_per_neuron_step",
            "total_neurons",
        ],
        model_path=model_path,
    )
    df_engram = load_required_csv(
        results_root,
        engram_context,
        "engram_decode_metrics.csv",
        ["layer", "delay_ms", "acc", "acc_ci_low", "acc_ci_high"],
        ensure_materialized=False,
    )
    df_triplet_summary = load_required_csv(
        results_root,
        triplet_summary_context,
        "metrics_decode_window_summary.csv",
        ["layer", "substrate", "window_name", "decode_acc", "ci95_lower", "ci95_upper", "perm_p"],
        ensure_materialized=False,
    )
    df_triplet_conf = load_required_csv(
        results_root,
        triplet_conf_context,
        "trial_level_decoder_confidence.csv",
        ["trial_id", "layer", "substrate", "eval_bin_start_ms", "pred_label", "true_label", "confidence_margin"],
        ensure_materialized=False,
    )

    fig = plt.figure(figsize=(12.2, 12.9))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.55, 1.0, 0.95], hspace=0.5, wspace=0.34)
    b_spec = grid[0, :]
    raster_axes, _ = plot_representative_silent_memory_panel(fig, b_spec, df_raster, df_population)
    ax_c = fig.add_subplot(grid[1, 0])
    plot_engram_panel(ax_c, df_engram)
    ax_d = fig.add_subplot(grid[1, 1])
    plot_triplet_confidence_only(ax_d, df_triplet_conf, df_triplet_summary)
    ax_e = fig.add_subplot(grid[2, :])
    plot_triplet_summary_only(ax_e, df_triplet_summary)
    b_label_x = (ax_c.get_position().x0 - 0.14 * ax_c.get_position().width - raster_axes[0].get_position().x0) / raster_axes[0].get_position().width
    add_panel_label(raster_axes[0], "B", x=b_label_x)
    add_panel_label(ax_c, "C")
    add_panel_label(ax_d, "D")
    add_panel_label(ax_e, "E", x=-0.08)
    _clear_all_axis_titles(fig)
    return fig


def build_figure3(
    *,
    results_root: Path | str | None = None,
    model_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> plt.Figure:
    del model_path, dataset_root
    results_root = Path(default_results_root() if results_root is None else results_root).resolve()
    shuffle_context = PanelContext("Fig3", "B", "ux_shuffle_memory_collapse")
    causal_condition_context = PanelContext("Fig3", "D", "causal_substrate_dissociation")
    causal_error_context = PanelContext("Fig3", "C", "causal_substrate_dissociation")

    df_shuffle = load_required_csv(
        results_root,
        shuffle_context,
        "metrics_condition_summary.csv",
        [
            "condition",
            "acc_probe",
            "abs_rate_pred_original_sample",
            "abs_rate_pred_change_under_bmap",
        ],
        ensure_materialized=False,
    )
    df_condition = load_required_csv(
        results_root,
        causal_condition_context,
        "metrics_condition_summary.csv",
        ["condition", "acc_probe", "sample_related_bias"],
        ensure_materialized=False,
    )
    df_error = load_required_csv(
        results_root,
        causal_error_context,
        "metrics_error_destination.csv",
        ["condition", "destination", "rate_percent"],
        ensure_materialized=False,
    )

    fig = plt.figure(figsize=(12.4, 9.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], hspace=0.52, wspace=0.38)
    ax_b = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    ax_d = fig.add_subplot(grid[1, :])

    plot_ux_shuffle_accuracy_panel(ax_b, df_shuffle)
    plot_causal_bias_only_panel(ax_c, df_condition)
    plot_causal_error_destination_panel(ax_d, df_error, legend_outside=True)

    d_label_x = (ax_b.get_position().x0 - 0.14 * ax_b.get_position().width - ax_d.get_position().x0) / ax_d.get_position().width
    add_panel_label(ax_b, "B")
    add_panel_label(ax_c, "C")
    add_panel_label(ax_d, "D", x=d_label_x)
    _clear_all_axis_titles(fig)
    return fig


def build_figure4(
    *,
    results_root: Path | str | None = None,
    model_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> plt.Figure:
    results_root = Path(default_results_root() if results_root is None else results_root).resolve()
    model_path = default_model_path() if model_path is None else model_path
    dataset_root = default_dataset_root() if dataset_root is None else dataset_root

    source_context = PanelContext("Fig4", "B", "external_input_interrogation")
    ping_context = PanelContext("Fig4", "C", "ping_impulse_readout")
    df_ping_summary = load_required_csv(
        results_root,
        source_context,
        "metrics_ping_summary.csv",
        [
            "ping_target_label",
            "activation_target",
            "achieved_activation_frac",
            "delta_decode_ping_acc_mean",
            "delta_decode_ping_acc_ci95_lower",
            "delta_decode_ping_acc_ci95_upper",
        ],
        model_path=model_path,
        dataset_root=dataset_root,
    )
    df_ping_selectivity = load_required_csv(
        results_root,
        source_context,
        "metrics_ping_selectivity.csv",
        [
            "ping_target_label",
            "activation_target",
            "achieved_activation_frac",
            "sample_aligned_selectivity_mean",
            "sample_aligned_selectivity_ci95_lower",
            "sample_aligned_selectivity_ci95_upper",
        ],
        model_path=model_path,
        dataset_root=dataset_root,
    )
    df_first_fire_summary = load_required_csv(
        results_root,
        ping_context,
        "metrics_condition_summary.csv",
        ["condition", "first_fire_ping_acc_mean"],
        ensure_materialized=False,
    )
    df_cross_seed = load_required_csv(
        results_root,
        ping_context,
        "cross_seed_tests.csv",
        ["metric", "condition_a", "condition_b", "n_pairs", "obs_mean_diff_a_minus_b", "p_one_sided_a_gt_b"],
        ensure_materialized=False,
    )

    fig = plt.figure(figsize=(12.2, 5.6))
    grid = fig.add_gridspec(1, 2, wspace=0.42)
    ax_b = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    plot_external_input_overlay_panel(ax_b, df_ping_summary, df_ping_selectivity)
    plot_ping_first_fire_accuracy_panel(ax_c, df_first_fire_summary, df_cross_seed)
    for label, axis in [("B", ax_b), ("C", ax_c)]:
        add_panel_label(axis, label)
    _clear_all_axis_titles(fig)
    return fig


def build_figure5(
    *,
    results_root: Path | str | None = None,
    model_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> plt.Figure:
    del model_path, dataset_root
    results_root = Path(default_results_root() if results_root is None else results_root).resolve()
    distractor_context = PanelContext("Fig5", "B", "dual_task_retention")
    survival_context = PanelContext("Fig5", "C", "dual_task_retention")
    similarity_context = PanelContext("Fig5", "D", "dual_task_similarity_boundary")

    df_metrics_dist = load_required_csv(
        results_root,
        distractor_context,
        "metrics_distractor_sensory_readout.csv",
        [
            "acc_distractor_dynamic",
            "acc_distractor_dynamic_ci95_lower",
            "acc_distractor_dynamic_ci95_upper",
            "acc_distractor_static",
            "acc_distractor_static_ci95_lower",
            "acc_distractor_static_ci95_upper",
            "silent_rate_distractor_dynamic",
            "silent_rate_distractor_dynamic_ci95_lower",
            "silent_rate_distractor_dynamic_ci95_upper",
            "silent_rate_distractor_static",
            "silent_rate_distractor_static_ci95_lower",
            "silent_rate_distractor_static_ci95_upper",
            "acc_distractor_gap_dynamic_minus_static",
            "acc_distractor_gap_p_two_sided_ne0",
            "silent_rate_distractor_gap_dynamic_minus_static",
            "silent_rate_distractor_gap_p_two_sided_ne0",
        ],
        ensure_materialized=False,
    )
    df_metrics_survival = load_required_csv(
        results_root,
        survival_context,
        "metrics_sample_readout_survival.csv",
        [
            "clean_sample_bias",
            "clean_sample_bias_ci95_lower",
            "clean_sample_bias_ci95_upper",
            "distracted_sample_bias",
            "distracted_sample_bias_ci95_lower",
            "distracted_sample_bias_ci95_upper",
            "distracted_noise_bias",
            "distracted_noise_bias_ci95_lower",
            "distracted_noise_bias_ci95_upper",
            "sample_bias_retention_pct",
            "p_one_sided_sample_gt_noise_distracted",
        ],
        ensure_materialized=False,
    )
    df_similarity = load_required_csv(
        results_root,
        similarity_context,
        "metrics_input_similarity_summary_pixel.csv",
        [
            "bucket_name",
            "stsp_mode",
            "probe_accuracy",
            "distractor_accuracy",
            "sample_bias_retention_pct",
            "sample_bias_minus_noise",
            "p_one_sided_gt_chance",
        ],
        ensure_materialized=False,
    )

    fig = plt.figure(figsize=(12.5, 10.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.86, 1.34], hspace=0.5, wspace=0.36)
    ax_b = fig.add_subplot(grid[0, 0])
    plot_distractor_accuracy_only_panel(ax_b, df_metrics_dist, star_pvalues=True)
    ax_c = fig.add_subplot(grid[0, 1])
    plot_sample_survival_panel(ax_c, df_metrics_survival, star_pvalues=True)
    d_spec = grid[1, :]
    d_axes = plot_input_similarity_dynamic_only_panel(fig, d_spec, df_similarity)

    add_panel_label(ax_b, "B")
    add_panel_label(ax_c, "C")
    add_panel_label(d_axes[0], "D", y=1.16)
    _clear_all_axis_titles(fig)
    return fig


def save_figure(
    figure_id: str,
    *,
    results_root: Path | str | None = None,
    output_dir: Path | str | None = None,
    model_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
) -> dict[str, str]:
    build_map: dict[str, Callable[..., plt.Figure]] = {
        "figure1": build_figure1,
        "figure2": build_figure2,
        "figure3": build_figure3,
        "figure4": build_figure4,
        "figure5": build_figure5,
    }
    builder = build_map[figure_id]
    fig = builder(results_root=results_root, model_path=model_path, dataset_root=dataset_root)
    out_root = Path(default_results_root() / "paper_figures" if output_dir is None else output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    saved = save_figure_all_formats(fig, out_root / figure_id)
    plt.close(fig)
    return saved


def main_figure1() -> None:
    _main("figure1")


def main_figure2() -> None:
    _main("figure2")


def main_figure3() -> None:
    _main("figure3")


def main_figure4() -> None:
    _main("figure4")


def main_figure5() -> None:
    _main("figure5")


def _build_parser(figure_id: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Build {figure_id} from result records.")
    parser.add_argument("--results-root", type=str, default=str(default_results_root()))
    parser.add_argument("--output-dir", type=str, default=str(default_results_root() / "paper_figures"))
    parser.add_argument("--model-path", type=str, default=str(default_model_path()))
    parser.add_argument("--dataset-root", type=str, default=str(default_dataset_root()))
    return parser


def _main(figure_id: str) -> None:
    args = _build_parser(figure_id).parse_args()
    apply_paper_style()
    saved = save_figure(
        figure_id,
        results_root=args.results_root,
        output_dir=args.output_dir,
        model_path=args.model_path,
        dataset_root=args.dataset_root,
    )
    for ext, path in saved.items():
        print(f"{figure_id}.{ext}: {path}")


def _clear_all_axis_titles(fig: plt.Figure) -> None:
    for axis in fig.axes:
        axis.set_title("")
