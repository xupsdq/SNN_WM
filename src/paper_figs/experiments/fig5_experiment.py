from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.distractor_chunk_holistic_invocation_experiment import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DELAY1_MS,
    DEFAULT_DELAY2_MS,
    DEFAULT_DILATION_RADIUS,
    DEFAULT_DISTRACTOR_MS,
    DEFAULT_FOREGROUND_THRESHOLD,
    DEFAULT_MAX_PROBES,
    DEFAULT_MAX_TRIPLETS,
    DEFAULT_NUM_SIM_BINS,
    DEFAULT_PROBE_MS,
    DEFAULT_REDISTRIBUTION_FRACTION,
    DEFAULT_SAMPLE_MS,
    DEFAULT_SAMPLES_PER_PROBE,
    DEFAULT_TIE_THRESHOLD,
    DEFAULT_WINNER_WINDOW_FRAC,
    ExperimentConfig as ChunkExperimentConfig,
    SMOKE_COMMAND,
    SMOKE_NOTE,
    build_fig5_fusion_summary,
    run_fig5_fusion_backbone_from_config,
)
from src.paper_figs.common.io import prepare_layout, save_csv, save_json, save_npz, write_artifact_manifest
from src.paper_figs.common.model_env import load_mnist_skeleton_dataset
from src.paper_figs.common.runtime import (
    build_common_parser,
    format_smoke_command,
    resolve_device_strict,
    seed_everything,
    setup_logger,
)

FIGURE_ID = "fig5"
MODULE_NAME = "src.paper_figs.experiments.fig5_experiment"
DEFAULT_OUTPUT_DIR = str(Path("results") / "paper_figs" / FIGURE_ID)


def build_argparser():
    return build_common_parser(
        description="Fig5 paper experiment: fused latent memory form and fused latent memory formation.",
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )


def _foreground_mask(image_tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    if image.ndim == 3:
        image = image[0]
    return (image > 0).astype(np.uint8)


def _build_chunk_config(args, layout) -> ChunkExperimentConfig:
    batch_size = int(DEFAULT_BATCH_SIZE)
    max_probes = int(DEFAULT_MAX_PROBES)
    samples_per_probe = int(DEFAULT_SAMPLES_PER_PROBE)
    max_triplets = int(DEFAULT_MAX_TRIPLETS)
    if bool(args.smoke):
        batch_size = min(batch_size, 2)
        max_probes = min(max_probes, 2)
        samples_per_probe = min(samples_per_probe, 1)
        max_triplets = min(max_triplets, 4)
    return ChunkExperimentConfig(
        model_path=str(args.model_path),
        config=None,
        dataset_root=str(args.dataset_root),
        split="test",
        device=str(args.device),
        seed=int(args.seed),
        output_dir=str(layout.staging_path("fig5_fusion_backbone")),
        sample_ms=float(DEFAULT_SAMPLE_MS),
        delay1_ms=float(DEFAULT_DELAY1_MS),
        distractor_ms=float(DEFAULT_DISTRACTOR_MS),
        delay2_ms=float(DEFAULT_DELAY2_MS),
        probe_ms=float(DEFAULT_PROBE_MS),
        batch_size=batch_size,
        max_probes=max_probes,
        samples_per_probe=samples_per_probe,
        max_triplets=max_triplets,
        num_sim_bins=int(DEFAULT_NUM_SIM_BINS),
        foreground_threshold=float(DEFAULT_FOREGROUND_THRESHOLD),
        dilation_radius=int(DEFAULT_DILATION_RADIUS),
        winner_window_frac=float(DEFAULT_WINNER_WINDOW_FRAC),
        tie_threshold=float(DEFAULT_TIE_THRESHOLD),
        redistribution_fraction=float(DEFAULT_REDISTRIBUTION_FRACTION),
        skip_figures=True,
        smoke=bool(args.smoke),
    )


def _build_example_regions_npz(df_triplets: pd.DataFrame, dataset, example_triplet_id: int) -> dict[str, np.ndarray]:
    triplet = df_triplets.loc[df_triplets["triplet_id"] == int(example_triplet_id)].iloc[0]
    sample_mask = _foreground_mask(dataset[int(triplet["sample_id"])][0])
    distractor_mask = _foreground_mask(dataset[int(triplet["distractor_id"])][0])
    probe_mask = _foreground_mask(dataset[int(triplet["probe_id"])][0])
    shared_mask = (sample_mask & distractor_mask).astype(np.uint8)
    sample_only_mask = (sample_mask & (1 - distractor_mask)).astype(np.uint8)
    distractor_only_mask = (distractor_mask & (1 - sample_mask)).astype(np.uint8)
    return {
        "triplet_id": np.asarray([int(triplet["triplet_id"])], dtype=np.int64),
        "sample_id": np.asarray([int(triplet["sample_id"])], dtype=np.int64),
        "distractor_id": np.asarray([int(triplet["distractor_id"])], dtype=np.int64),
        "probe_id": np.asarray([int(triplet["probe_id"])], dtype=np.int64),
        "sample_mask": sample_mask,
        "distractor_mask": distractor_mask,
        "probe_mask": probe_mask,
        "sample_only_mask": sample_only_mask,
        "distractor_only_mask": distractor_only_mask,
        "shared_mask": shared_mask,
    }


def _bridge_lookup(df_bridge: pd.DataFrame, *, x_name: str, y_name: str) -> dict[str, float | int | None]:
    subset = df_bridge[
        (df_bridge["analysis"] == "correlation")
        & (df_bridge["x"] == str(x_name))
        & (df_bridge["y"] == str(y_name))
    ]
    if subset.empty:
        return {"pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None, "n": 0}
    row = subset.iloc[0]
    return {
        "pearson_r": float(row["pearson_r"]) if pd.notna(row["pearson_r"]) else None,
        "pearson_p": float(row["pearson_p"]) if pd.notna(row["pearson_p"]) else None,
        "spearman_rho": float(row["spearman_rho"]) if pd.notna(row["spearman_rho"]) else None,
        "spearman_p": float(row["spearman_p"]) if pd.notna(row["spearman_p"]) else None,
        "n": int(row["n"]) if pd.notna(row["n"]) else 0,
    }


def build_fig5_summary(
    df_fusion: pd.DataFrame,
    df_specificity: pd.DataFrame,
    df_pull_summary: pd.DataFrame,
    df_bridge: pd.DataFrame,
    df_intervention: pd.DataFrame,
    *,
    smoke: bool,
    smoke_command: str,
) -> dict[str, object]:
    return {
        "figure": FIGURE_ID,
        "panel_b_fusion_form": {
            "mean_sim_to_sample_L3": float(df_fusion["sim_to_sample_L3"].mean()),
            "mean_sim_to_distractor_L3": float(df_fusion["sim_to_distractor_L3"].mean()),
            "mean_fusion_dual_score_L3": float(df_fusion["fusion_dual_score_L3"].mean()),
            "mean_fusion_imbalance_L3": float(df_fusion["fusion_imbalance_L3"].mean()),
        },
        "panel_b_specificity": {
            "mean_true_pair_percentile_L3": float(df_specificity["true_pair_percentile_L3"].mean()),
            "mean_true_pair_z_L3": float(df_specificity["true_pair_z_L3"].mean()),
            "top1_rate_L3": float(df_specificity["true_pair_top1_L3"].mean()),
        },
        "panel_c_rewriting": {
            "mean_barP_L2": float(df_pull_summary["barP_L2"].mean()),
            "mean_barP_L3": float(df_pull_summary["barP_L3"].mean()),
            "mean_peakP_L3": float(df_pull_summary["peakP_L3"].mean()),
            "mean_earlyP_L3": float(df_pull_summary["earlyP_L3"].mean()),
        },
        "panel_d_bridge": {
            "barP_L3_to_fusion_dual_score_L3": _bridge_lookup(
                df_bridge,
                x_name="barP_L3",
                y_name="fusion_dual_score_L3",
            ),
            "barP_L3_to_true_pair_z_L3": _bridge_lookup(
                df_bridge,
                x_name="barP_L3",
                y_name="true_pair_z_L3",
            ),
        },
        "panel_e_intervention": {
            "mean_delta_barP_L3": float(df_intervention["delta_barP_L3"].mean()),
            "mean_delta_fusion_dual_score_L3": float(df_intervention["delta_fusion_dual_score_L3"].mean()),
            "mean_delta_true_pair_z_L3": float(df_intervention["delta_true_pair_z_L3"].mean()),
        },
        "smoke": {
            "enabled": bool(smoke),
            "command": str(smoke_command),
            "note": SMOKE_NOTE,
        },
    }


def _save_main_figure(
    *,
    layout,
    example_regions: dict[str, np.ndarray],
    df_fusion: pd.DataFrame,
    df_specificity: pd.DataFrame,
    df_pull_timeseries: pd.DataFrame,
    df_pull_summary: pd.DataFrame,
    df_bridge: pd.DataFrame,
    df_intervention: pd.DataFrame,
) -> dict[str, str]:
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 13.0))

    region_map = (
        example_regions["sample_only_mask"].astype(np.int32)
        + 2 * example_regions["distractor_only_mask"].astype(np.int32)
        + 3 * example_regions["shared_mask"].astype(np.int32)
    )
    axes[0, 0].imshow(region_map, cmap="viridis", interpolation="nearest")
    axes[0, 0].set_title("A: triplet definition")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].text(
        0.02,
        -0.12,
        f"S={int(example_regions['sample_id'][0])} D={int(example_regions['distractor_id'][0])} P={int(example_regions['probe_id'][0])}",
        transform=axes[0, 0].transAxes,
        fontsize=9,
    )

    axes[0, 1].scatter(
        df_fusion["sim_to_sample_L3"],
        df_fusion["sim_to_distractor_L3"],
        s=26,
        alpha=0.72,
        color="#d95f02",
    )
    axes[0, 1].plot([-1, 1], [-1, 1], linestyle="--", linewidth=1.0, color="#1b9e77")
    axes[0, 1].set_xlabel("sim_to_sample_L3")
    axes[0, 1].set_ylabel("sim_to_distractor_L3")
    axes[0, 1].set_title("B: fusion form")

    layer_colors = {"layer2": "#1f77b4", "layer3": "#d62728"}
    for layer_name in ("layer2", "layer3"):
        layer_df = df_pull_timeseries[df_pull_timeseries["layer"] == layer_name].copy()
        if layer_df.empty:
            continue
        mean_df = layer_df.groupby("distractor_step", as_index=False)["pull_t"].mean()
        axes[1, 0].plot(
            mean_df["distractor_step"],
            mean_df["pull_t"],
            linewidth=1.7,
            label=layer_name,
            color=layer_colors[layer_name],
        )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].set_xlabel("Distractor step")
    axes[1, 0].set_ylabel("pull_t")
    axes[1, 0].set_title("C: grouped-pattern rewriting")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(
        df_pull_summary["barP_L3"],
        df_fusion["fusion_dual_score_L3"],
        s=26,
        alpha=0.72,
        color="#7570b3",
    )
    axes[1, 1].set_xlabel("barP_L3")
    axes[1, 1].set_ylabel("fusion_dual_score_L3")
    axes[1, 1].set_title("D: rewriting to fusion")

    axes[2, 0].hist(
        df_specificity["true_pair_percentile_L3"].dropna().to_numpy(dtype=np.float64),
        bins=np.linspace(0.0, 1.0, 16),
        color="#66a61e",
        alpha=0.75,
    )
    axes[2, 0].set_xlabel("true_pair_percentile_L3")
    axes[2, 0].set_ylabel("count")
    axes[2, 0].set_title("B: fusion specificity")

    metrics = [
        float(df_intervention["delta_barP_L3"].mean()),
        float(df_intervention["delta_fusion_dual_score_L3"].mean()),
        float(df_intervention["delta_true_pair_z_L3"].mean()),
    ]
    axes[2, 1].bar(
        np.arange(3),
        metrics,
        color=["#e7298a", "#1b9e77", "#e6ab02"],
        alpha=0.82,
    )
    axes[2, 1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[2, 1].set_xticks(np.arange(3))
    axes[2, 1].set_xticklabels(["barP_L3", "fusion_dual", "true_pair_z"], rotation=12)
    axes[2, 1].set_ylabel("delta")
    axes[2, 1].set_title("E: formation intervention")

    fig.tight_layout()
    png_path = layout.root_file("fig5_main.png")
    pdf_path = layout.root_file("fig5_main.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "main_figure_png": str(png_path),
        "main_figure_pdf": str(pdf_path),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    seed_everything(int(args.seed))
    device = resolve_device_strict(args.device)
    layout = prepare_layout(args.output_dir)
    logger = setup_logger(layout.log_file(), f"paper_{FIGURE_ID}")
    smoke_command = format_smoke_command(MODULE_NAME, layout.root)

    logger.info("[Init] figure=%s", FIGURE_ID)
    logger.info("[Init] output_dir=%s", layout.root)
    logger.info("[Init] device=%s", device)
    logger.info("[Init] smoke=%s", bool(args.smoke))
    logger.info("[Init] smoke_note=%s", SMOKE_NOTE)

    backbone_config = _build_chunk_config(args, layout)
    backbone = run_fig5_fusion_backbone_from_config(backbone_config, device=device, logger=logger)
    dataset = load_mnist_skeleton_dataset(args.dataset_root, split="test")

    df_triplets = backbone.triplets.copy()
    df_fusion = backbone.preprobe_fusion_metrics.copy()
    df_specificity = backbone.fusion_specificity_metrics.copy()
    df_rewriting_timeseries = backbone.sample_induced_rewriting_timeseries.copy()
    df_rewriting_summary = backbone.sample_induced_rewriting_summary.copy()
    df_pull_timeseries = backbone.distractor_pull_timeseries.copy()
    df_pull_summary = backbone.distractor_pull_summary.copy()
    df_bridge = backbone.rewriting_fusion_bridge.copy()
    df_intervention = backbone.formation_intervention_metrics.copy()

    example_regions = _build_example_regions_npz(df_triplets, dataset, int(backbone.example_triplet_id))

    artifact_paths = {
        "run_config_json": str(
            save_json(
                {
                    "figure": FIGURE_ID,
                    "module_name": MODULE_NAME,
                    "model_path": str(Path(args.model_path).resolve()),
                    "dataset_root": str(Path(args.dataset_root).resolve()),
                    "device_requested": str(args.device),
                    "device_resolved": str(device),
                    "seed": int(args.seed),
                    "smoke": bool(args.smoke),
                    "smoke_command": smoke_command,
                    "smoke_note": SMOKE_NOTE,
                    "backbone_config": backbone.config,
                    "backbone_stats": backbone.stats,
                },
                layout.root_file("run_config.json"),
            )
        ),
        "panel_a_triplet_definition_csv": str(save_csv(df_triplets, layout.data_file("panel_a_triplet_definition.csv"), sort_by=["triplet_id"])),
        "panel_b_preprobe_fusion_metrics_csv": str(save_csv(df_fusion, layout.data_file("panel_b_preprobe_fusion_metrics.csv"), sort_by=["triplet_id"])),
        "panel_b_fusion_specificity_csv": str(save_csv(df_specificity, layout.data_file("panel_b_fusion_specificity.csv"), sort_by=["triplet_id"])),
        "panel_c_sample_induced_rewriting_timeseries_csv": str(save_csv(df_rewriting_timeseries, layout.data_file("panel_c_sample_induced_rewriting_timeseries.csv"), sort_by=["triplet_id", "layer", "distractor_step"])),
        "panel_c_sample_induced_rewriting_summary_csv": str(save_csv(df_rewriting_summary, layout.data_file("panel_c_sample_induced_rewriting_summary.csv"), sort_by=["triplet_id"])),
        "panel_c_distractor_pull_timeseries_csv": str(save_csv(df_pull_timeseries, layout.data_file("panel_c_distractor_pull_timeseries.csv"), sort_by=["triplet_id", "layer", "distractor_step"])),
        "panel_c_distractor_pull_summary_csv": str(save_csv(df_pull_summary, layout.data_file("panel_c_distractor_pull_summary.csv"), sort_by=["triplet_id"])),
        "panel_d_rewriting_to_fusion_bridge_csv": str(save_csv(df_bridge, layout.data_file("panel_d_rewriting_to_fusion_bridge.csv"))),
        "panel_e_formation_intervention_csv": str(save_csv(df_intervention, layout.data_file("panel_e_formation_intervention.csv"), sort_by=["triplet_id"])),
        "panel_e_formation_intervention_comparison_csv": str(save_csv(df_intervention, layout.data_file("panel_e_formation_intervention_comparison.csv"), sort_by=["triplet_id"])),
        "panel_a_example_regions_npz": str(save_npz(layout.array_file("panel_a_example_regions.npz"), **example_regions)),
        "panel_b_example_preprobe_fusion_state_npz": str(save_npz(layout.array_file("panel_b_example_preprobe_fusion_state.npz"), **backbone.example_preprobe_fusion_state)),
        "panel_c_example_distractor_pull_trace_npz": str(save_npz(layout.array_file("panel_c_example_distractor_pull_trace.npz"), **backbone.example_distractor_pull_trace)),
        "supp_layer1_region_support_trial_csv": str(save_csv(backbone.region_support_condition, layout.data_file("supp_layer1_region_support_trial.csv"), sort_by=["triplet_id", "condition", "layer", "region"])),
        "supp_layer1_predicted_vs_observed_csv": str(save_csv(backbone.layer1_formula_fit, layout.data_file("supp_layer1_predicted_vs_observed.csv"))),
    }

    figure_paths = _save_main_figure(
        layout=layout,
        example_regions=example_regions,
        df_fusion=df_fusion,
        df_specificity=df_specificity,
        df_pull_timeseries=df_pull_timeseries,
        df_pull_summary=df_pull_summary,
        df_bridge=df_bridge,
        df_intervention=df_intervention,
    )
    artifact_paths.update(figure_paths)

    summary = build_fig5_summary(
        df_fusion,
        df_specificity,
        df_pull_summary,
        df_bridge,
        df_intervention,
        smoke=bool(args.smoke),
        smoke_command=smoke_command,
    )
    summary["backbone_summary"] = build_fig5_fusion_summary(
        triplets=df_triplets,
        preprobe_fusion_metrics=df_fusion,
        fusion_specificity_metrics=df_specificity,
        distractor_pull_summary=df_pull_summary,
        rewriting_fusion_bridge=df_bridge,
        formation_intervention_metrics=df_intervention,
        smoke=bool(args.smoke),
    )
    summary["saved_artifacts"] = artifact_paths
    summary["supplement_note"] = (
        "Layer1 region support and predicted-vs-observed exports are preserved as supplementary files only. "
        "The main Fig.5 wrapper is now organized around fusion form, specificity, rewriting, bridge, and formation intervention."
    )
    artifact_paths["summary_json"] = str(save_json(summary, layout.root_file("summary.json")))
    artifact_paths["artifact_manifest_json"] = str(write_artifact_manifest(layout, artifact_paths))

    logger.info("[Done] Fig5 artifacts saved.")
    logger.info("[Done] smoke_note=%s", SMOKE_NOTE)


if __name__ == "__main__":
    main()
