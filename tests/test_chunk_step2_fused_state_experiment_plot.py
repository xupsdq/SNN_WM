from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.plotting.experiments.chunk_step2_fused_state_experiment_plot import main


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_chunk_step2_plot_only_generates_retained_panels(tmp_path: Path) -> None:
    result_dir = tmp_path / "bundle"
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (result_dir / name).mkdir(parents=True, exist_ok=True)

    _write_csv(
        result_dir / "metrics" / "preprobe_fusion_metrics.csv",
        [
            {
                "triplet_id": 1,
                "layer": "L3",
                "sim_to_sample": 0.5,
                "sim_to_distractor": 0.4,
                "fusion_dual_score": 0.9,
                "fusion_imbalance": 0.1,
            },
            {
                "triplet_id": 2,
                "layer": "L3",
                "sim_to_sample": 0.6,
                "sim_to_distractor": 0.35,
                "fusion_dual_score": 1.0,
                "fusion_imbalance": 0.05,
            },
        ],
    )
    _write_csv(
        result_dir / "metrics" / "fusion_specificity_metrics.csv",
        [
            {
                "triplet_id": 1,
                "layer": "L3",
                "true_pair_score": 0.8,
                "true_pair_percentile": 0.9,
                "true_pair_z": 1.6,
                "true_pair_top1": 1,
                "shuffled_pair_score": 0.3,
            },
            {
                "triplet_id": 2,
                "layer": "L3",
                "true_pair_score": 0.7,
                "true_pair_percentile": 0.85,
                "true_pair_z": 1.2,
                "true_pair_top1": 0,
                "shuffled_pair_score": 0.35,
            },
        ],
    )
    _write_csv(
        result_dir / "metrics" / "whole_over_part_metrics.csv",
        [
            {
                "triplet_id": 1,
                "layer": "L3",
                "sim_to_true_pair": 0.72,
                "best_constituent_similarity": 0.63,
                "WPRI": 0.09,
            },
            {
                "triplet_id": 2,
                "layer": "L3",
                "sim_to_true_pair": 0.68,
                "best_constituent_similarity": 0.6,
                "WPRI": 0.08,
            },
        ],
    )
    (result_dir / "meta" / "plot_bundle_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "experiment_name": "chunk_step2_fused_state_experiment",
                "inputs": {
                    "preprobe_fusion_metrics": {"path": "metrics/preprobe_fusion_metrics.csv"},
                    "fusion_specificity_metrics": {"path": "metrics/fusion_specificity_metrics.csv"},
                    "whole_over_part_metrics": {"path": "metrics/whole_over_part_metrics.csv"},
                },
                "excluded_outputs": [
                    "figures/panel_a_sample_image.*",
                    "figures/panel_a_distractor_image.*",
                    "figures/panel_a_probe_image.*",
                    "figures/panel_a_overlap_support.*",
                ],
            }
        ),
        encoding="utf-8",
    )

    output_dir = result_dir / "figures_plot_only"
    assert main(["--input-dir", str(result_dir), "--output-dir", str(output_dir)]) == 0
    for stem in (
        "panel_b_fusion_form_scatter",
        "panel_c_fusion_dual_score",
        "panel_c_fusion_imbalance",
        "panel_d_true_pair_percentile",
        "panel_d_true_pair_z_score",
        "panel_d_true_pair_top1_rate",
        "panel_e_true_pair_vs_best_part",
        "panel_e_wpri_distribution",
        "panel_f_true_vs_shuffled_pair_score",
        "panel_f_true_minus_shuffled_control",
    ):
        assert (output_dir / f"{stem}.png").exists()
    assert not (output_dir / "panel_a_sample_image.png").exists()


def test_chunk_step2_plot_only_requires_metrics_bundle(tmp_path: Path) -> None:
    result_dir = tmp_path / "bundle_missing"
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (result_dir / name).mkdir(parents=True, exist_ok=True)
    (result_dir / "meta" / "plot_bundle_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "experiment_name": "chunk_step2_fused_state_experiment",
                "inputs": {
                    "preprobe_fusion_metrics": {"path": "metrics/preprobe_fusion_metrics.csv"},
                    "fusion_specificity_metrics": {"path": "metrics/fusion_specificity_metrics.csv"},
                    "whole_over_part_metrics": {"path": "metrics/whole_over_part_metrics.csv"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="preprobe_fusion_metrics.csv"):
        main(["--input-dir", str(result_dir), "--output-dir", str(result_dir / "figures_plot_only")])
