from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.plotting.experiments.overlap_causal_input_perturbation_experiment_plot import main


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_overlap_causal_plot_only_generates_all_figures(tmp_path: Path) -> None:
    result_dir = tmp_path / "bundle"
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (result_dir / name).mkdir(parents=True, exist_ok=True)

    _write_csv(
        result_dir / "data" / "pair_condition_pattern_results.csv",
        [
            {"pair_id": 1, "condition": "sample_keep_overlap_only_dynamic", "DPI_L3": 0.2, "mean_S_dyn_L3": 0.7, "mean_S_sta_L3": 0.5},
            {"pair_id": 2, "condition": "sample_keep_overlap_only_dynamic", "DPI_L3": 0.1, "mean_S_dyn_L3": 0.65, "mean_S_sta_L3": 0.55},
            {"pair_id": 1, "condition": "sample_keep_nonoverlap_only_dynamic", "DPI_L3": -0.05, "mean_S_dyn_L3": 0.4, "mean_S_sta_L3": 0.45},
            {"pair_id": 2, "condition": "sample_keep_nonoverlap_only_dynamic", "DPI_L3": -0.1, "mean_S_dyn_L3": 0.35, "mean_S_sta_L3": 0.45},
        ],
    )
    np.savez_compressed(
        result_dir / "data" / "pair_trace_similarity.npz",
        condition_name=np.asarray(
            [
                "sample_keep_overlap_only_dynamic",
                "sample_keep_overlap_only_dynamic",
                "sample_keep_nonoverlap_only_dynamic",
                "sample_keep_nonoverlap_only_dynamic",
            ]
        ),
        S_dyn_L3=np.asarray([[0.8, 0.7], [0.75, 0.65], [0.4, 0.35], [0.45, 0.4]], dtype=np.float32),
        S_sta_L3=np.asarray([[0.5, 0.45], [0.55, 0.5], [0.5, 0.45], [0.48, 0.44]], dtype=np.float32),
        DPI_L3=np.asarray([[0.3, 0.25], [0.2, 0.15], [-0.1, -0.1], [-0.03, -0.04]], dtype=np.float32),
    )
    (result_dir / "meta" / "plot_bundle_manifest.json").write_text(
        json.dumps({"version": 1, "experiment_name": "overlap_causal_input_perturbation_experiment", "inputs": {"pair_condition_pattern_results": {"path": "data/pair_condition_pattern_results.csv", "required_columns": ["pair_id", "condition", "DPI_L3", "mean_S_dyn_L3", "mean_S_sta_L3"]}, "pair_trace_similarity": {"path": "data/pair_trace_similarity.npz"}}}),
        encoding="utf-8",
    )

    output_dir = result_dir / "figures_plot_only"
    assert main(["--input-dir", str(result_dir), "--output-dir", str(output_dir)]) == 0
    for stem in (
        "dpi_l3_trace_overlap_vs_nonoverlap",
        "dpi_l3_summary_overlap_vs_nonoverlap",
        "supplementary_s2p_trace_similarity_keep_overlap_only",
        "supplementary_s2p_trace_similarity_keep_nonoverlap_only",
        "supplementary_s2p_dpi_keep_overlap_only",
        "supplementary_s2p_dpi_keep_nonoverlap_only",
    ):
        assert (output_dir / f"{stem}.png").exists()
