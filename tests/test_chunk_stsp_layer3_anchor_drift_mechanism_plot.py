from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.plotting.experiments.chunk_stsp_layer3_anchor_drift_mechanism_plot import main


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_chunk_stsp_layer3_anchor_drift_plot_only_generates_all_panels(tmp_path: Path) -> None:
    result_dir = tmp_path / "bundle"
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (result_dir / name).mkdir(parents=True, exist_ok=True)

    _write_csv(
        result_dir / "metrics" / "layer3_changed_synapse_metrics.csv",
        [
            {"record_type": "stage_summary", "seq_len": 3, "stage_k": 1, "changed_synapse_fraction": 0.1, "positive_change_mass_ratio_active": 0.2},
            {"record_type": "stage_summary", "seq_len": 3, "stage_k": 2, "changed_synapse_fraction": 0.2, "positive_change_mass_ratio_active": 0.3},
        ],
    )
    _write_csv(
        result_dir / "metrics" / "layer3_changed_rank_metrics.csv",
        [
            {"record_type": "stage_summary", "seq_len": 3, "stage_k": 1, "changed_rank_percentile_mean": 0.4, "changed_top_5pct_enrichment": 1.2},
            {"record_type": "stage_summary", "seq_len": 3, "stage_k": 2, "changed_rank_percentile_mean": 0.5, "changed_top_5pct_enrichment": 1.5},
        ],
    )
    _write_csv(
        result_dir / "metrics" / "layer3_ping_coupling_metrics.csv",
        [
            {"record_type": "trial_level", "seq_len": 3, "stage_k": 1, "changed_topness_default": 1.1, "ping_normalized_recency": 0.5, "ping_latest_item_hit_chance_corrected": 0.2, "stage_to_stage_anchor_shift": 0.1},
            {"record_type": "trial_level", "seq_len": 3, "stage_k": 2, "changed_topness_default": 1.3, "ping_normalized_recency": 0.7, "ping_latest_item_hit_chance_corrected": 0.4, "stage_to_stage_anchor_shift": 0.2},
        ],
    )
    (result_dir / "meta" / "plot_bundle_manifest.json").write_text(json.dumps({"version": 1, "experiment_name": "chunk_stsp_layer3_anchor_drift_mechanism"}), encoding="utf-8")

    output_dir = result_dir / "figures_plot_only"
    assert main(["--input-dir", str(result_dir), "--output-dir", str(output_dir)]) == 0
    for stem in (
        "changed_synapse_fraction_vs_stage",
        "positive_change_mass_vs_stage",
        "changed_rank_enrichment",
        "ping_coupling_with_changed_topness",
        "changed_topness_vs_chance_corrected_latest_hit",
    ):
        assert (output_dir / f"{stem}.png").exists()
