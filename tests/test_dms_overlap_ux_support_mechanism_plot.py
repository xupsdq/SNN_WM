from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.plotting.experiments.dms_overlap_ux_support_mechanism_experiment_plot import main


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_dms_overlap_plot_only_generates_all_panels(tmp_path: Path) -> None:
    result_dir = tmp_path / "bundle"
    for name in ("data", "figures", "logs", "metrics", "meta"):
        (result_dir / name).mkdir(parents=True, exist_ok=True)

    _write_csv(result_dir / "data" / "pair_metadata.csv", [{"trial_id": 0, "sample_id": 1, "probe_id": 2}])
    (result_dir / "data" / "pair_mask_metadata.json").write_text(json.dumps({"trials": []}), encoding="utf-8")
    _write_csv(
        result_dir / "metrics" / "preprobe_stsp_summary.csv",
        [
            {"trial_id": 0, "model_type": "dynamic", "ux_overlap_pre": 0.6, "ux_probe_only_pre": 0.3, "support_area": 12, "mean_ux_on_overlap": 0.6, "total_memory_support": 1.8},
            {"trial_id": 0, "model_type": "static", "ux_overlap_pre": 0.4, "ux_probe_only_pre": 0.2, "support_area": 12, "mean_ux_on_overlap": 0.4, "total_memory_support": 1.2},
        ],
    )
    _write_csv(
        result_dir / "metrics" / "l1_firing_transition_summary.csv",
        [
            {"aggregation_scope": "per_trial", "unit_group": group, "P_advance": 0.2, "P_recruit": 0.1, "P_loss": 0.05, "delta_early_spike_count": 1.0, "delta_first_spike_latency": -0.5}
            for group in ("all_units", "overlap_dominant", "probe_only_dominant")
        ],
    )
    _write_csv(
        result_dir / "metrics" / "l1_input_source_gain_summary.csv",
        [
            {"aggregation_scope": "per_trial", "unit_group": group, "transition_focus": "advance_or_recruit", "overlap_input_gain": 0.3, "probe_only_input_gain": 0.1, "input_selectivity_gain": 0.2}
            for group in ("all_units", "overlap_dominant", "probe_only_dominant")
        ],
    )
    _write_csv(
        result_dir / "metrics" / "l1_loss_inhibition_summary.csv",
        [
            {"aggregation_scope": "per_trial", "unit_group": group, "lost_spike_delta_inh": 0.05, "n_lost_spike_units": 2}
            for group in ("all_units", "overlap_dominant", "probe_only_dominant")
        ],
    )
    _write_csv(result_dir / "data" / "l1_local_winner_loser_pairs.csv", [{"winner_loser_contrast_shift": 0.01}])
    _write_csv(
        result_dir / "data" / "l1_local_causal_chain_events.csv",
        [{"winner_pre_spike_boost": 1.0, "winner_spikes_earlier": 1.0, "loser_post_winner_suppressed": 1.0, "full_chain_satisfied": 1.0}],
    )
    _write_csv(result_dir / "metrics" / "l1_local_winner_support_summary.csv", [{"aggregation_scope": "per_trial", "local_winner_support_rate": 0.8}])
    np.savez_compressed(
        result_dir / "data" / "l1_local_event_time_alignment.npz",
        relative_time=np.asarray([-1, 0, 1], dtype=np.int64),
        winner_delta_v_aligned=np.asarray([[0.1, 0.2, 0.1]], dtype=np.float32),
        loser_delta_v_aligned=np.asarray([[0.0, -0.1, -0.05]], dtype=np.float32),
        loser_inh_before_aligned=np.asarray([[0.05, 0.08, 0.09]], dtype=np.float32),
        loser_inh_after_aligned=np.asarray([[0.05, 0.08, 0.09]], dtype=np.float32),
    )
    np.savez_compressed(
        result_dir / "data" / "l1_local_winner_loser_exemplar_trace.npz",
        t_axis=np.asarray([0, 1, 2], dtype=np.int64),
        winner_v_effective_dynamic=np.asarray([-0.061, -0.058, -0.055], dtype=np.float32),
        winner_v_effective_static=np.asarray([-0.062, -0.060, -0.058], dtype=np.float32),
        loser_v_effective_dynamic=np.asarray([-0.064, -0.065, -0.067], dtype=np.float32),
        loser_v_effective_static=np.asarray([-0.063, -0.064, -0.065], dtype=np.float32),
        winner_first_spike_dynamic=np.asarray(1, dtype=np.int64),
        winner_first_spike_static=np.asarray(2, dtype=np.int64),
        loser_first_spike_dynamic=np.asarray(-1, dtype=np.int64),
        loser_first_spike_static=np.asarray(-1, dtype=np.int64),
    )
    np.savez_compressed(
        result_dir / "data" / "l1_panel_a_preprobe_gain_map.npz",
        sample_image=np.asarray([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32),
        probe_image=np.asarray([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32),
        overlap_mask=np.asarray([[0, 1], [0, 0]], dtype=np.uint8),
        probe_only_mask=np.asarray([[0, 0], [1, 0]], dtype=np.uint8),
        ux_map_pre_dynamic=np.asarray([[0.1, 0.5], [0.3, 0.2]], dtype=np.float32),
    )
    (result_dir / "meta" / "plot_bundle_manifest.json").write_text(json.dumps({"version": 1, "experiment_name": "dms_overlap_ux_support_mechanism_experiment"}), encoding="utf-8")

    output_dir = result_dir / "figures_plot_only"
    assert main(["--input-dir", str(result_dir), "--output-dir", str(output_dir)]) == 0
    assert (output_dir / "fig4_panel_a_overlap_definition.png").exists()
    assert (output_dir / "fig4_panel_s_causal_chain_prevalence.png").exists()
