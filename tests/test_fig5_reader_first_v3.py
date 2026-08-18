import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from scripts.audit_layout_profile import audit
from src.plotting.common.colors import get_plot_color
from src.plotting.paper_fig.candidates import manuscript_fig5_reader_first_v3 as fig5_v3


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "src" / "plotting" / "paper_fig" / "candidates" / "specs" / "manuscript_fig5_reader_first_v3.json"


def test_fig5_v3_2_plus_3_layout_profile():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    report = audit(spec)
    assert report["status"] == "passed", report
    assert report["plot_alignment"] == {"a_b": True, "c_d": True, "d_e": True}
    assert spec["canvas_mm"] == [165.0, 102.0]
    assert spec["slots"]["a"] == [2.0, 2.0, 79.5, 48.0]
    assert spec["slots"]["e"] == [110.667, 52.0, 52.333, 48.0]
    assert spec["panels"]["c"]["plot_bbox_mm"] == [17.0, 62.0, 34.333, 28.0]
    assert spec["panels"]["d"]["plot_bbox_mm"] == [69.333, 62.0, 36.334, 28.0]
    assert spec["panels"]["e"]["plot_bbox_mm"] == [123.667, 62.0, 36.333, 28.0]
    assert report["fig5_v3_bottom_geometry"] is True


def test_fig5_v3_has_five_quantitative_panels_and_global_colors():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert set(spec["panels"]) == set("abcde")
    assert all("schematic" not in panel["chart"] for panel in spec["panels"].values())
    assert spec["panels"]["a"]["endpoint_colors"] == {"input_response_l2": "layer2", "successor_state_l3": "layer3"}
    assert spec["panels"]["b"]["condition_colors"] == {"overlap": "sample_probe_overlap", "non_overlap": "non_overlap_control", "random": "random_control"}
    assert get_plot_color("layer2") == "#0072B2"
    assert get_plot_color("layer3") == "#009E73"
    assert get_plot_color("sample_probe_overlap") == "#009E73"
    assert get_plot_color("random_control") == "#666666"


def test_fig5_v3_bootstrap_requires_network_cohort():
    try:
        fig5_v3._bootstrap_mean_ci([0.1] * 19)
    except ValueError as exc:
        assert "exactly 20" in str(exc)
    else:
        raise AssertionError("bootstrap accepted an incomplete network cohort")


def test_fig5_v3_panel_b_zero_controls_use_colored_bar_caps_and_patch_legend():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    stats = pd.DataFrame([
        {"endpoint": endpoint, "removed_sites": condition, "estimate": value, "ci95_low": value, "ci95_high": value}
        for endpoint in ["input_response", "post_input_state"]
        for condition, value in [("overlap", 0.72), ("non_overlap", 0.0), ("random", 0.0)]
    ])
    figure, axis = plt.subplots()
    try:
        qa = fig5_v3._draw_overlap(axis, stats, spec["panels"]["b"])
        assert {item["condition"] for item in qa["zero_caps"]} == {"non_overlap", "random"}
        assert len(qa["zero_caps"]) == 4
        assert "zero_markers" not in qa
        assert "zero_marker_shapes" not in qa
        assert qa["legend_handle_types"] == ["Patch", "Patch", "Patch"]
        assert all(isinstance(handle, Patch) for handle in axis.get_legend().legend_handles)
        assert axis.get_ylim() == (-3.0, 100.0)
        assert list(axis.get_yticks()) == [0.0, 25.0, 50.0, 75.0, 100.0]
        assert axis.get_xlabel() == ""
        assert axis.get_title() == ""
    finally:
        plt.close(figure)


def test_fig5_v3_gate_fill_and_interval_metadata():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["panels"]["b"]["interval_policy"]["non_overlap"] == "candidate_network_bootstrap_20000"
    assert spec["panels"]["b"]["interval_policy"]["random"] == "candidate_network_bootstrap_20000"
    assert "condition_label" not in spec["panels"]["b"]
    assert "condition_markers" not in spec["panels"]["b"]
    assert "condition_label" not in spec["panels"]["c"]
    assert spec["panels"]["c"]["inference_roles"]["next_response"] == "descriptive_gate"
    assert spec["panels"]["c"]["endpoint_filled"] == {"next_response": False, "following_response": True, "new_successor": True}
    gate_raw, gate_stats = fig5_v3._build_gate_frame({seed: {"gate_mean_early_layer2_C_donor_transfer": 0.34} for seed in fig5_v3.EXPECTED_SEEDS})
    assert len(gate_raw) == 20
    assert gate_stats.iloc[0]["interval_source"] == "candidate_network_bootstrap_20000"
    assert int(gate_stats.iloc[0]["bootstrap_draws"]) == 20000
    assert gate_stats.iloc[0]["inference_role"] == "descriptive_gate"
    extension_root = fig5_v3._repo_root() / fig5_v3.EXTENSION_ROOT_REL
    extension, extension_network, extension_population, _verdict, _aggregate_access = fig5_v3._load_extension(extension_root)
    seed_overlap, _seed_twohop, _seed_access, _seed_parent, _seed_before = fig5_v3._load_seed_summaries(fig5_v3._repo_root())
    _overlap_raw, overlap_stats = fig5_v3._build_overlap_frames(seed_overlap, extension_population, extension_network)
    control_rows = overlap_stats.loc[overlap_stats["removed_sites"].isin(["non_overlap", "random"])]
    assert set(control_rows["interval_source"]) == {"candidate_network_bootstrap_20000"}
    assert set(control_rows["inference_role"]) == {"control_summary"}
    assert set(control_rows["bootstrap_draws"].astype(int)) == {20000}
    overlap_rows = overlap_stats.loc[overlap_stats["removed_sites"].eq("overlap")]
    assert set(overlap_rows["interval_source"]) == {"confirmatory_supplied_network_bootstrap"}
    assert set(overlap_rows["inference_role"]) == {"confirmatory_endpoint"}
    stats = pd.DataFrame([
        {"endpoint": endpoint, "estimate": value, "ci95_low": value - 0.001, "ci95_high": value + 0.001}
        for endpoint, value in [("next_response", 0.34), ("following_response", 0.31), ("new_successor", 0.27)]
    ])
    figure, axis = plt.subplots()
    try:
        qa = fig5_v3._draw_propagation(axis, stats, spec["panels"]["c"])
        assert qa["descriptive_gate"] == "next_response"
        assert qa["endpoint_filled"] == {"next_response": False, "following_response": True, "new_successor": True}
        assert qa["y_axis_labels"] == ["Next\nresponse", "Following\nresponse", "New\nsuccessor"]
        assert [label.get_text() for label in axis.get_yticklabels()] == qa["y_axis_labels"]
        assert axis.get_yticks().tolist() == [0.0, 1.0, 2.0]
        assert len(axis.texts) == 0
        patches = axis.patches
        assert len(patches) == 3
        assert patches[0].get_facecolor()[3] == 0.0
        assert patches[1].get_facecolor()[3] > 0.0
        assert patches[2].get_facecolor()[3] > 0.0
    finally:
        plt.close(figure)


def test_fig5_v3_panel_d_omits_analytic_zero_passive_series():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    stats = pd.DataFrame([
        {
            "condition": condition,
            "stage_k": stage,
            "estimate": value,
            "ci95_low": value - 0.01 if condition == "observed" else value,
            "ci95_high": value + 0.01 if condition == "observed" else value,
        }
        for condition, value in (("observed", 0.53), ("passive", 0.0))
        for stage in range(2, 11)
    ])
    figure, axis = plt.subplots()
    try:
        qa = fig5_v3._draw_recurrence(axis, stats, spec["panels"]["d"])
        assert qa["conditions"] == ["observed"]
        assert qa["legend_rendered"] is False
        assert set(qa["rendered"]) == {"observed"}
        assert axis.get_legend() is None
        assert len(axis.lines) == 1
    finally:
        plt.close(figure)


def test_fig5_v3_panel_e_rescue_is_top_and_loss_bottom():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    stats = pd.DataFrame([
        {"outcome_type": outcome, "prefix_k": prefix, "estimate": value, "ci95_low": value - 1.0, "ci95_high": value + 1.0}
        for outcome, values in [("rescue", (40.0, 17.0)), ("loss", (28.0, 62.0))]
        for prefix, value in zip(("K1", "K5"), values)
    ])
    figure, axis = plt.subplots()
    try:
        qa = fig5_v3._draw_behavior(axis, stats, spec["panels"]["e"])
        assert qa["top_to_bottom"] == ["rescue", "loss"]
        assert qa["y_inverted"] is True
        assert axis.yaxis_inverted()
        assert [label.get_text() for label in axis.get_yticklabels()] == ["Rescue", "Loss"]
    finally:
        plt.close(figure)
