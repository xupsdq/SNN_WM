from __future__ import annotations

from src.plotting.paper_fig.final_six.specs import get_figure_spec
from src.plotting.paper_fig.layout_contract import validate_layout_contract


def test_fig5_uses_compact_pair_band_and_multi_item_row() -> None:
    spec = get_figure_spec("fig5")
    report = validate_layout_contract(spec)
    assert report.ok, report.failures
    assert spec["canvas_mm"] == [165.0, 102.0]
    assert [spec["slots"][panel][1] for panel in "abc"] == [2.0, 2.0, 2.0]
    assert [spec["slots"][panel][1] for panel in "def"] == [52.0, 52.0, 52.0]
    assert [spec["slots"][panel][2] for panel in "abcdef"] == [
        52.333,
        52.334,
        52.333,
        52.333,
        52.334,
        52.333,
    ]
    assert [spec["panels"][panel]["chart"] for panel in "abc"] == [
        "ordered_bars",
        "boxplot",
        "ordered_bars",
    ]
    assert spec["panels"]["d"]["show_individual_traces"] is False
    assert spec["panels"]["e"]["unavailable_color"] == "#FFFFFF"
    assert spec["panels"]["e"]["colorbar_orientation"] == "horizontal_top"
    assert spec["panels"]["f"]["colorbar_orientation"] == "horizontal_top"
    assert [spec["panels"][panel]["plot_bbox_mm"][1:] for panel in "def"] == [
        [62.0, 38.0, 28.0],
        [62.0, 38.0, 28.0],
        [62.0, 38.0, 28.0],
    ]


def test_fig6_uses_one_coordinate_system_per_panel() -> None:
    spec = get_figure_spec("fig6")
    report = validate_layout_contract(spec)
    assert report.ok, report.failures
    assert spec["canvas_mm"] == [165.0, 152.0]
    assert [spec["panels"][panel]["chart"] for panel in "abcdef"] == [
        "partial_cue_split",
        "ordered_lines",
        "ordered_lines",
        "heatmap",
        "ordered_bars",
        "two_by_two",
    ]
    assert [spec["slots"][panel][1] for panel in "abcdef"] == [
        2.0,
        2.0,
        52.0,
        52.0,
        102.0,
        102.0,
    ]
    assert spec["slots"]["a"] == [2.0, 2.0, 94.0, 48.0]
    assert spec["slots"]["b"] == [98.0, 2.0, 65.0, 48.0]
    assert all(spec["slots"][panel][2] == 79.5 for panel in "cdef")
    assert spec["panels"]["a"]["child_plot_bboxes_mm"] == [
        [13.0, 12.0, 38.0, 28.0],
        [55.0, 12.0, 38.0, 28.0],
    ]
    assert spec["panels"]["b"]["plot_bbox_mm"] == [109.0, 12.0, 51.0, 28.0]
    assert spec["panels"]["a"]["approved_internal_split"] is True
    assert spec["panels"]["d"]["colorbar_orientation"] == "horizontal_top"
    assert spec["panels"]["c"]["plot_bbox_mm"][1:] == [62.0, 65.5, 28.0]
    assert spec["panels"]["d"]["plot_bbox_mm"][1:] == [62.0, 65.5, 28.0]
    assert spec["panels"]["f"]["show_contrast_panel"] is False
    display_text: list[str] = []
    for panel in spec["panels"].values():
        for key, value in panel.items():
            if key.endswith("label") and isinstance(value, str):
                display_text.append(value)
            if key.endswith("labels") and isinstance(value, dict):
                display_text.extend(str(label) for label in value.values())
    assert all(" pp" not in label.lower() for label in display_text)
