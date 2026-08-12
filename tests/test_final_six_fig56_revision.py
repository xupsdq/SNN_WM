from __future__ import annotations

import base64
import hashlib
import json
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.plotting.paper_fig.final_six.specs import get_figure_spec
from src.plotting.paper_fig.layout_contract import validate_layout_contract


def test_fig5_uses_three_equal_two_panel_rows() -> None:
    spec = get_figure_spec("fig5")
    report = validate_layout_contract(spec)
    assert report.ok, report.failures
    assert spec["canvas_mm"] == [165.0, 152.0]
    assert [spec["slots"][panel][1] for panel in "abcdef"] == [
        2.0,
        2.0,
        52.0,
        52.0,
        102.0,
        102.0,
    ]
    assert all(spec["slots"][panel][2] == 79.5 for panel in "abcdef")
    assert [spec["panels"][panel]["chart"] for panel in "abcdef"] == [
        "ordered_bars",
        "boxplot",
        "ordered_lines",
        "ordered_lines",
        "heatmap",
        "heatmap",
    ]
    assert spec["panels"]["c"]["identity_reference"] is True
    assert spec["panels"]["d"]["references"] == [
        {"value": 0.5, "label": "latest-only"}
    ]
    assert spec["panels"]["e"]["unavailable_color"] == "#FFFFFF"
    assert spec["panels"]["e"]["colorbar_orientation"] == "horizontal_top"
    assert spec["panels"]["f"]["colorbar_orientation"] == "horizontal_top"
    assert [spec["panels"][panel]["plot_bbox_mm"][1] for panel in "abcdef"] == [
        12.0,
        12.0,
        62.0,
        62.0,
        112.0,
        112.0,
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
    assert spec["slots"]["a"] == [2.0, 2.0, 79.5, 48.0]
    assert spec["slots"]["b"] == [83.5, 2.0, 79.5, 48.0]
    assert all(spec["slots"][panel][2] == 79.5 for panel in "abcdef")
    assert spec["panels"]["a"]["child_plot_bboxes_mm"] == [
        [13.0, 12.0, 31.0, 28.0],
        [47.0, 12.0, 31.0, 28.0],
    ]
    assert spec["panels"]["b"]["plot_bbox_mm"] == [94.5, 12.0, 65.5, 28.0]
    assert spec["panels"]["a"]["approved_internal_split"] is True
    assert spec["panels"]["d"]["colorbar_orientation"] == "horizontal_top"
    assert spec["panels"]["c"]["plot_bbox_mm"][1:] == [62.0, 65.5, 28.0]
    assert spec["panels"]["d"]["plot_bbox_mm"][1:] == [62.0, 65.5, 28.0]
    assert spec["panels"]["f"]["x_order"] == ["no_overlap", "overlap"]
    assert spec["panels"]["f"]["hue_order"] == ["low", "high"]
    display_text: list[str] = []
    for panel in spec["panels"].values():
        for key, value in panel.items():
            if key.endswith("label") and isinstance(value, str):
                display_text.append(value)
            if key.endswith("labels") and isinstance(value, dict):
                display_text.extend(str(label) for label in value.values())
    assert all(" pp" not in label.lower() for label in display_text)


def test_reviewer_requested_panels_use_claim_faithful_encodings() -> None:
    fig2 = get_figure_spec("fig2")["panels"]["d"]
    assert fig2["chart"] == "category_points"
    assert fig2["x_order"] == ["matched_random", "changed_events"]
    assert fig2["y_label"] == "Residual magnitude"
    assert "references" not in fig2

    fig3 = get_figure_spec("fig3")["panels"]["c"]
    assert fig3["chart"] == "ordered_lines"
    assert fig3["x_order"] == [
        "overlap_dominant",
        "probe_only_dominant",
        "random_matched",
    ]
    assert fig3["hue_order"] == ["P_advance", "P_recruit", "P_loss"]
    assert fig3["y_label"] == "Transition probability (%)"

    fig5 = get_figure_spec("fig5")["panels"]["f"]
    assert fig5["chart"] == "heatmap"
    assert fig5["x_field"] == "seq_len"
    assert fig5["y_field"] == "delay_ms"
    assert fig5["colorbar_label"] == "Matched − deranged"

    fig6 = get_figure_spec("fig6")["panels"]["f"]
    assert fig6["chart"] == "two_by_two"
    assert fig6["x_order"] == ["no_overlap", "overlap"]
    assert fig6["hue_order"] == ["low", "high"]
    assert fig6["show_contrast_panel"] is False


def test_fig2_exact_input_schematic_uses_one_shared_current_input() -> None:
    panel = get_figure_spec("fig2")["panels"]["a"]
    assert panel["custom_renderer"] == "fig2_paired_dms"
    assert panel["legend_owner"] == "none"
    assert panel["role"] == (
        "define the exact-input counterfactual and the paired response comparison"
    )

    layout = panel["schematic_layout"]
    assert layout["content_bounds"] == [0.0, 0.0, 152.0, 40.0]
    assert set(layout["history_rows"]) == {"A", "C"}
    assert layout["history_rows"]["A"]["center_y"] == 29.0
    assert layout["history_rows"]["C"]["center_y"] == 11.0
    assert layout["shared_b"]["image_bbox"] == [50.0, 14.0, 12.0, 12.0]
    assert layout["comparison_bbox"] == [76.0, 3.0, 73.0, 34.0]
    assert layout["state_icon_bbox"] == [88.0, 16.0, 12.0, 12.0]
    assert layout["behavior_icon_bbox"] == [125.0, 16.0, 12.0, 12.0]
    assert "title" not in layout


def test_fig3_ends_with_full_width_state_evolution_synthesis() -> None:
    spec = get_figure_spec("fig3")
    report = validate_layout_contract(spec)
    assert report.ok, report.failures
    assert spec["canvas_mm"] == [165.0, 202.0]
    assert list(spec["panels"]) == list("abcdefg")
    assert all(
        spec["panels"][panel_id]["chart"] != "svg_asset"
        for panel_id in "abcdef"
    )
    panel_g = spec["panels"]["g"]
    assert panel_g["chart"] == "svg_asset"
    assert (
        panel_g["source"]
        == "meta/panel_g_asset_manifest.csv"
    )
    assert panel_g["asset_embedding"] == "inline"
    assert panel_g["asset_viewbox_override"] == "0 0 1560 420"
    assert panel_g["legend_owner"] == "none"
    assert spec["slots"]["g"] == [2.0, 152.0, 161.0, 48.0]


def test_fig3_state_evolution_embedded_inputs_decode() -> None:
    asset = Path(
        "src/plotting/paper_fig/assets/fig3_state_evolution.svg"
    )
    root = ET.parse(asset).getroot()
    images = root.findall(".//{http://www.w3.org/2000/svg}image")
    assert len(images) == 3
    for element in images:
        href = element.get("href")
        assert href is not None
        prefix, encoded = href.split(",", maxsplit=1)
        assert prefix == "data:image/png;base64"
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(payload)) as embedded:
            embedded.load()
            assert embedded.size == (28, 28)
            assert embedded.convert("L").getextrema()[1] > 200


def test_fig4_is_a_two_by_two_quantitative_figure() -> None:
    spec = get_figure_spec("fig4")
    report = validate_layout_contract(spec)
    assert report.ok, report.failures
    assert spec["canvas_mm"] == [165.0, 102.0]
    assert list(spec["panels"]) == list("abcd")
    assert spec["slots"] == {
        "a": [2.0, 2.0, 79.5, 48.0],
        "b": [83.5, 2.0, 79.5, 48.0],
        "c": [2.0, 52.0, 79.5, 48.0],
        "d": [83.5, 52.0, 79.5, 48.0],
    }
    assert all(
        panel["chart"] != "protocol"
        and panel["chart"] != "schematic"
        and "schematic_layout" not in panel
        for panel in spec["panels"].values()
    )
    assert spec["panels"]["a"]["x_order"] == ["K1", "K5"]
    assert spec["panels"]["b"]["x_order"] == ["K1", "K5"]
    assert spec["panels"]["c"]["hue_order"] == ["observed", "passive"]
    assert spec["panels"]["d"]["hue_order"] == ["rescue", "loss"]
    assert {
        group["group_id"]: group["panels"]
        for group in spec["layout_contract"]["alignment_groups"]
        if group["group_id"].startswith("fig4_")
    } == {
        "fig4_row_1_plot_axes": ["a", "b"],
        "fig4_row_2_plot_axes": ["c", "d"],
    }
    forbidden = " ".join(spec["reader_contract"]["forbidden_inferences"])
    assert "necessity" in forbidden
    assert (
        "Fig.4 causally generates either Fig.5 morphology or Fig.6 function"
        in forbidden
    )


def test_schematic_icon_assets_are_vendored_and_attributed() -> None:
    asset_root = (
        Path("src")
        / "plotting"
        / "paper_fig"
        / "assets"
        / "tabler-icons-v3.46.0"
    )
    manifest = json.loads(
        (asset_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["collection"] == "Tabler Icons"
    assert manifest["version"] == "3.46.0"
    assert manifest["license"] == "MIT"
    assert (asset_root / manifest["license_file"]).is_file()
    assert {icon["name"] for icon in manifest["icons"]} == {
        "hierarchy-3",
        "matrix",
        "replace",
        "target-arrow",
    }
    for icon in manifest["icons"]:
        icon_bytes = (asset_root / icon["file"]).read_bytes()
        assert hashlib.sha256(icon_bytes).hexdigest() == icon["sha256"]
        assert icon["source"].startswith(
            "https://github.com/tabler/tabler-icons/blob/v3.46.0/"
        )
