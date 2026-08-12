from __future__ import annotations

import ast
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import to_rgb
import numpy as np

from src.plotting.common.colors import (
    NATURE_COMPATIBLE_PALETTE,
    get_plot_cmap,
    get_plot_color,
    get_plot_distinction,
)


def _relative_luminance(color: str | tuple[float, ...]) -> float:
    rgb = np.asarray(to_rgb(color), dtype=float)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ np.asarray([0.2126, 0.7152, 0.0722]))


def _contrast_ratio(foreground: str, background: str) -> float:
    first, second = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (first + 0.05) / (second + 0.05)


def test_nature_compatible_palette_has_restrained_semantic_roots() -> None:
    assert NATURE_COMPATIBLE_PALETTE["ink"] == "#222222"
    assert NATURE_COMPATIBLE_PALETTE["primary_navy"] == "#0072B2"
    assert NATURE_COMPATIBLE_PALETTE["mechanism_teal"] == "#009E73"
    assert NATURE_COMPATIBLE_PALETTE["comparison_coral"] == "#D55E00"
    assert NATURE_COMPATIBLE_PALETTE["fused_slate"] == "#CC79A7"

    assert get_plot_color("dynamic") == NATURE_COMPATIBLE_PALETTE["primary_navy"]
    assert get_plot_color("trial_shuffled_ux") == NATURE_COMPATIBLE_PALETTE["comparison_coral"]
    assert get_plot_color("sample_probe_overlap") == NATURE_COMPATIBLE_PALETTE["mechanism_teal"]
    assert get_plot_color("fused_state") == NATURE_COMPATIBLE_PALETTE["fused_slate"]


def test_controls_are_achromatic_and_high_stsp_is_not_warning_red() -> None:
    for role in (
        "static_frozen",
        "baseline_control",
        "random_control",
        "non_overlap_control",
        "other_residual",
        "transition_loss",
        "loser",
    ):
        rgb = np.asarray(to_rgb(get_plot_color(role)), dtype=float)
        assert float(rgb.max() - rgb.min()) < 0.05, (role, get_plot_color(role))

    warning_colours = {
        NATURE_COMPATIBLE_PALETTE["comparison_coral"],
        NATURE_COMPATIBLE_PALETTE["comparison_salmon"],
    }
    assert get_plot_color("high_stsp") not in warning_colours
    assert get_plot_color("peak_region") not in warning_colours
    assert get_plot_color("high_stsp") == NATURE_COMPATIBLE_PALETTE["mechanism_teal"]
    assert get_plot_color("transition_loss") != NATURE_COMPATIBLE_PALETTE["fused_slate"]
    assert get_plot_color("loser") != NATURE_COMPATIBLE_PALETTE["fused_slate"]


def test_ordered_stages_share_a_cool_family_and_mechanism_uses_presence_contrast() -> None:
    stage_progression = (
        NATURE_COMPATIBLE_PALETTE["primary_cyan"],
        NATURE_COMPATIBLE_PALETTE["primary_navy"],
        NATURE_COMPATIBLE_PALETTE["mechanism_teal"],
    )
    assert tuple(get_plot_color(role) for role in ("old_input", "middle_input", "recent_input")) == stage_progression
    assert tuple(get_plot_color(role) for role in ("layer1", "layer2", "layer3")) == stage_progression
    assert get_plot_color("low_stsp") == NATURE_COMPATIBLE_PALETTE["neutral_light"]
    assert get_plot_color("high_stsp") == NATURE_COMPATIBLE_PALETTE["mechanism_teal"]

    stsp = [_relative_luminance(get_plot_color(role)) for role in ("low_stsp", "high_stsp")]
    assert stsp[0] > stsp[1]


def test_key_pairs_survive_common_colour_vision_simulations() -> None:
    matrices = (
        np.eye(3),
        np.asarray([[0.367, 0.861, -0.228], [0.280, 0.673, 0.047], [-0.012, 0.043, 0.969]]),
        np.asarray([[0.152, 1.053, -0.205], [0.115, 0.786, 0.099], [-0.004, -0.048, 1.052]]),
    )
    pairs = (
        ("first_item_reference", "second_item_reference"),
        ("first_item_reference", "fused_state"),
        ("second_item_reference", "fused_state"),
        ("dynamic", "trial_shuffled_ux"),
        ("dynamic", "static_frozen"),
        ("old_input", "middle_input"),
        ("middle_input", "recent_input"),
        ("low_stsp", "high_stsp"),
        ("transition_recruit", "transition_loss"),
        ("winner", "loser"),
    )
    for left, right in pairs:
        first = np.asarray(to_rgb(get_plot_color(left)))
        second = np.asarray(to_rgb(get_plot_color(right)))
        left_style = get_plot_distinction(left)
        right_style = get_plot_distinction(right)
        redundant = (
            left_style.hatch,
            left_style.linestyle,
            left_style.marker_fill,
        ) != (
            right_style.hatch,
            right_style.linestyle,
            right_style.marker_fill,
        )
        for matrix in matrices:
            distance = float(
                np.linalg.norm(
                    np.clip(matrix @ first, 0, 1) - np.clip(matrix @ second, 0, 1)
                )
            )
            assert distance >= 0.12 or redundant, (left, right, distance)


def test_text_and_large_fill_contrast_meet_nature_guidance() -> None:
    ink = NATURE_COMPATIBLE_PALETTE["ink"]
    for background in (
        "white",
        "neutral_pale",
        "primary_tint",
        "mechanism_tint",
        "comparison_tint",
        "fused_tint",
    ):
        assert _contrast_ratio(ink, NATURE_COMPATIBLE_PALETTE[background]) >= 4.5


def test_quantitative_colormaps_are_established_and_luminance_ordered() -> None:
    expected = {
        "stsp_support": "Blues",
        "item_contribution": "Blues",
        "update_count": "cividis",
        "peak_strength": "magma",
        "signed_effect": "PuOr_r",
    }
    for role, name in expected.items():
        assert get_plot_cmap(role).name == name

    samples = np.linspace(0.0, 1.0, 256)
    for role in ("stsp_support", "item_contribution", "update_count", "peak_strength"):
        luminance = np.asarray([_relative_luminance(get_plot_cmap(role)(value)) for value in samples])
        differences = np.diff(luminance)
        is_increasing = bool(np.all(differences >= -2e-4))
        is_decreasing = bool(np.all(differences <= 2e-4))
        assert is_increasing or is_decreasing, role


def test_paper_panel_renderers_do_not_bypass_the_shared_palette() -> None:
    panel_dir = Path(__file__).resolve().parents[1] / "src" / "plotting" / "paper_fig" / "panels"
    offenders: dict[str, list[str]] = {}
    for path in panel_dir.glob("fig*_panels.py"):
        matches = re.findall(r"#[0-9A-Fa-f]{6}", path.read_text(encoding="utf-8"))
        if matches:
            offenders[path.name] = matches
    assert not offenders


def test_paper_panel_renderers_do_not_enable_background_grids() -> None:
    panel_dir = Path(__file__).resolve().parents[1] / "src" / "plotting" / "paper_fig" / "panels"
    offenders: list[tuple[str, int]] = []
    for path in panel_dir.glob("fig*_panels.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "grid":
                continue
            disabled_positionally = bool(
                node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value is False
            )
            disabled_by_keyword = any(
                keyword.arg in {"b", "visible"}
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
                for keyword in node.keywords
            )
            if not (disabled_positionally or disabled_by_keyword):
                offenders.append((path.name, node.lineno))
    assert not offenders
