from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lxml import etree  # noqa: E402
from PIL import Image  # noqa: E402

from src.plotting.common.colors import NATURE_COMPATIBLE_PALETTE  # noqa: E402
from src.plotting.paper_fig.final_six.renderer import (  # noqa: E402
    render_composed_figure,
)
from src.plotting.paper_fig.final_six.specs import (  # noqa: E402
    build_layout_contract,
    get_figure_spec,
)


REDESIGN_PLOT_VERSION = "paper_fig1_fig2_redesign_plot_v1.0.0"
CANVAS_MM = (165.0, 102.0)
MM_TO_INCH = 1.0 / 25.4
INK = NATURE_COMPATIBLE_PALETTE["ink"]
NEUTRAL = NATURE_COMPATIBLE_PALETTE["neutral_mid"]
U_COLOR = NATURE_COMPATIBLE_PALETTE["comparison_coral"]
X_COLOR = NATURE_COMPATIBLE_PALETTE["primary_navy"]
STSP_COLOR = NATURE_COMPATIBLE_PALETTE["mechanism_teal"]
EVENT_TINT = NATURE_COMPATIBLE_PALETTE["mechanism_tint"]

FIG1_SLOTS = {
    "a": [2.0, 2.0, 161.0, 48.0],
    "b": [2.0, 52.0, 52.333, 48.0],
    "c": [56.333, 52.0, 52.334, 48.0],
    "d": [110.667, 52.0, 52.333, 48.0],
}
FIG2_SLOTS = {
    "a": [2.0, 2.0, 79.5, 48.0],
    "b": [83.5, 2.0, 79.5, 48.0],
    "c": [2.0, 52.0, 79.5, 48.0],
    "d": [83.5, 52.0, 79.5, 48.0],
}


class BundleReader:
    def __init__(self, bundle: Path):
        self.bundle = bundle.resolve()
        self.accesses: list[dict[str, Any]] = []

    def _resolve(self, relative: str, *, suffixes: set[str]) -> Path:
        if Path(relative).is_absolute():
            raise PermissionError(
                f"absolute plot source is forbidden: {relative}"
            )
        path = (self.bundle / relative).resolve()
        try:
            path.relative_to(self.bundle)
        except ValueError as exc:
            raise PermissionError(
                f"plot source escapes the bundle: {path}"
            ) from exc
        if path.suffix.lower() not in suffixes:
            raise PermissionError(f"unsupported plot source type: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"required plot source is missing: {path}")
        self.accesses.append(
            {
                "path": str(path),
                "relative_path": path.relative_to(self.bundle).as_posix(),
                "sha256": _sha256(path),
            }
        )
        return path

    def read_csv(self, relative: str) -> pd.DataFrame:
        return pd.read_csv(self._resolve(relative, suffixes={".csv"}))

    def read_json(self, relative: str) -> dict[str, Any]:
        path = self._resolve(relative, suffixes={".json"})
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def svg_path(self, relative: str) -> Path:
        return self._resolve(relative, suffixes={".svg"})

    def write_access_log(self) -> None:
        pd.DataFrame(self.accesses).to_csv(
            self.bundle / "meta" / "plot_source_access.csv",
            index=False,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")


def _write_artifact_manifest(root: Path) -> None:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        root / "artifact_manifest.json",
        {
            "bundle_id": "paper_fig1_fig2_redesign_20260811",
            "plot_version": REDESIGN_PLOT_VERSION,
            "files": files,
        },
    )


def _fig1_spec() -> dict[str, Any]:
    spec: dict[str, Any] = {
        "figure_id": "fig1_model_stsp_intro_candidate",
        "canvas_mm": list(CANVAS_MM),
        "slots": deepcopy(FIG1_SLOTS),
        "reader_contract": {
            "figure_question": (
                "How does the feedforward model implement a facilitating "
                "STSP state that persists after presynaptic firing ceases?"
            ),
            "terminal_inference": (
                "The exact model dynamics convert one presynaptic event "
                "into a transiently elevated u-x state and effective support "
                "after rate returns to zero."
            ),
            "forbidden_inferences": [
                "the deterministic probe is a network-level empirical result",
                "STSP is the only possible working-memory substrate",
                "the mechanism illustration establishes functional "
                "inheritance",
            ],
            "semantic_units": {
                "a": "locate STSP in the feedforward model",
                "b": "define the presynaptic event",
                "c": "show the utilization and resource state variables",
                "d": "show their combined activity-silent state value",
            },
            "task_graph": {
                "nodes": ["a", "b", "c", "d"],
                "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
            },
            "comparison_obligations": [
                {
                    "panels": ["b", "c", "d"],
                    "comparison_basis": "one shared event and time axis",
                    "reader_task": (
                        "track event to state variables to combined state"
                    ),
                }
            ],
            "topology_invariants": [
                "full-width architecture",
                "three aligned mechanism panels",
            ],
            "topology_freedoms": ["optical padding inside each slot"],
        },
        "panels": {
            "a": {
                "claim": (
                    "The fixed feedforward SNN carries STSP on its "
                    "inter-layer synapses"
                ),
                "chart": "svg_asset",
                "source": "data/fig1_panel_a_architecture.svg",
                "role": "locate the modeled synaptic state",
                "legend_owner": "asset",
            },
            "b": {
                "claim": (
                    "A brief presynaptic event ends before the retained "
                    "state evolves"
                ),
                "chart": "event_rate",
                "source": "data/fig1_facilitating_stsp_probe.csv",
                "role": "define the standardized mechanism probe",
                "legend_owner": "none",
            },
            "c": {
                "claim": (
                    "The event increases utilization while resources "
                    "recover quickly"
                ),
                "chart": "stsp_variables",
                "source": "data/fig1_facilitating_stsp_probe.csv",
                "role": "expose the exact u and x dynamics",
                "legend_owner": "panel",
                "legend_ncol": 2,
            },
            "d": {
                "claim": (
                    "The combined u-x state remains elevated after firing "
                    "ceases"
                ),
                "chart": "stsp_state",
                "source": "data/fig1_facilitating_stsp_probe.csv",
                "role": "identify the activity-silent effective state",
                "legend_owner": "none",
            },
        },
    }
    spec["layout_contract"] = build_layout_contract(spec)
    spec["layout_contract"]["status"] = "candidate"
    spec["layout_contract"]["alignment_groups"].append(
        {
            "group_id": "fig1_mechanism_plot_axes",
            "panels": ["b", "c", "d"],
            "target": "plot_area",
            "edges": ["top", "bottom"],
            "rationale": (
                "Align the shared event-relative time axis across the "
                "mechanism row."
            ),
            "comparison_basis": (
                "Identical 48-mm slots and quantitative margins."
            ),
        }
    )
    spec["figure_id"] = "fig1"
    return spec


def _fig2_spec() -> dict[str, Any]:
    current = get_figure_spec("fig1")
    mapping = {"a": "b", "b": "c", "c": "d", "d": "e"}
    panels: dict[str, dict[str, Any]] = {}
    for new_panel, old_panel in mapping.items():
        panel = deepcopy(current["panels"][old_panel])
        panel["source"] = f"data/fig2_panel_{new_panel}_plot_data.csv"
        panels[new_panel] = panel
    panels["a"]["role"] = "establish task-capable networks"
    panels["b"]["role"] = "exclude persistent delay firing"
    panels["c"]["role"] = "establish silent state content"
    panels["d"]["role"] = "establish functional state attribution"

    spec: dict[str, Any] = {
        "figure_id": "fig2_activity_silent_state_candidate",
        "canvas_mm": list(CANVAS_MM),
        "slots": deepcopy(FIG2_SLOTS),
        "reader_contract": {
            "figure_question": (
                "Does STSP retain content after firing ceases and influence "
                "subsequent readout?"
            ),
            "terminal_inference": (
                "Across 20 networks, a decodable activity-silent u-x state "
                "persists and redirects readout attribution when reassigned "
                "between trials."
            ),
            "forbidden_inferences": [
                "high task accuracy alone proves an STSP mechanism",
                "decodability alone establishes functional use",
                "STSP is the only possible biological substrate",
            ],
            "semantic_units": {
                "a": panels["a"]["role"],
                "b": panels["b"]["role"],
                "c": panels["c"]["role"],
                "d": panels["d"]["role"],
            },
            "task_graph": {
                "nodes": ["a", "b", "c", "d"],
                "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
            },
            "comparison_obligations": [],
            "topology_invariants": ["two rows", "two equal slots per row"],
            "topology_freedoms": ["optical padding inside each slot"],
        },
        "panels": panels,
    }
    spec["layout_contract"] = build_layout_contract(spec)
    spec["layout_contract"]["status"] = "candidate"
    spec["layout_contract"]["alignment_groups"].extend(
        [
            {
                "group_id": "fig2_row_1_plot_axes",
                "panels": ["a", "b"],
                "target": "plot_area",
                "edges": ["top", "bottom"],
                "rationale": (
                    "Place the functional premise and firing control on one "
                    "row baseline."
                ),
                "comparison_basis": (
                    "Identical 79.5-mm slots and quantitative margins."
                ),
            },
            {
                "group_id": "fig2_row_2_plot_axes",
                "panels": ["c", "d"],
                "target": "plot_area",
                "edges": ["top", "bottom"],
                "rationale": (
                    "Place silent content and functional attribution on one "
                    "row baseline."
                ),
                "comparison_basis": (
                    "Identical 79.5-mm slots and quantitative margins."
                ),
            },
            {
                "group_id": "fig2_left_column_plot_axes",
                "panels": ["a", "c"],
                "target": "plot_area",
                "edges": ["left", "right"],
                "rationale": "Align the left-column y axes and plot widths.",
                "comparison_basis": "Identical slots and side margins.",
            },
            {
                "group_id": "fig2_right_column_plot_axes",
                "panels": ["b", "d"],
                "target": "plot_area",
                "edges": ["left", "right"],
                "rationale": "Align the right-column y axes and plot widths.",
                "comparison_basis": "Identical slots and side margins.",
            },
        ]
    )
    spec["figure_id"] = "fig2"
    return spec


def _event_band(axis: plt.Axes) -> None:
    axis.axvspan(0.0, 50.0, facecolor=EVENT_TINT, edgecolor="none", zorder=0)


def _time_axis(axis: plt.Axes) -> None:
    axis.set_xlim(-100.0, 1200.0)
    axis.set_xticks([0.0, 400.0, 800.0, 1200.0])
    axis.set_xticklabels(["0", "400", "800", "1200"])
    axis.set_xlabel("Time (ms)")


def _plot_event_rate(axis: plt.Axes, frame: pd.DataFrame) -> None:
    _event_band(axis)
    axis.plot(
        frame["time_ms"].to_numpy(dtype=float),
        frame["presynaptic_rate_hz"].to_numpy(dtype=float),
        color=INK,
        lw=1.45,
        drawstyle="steps-post",
        zorder=3,
    )
    _time_axis(axis)
    axis.set_ylim(0.0, 22.0)
    axis.set_yticks([0.0, 20.0])
    axis.set_yticklabels(["0", "20"])
    axis.set_ylabel("Spike rate (Hz)")


def _plot_stsp_variables(axis: plt.Axes, frame: pd.DataFrame) -> None:
    _event_band(axis)
    time = frame["time_ms"].to_numpy(dtype=float)
    axis.plot(
        time,
        frame["u"].to_numpy(dtype=float),
        color=U_COLOR,
        lw=1.5,
        label=r"$u$",
    )
    axis.plot(
        time,
        frame["x"].to_numpy(dtype=float),
        color=X_COLOR,
        lw=1.5,
        label=r"$x$",
    )
    _time_axis(axis)
    axis.set_ylim(0.0, 1.03)
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.set_yticklabels(["0", "0.5", "1"])
    axis.set_ylabel("State variable")


def _plot_stsp_state(axis: plt.Axes, frame: pd.DataFrame) -> None:
    _event_band(axis)
    time = frame["time_ms"].to_numpy(dtype=float)
    state = frame["stsp_state_value"].to_numpy(dtype=float)
    baseline = float(frame["baseline_state_value"].iloc[0])
    axis.axhline(baseline, color=NEUTRAL, lw=0.8, ls="--", zorder=1)
    axis.plot(time, state, color=STSP_COLOR, lw=1.6, zorder=3)
    _time_axis(axis)
    axis.set_ylim(0.0, 0.4)
    axis.set_yticks([0.0, 0.2, 0.4])
    axis.set_yticklabels(["0", "0.2", "0.4"])
    axis.set_ylabel(r"STSP support ($u x$)")


def _qa_report(
    *,
    bundle: Path,
    spec: Mapping[str, Any],
    png_path: Path,
    pdf_path: Path,
    svg_path: Path,
    plot_bboxes: Mapping[str, Sequence[float]],
    layout_passes: Sequence[str],
) -> dict[str, Any]:
    with Image.open(png_path) as image:
        image_size = (image.width, image.height)
    canvas_width, canvas_height = [float(value) for value in spec["canvas_mm"]]
    expected_pixels = (
        round(canvas_width * MM_TO_INCH * 300),
        round(canvas_height * MM_TO_INCH * 300),
    )
    parser = etree.XMLParser(resolve_entities=False)
    root = etree.parse(str(svg_path), parser).getroot()
    namespace = {"s": "http://www.w3.org/2000/svg"}
    text_count = len(root.xpath(".//s:text", namespaces=namespace))
    rows: list[dict[str, Any]] = []
    for panel_id, slot in spec["slots"].items():
        sx, sy, sw, sh = [float(value) for value in slot]
        px, py, pw, ph = [float(value) for value in plot_bboxes[panel_id]]
        inside = (
            px >= sx
            and py >= sy
            and px + pw <= sx + sw + 1e-9
            and py + ph <= sy + sh + 1e-9
        )
        rows.append(
            {
                "figure_id": spec["figure_id"],
                "panel_id": panel_id,
                "slot_x_mm": sx,
                "slot_y_mm": sy,
                "slot_w_mm": sw,
                "slot_h_mm": sh,
                "plot_x_mm": px,
                "plot_y_mm": py,
                "plot_w_mm": pw,
                "plot_h_mm": ph,
                "plot_inside_slot": inside,
            }
        )
    pd.DataFrame(rows).to_csv(
        bundle / "meta" / f"{spec['figure_id']}_layout_measurements.csv",
        index=False,
    )
    report = {
        "figure_id": spec["figure_id"],
        "plot_version": REDESIGN_PLOT_VERSION,
        "canvas_mm": [canvas_width, canvas_height],
        "png_pixels": list(image_size),
        "expected_png_pixels_at_300dpi": list(expected_pixels),
        "png_size_tolerance_pass": (
            abs(image_size[0] - expected_pixels[0]) <= 3
            and abs(image_size[1] - expected_pixels[1]) <= 3
        ),
        "pdf_bytes": pdf_path.stat().st_size,
        "svg_bytes": svg_path.stat().st_size,
        "svg_text_elements": text_count,
        "editable_text_pass": text_count > 0,
        "all_plot_areas_inside_slots": all(
            row["plot_inside_slot"] for row in rows
        ),
        "layout_passes": list(layout_passes),
        "panel_order": list(spec["panels"]),
    }
    report["status"] = (
        "passed"
        if report["png_size_tolerance_pass"]
        and report["editable_text_pass"]
        and report["all_plot_areas_inside_slots"]
        else "failed"
    )
    _write_json(
        bundle / "meta" / f"{spec['figure_id']}_visual_qa.json", report
    )
    if report["status"] != "passed":
        raise ValueError(f"{spec['figure_id']}: export QA failed: {report}")
    return report


def _render_one(
    *,
    bundle: Path,
    spec: dict[str, Any],
    frames: Mapping[str, pd.DataFrame],
    architecture_svg: Path | None = None,
) -> dict[str, Any]:
    _write_json(bundle / "meta" / f"{spec['figure_id']}_plot_spec.json", spec)
    asset: dict[str, Any] | None = None
    if architecture_svg is not None:
        parser = etree.XMLParser(
            remove_blank_text=False, resolve_entities=False
        )
        architecture_root = etree.parse(
            str(architecture_svg), parser
        ).getroot()
        viewbox = str(architecture_root.get("viewBox") or "")
        if not viewbox:
            raise ValueError(
                f"architecture SVG has no viewBox: {architecture_svg}"
            )
        asset = {
            "panel_id": "a",
            "asset_bytes": architecture_svg.read_bytes(),
            "asset_viewbox": viewbox,
            "embedding_mode": "inline",
            "top_padding_mm": 4.0,
        }
    rendered = render_composed_figure(
        spec=spec,
        frames=frames,
        figure_dir=bundle,
        svg_hashsalt="net_torch_paper_fig1_fig2_redesign_20260811",
        custom_renderers={
            "event_rate": lambda _fig, axis, frame, _panel: _plot_event_rate(
                axis, frame
            ),
            "stsp_variables": lambda _fig, axis, frame, _panel: (
                _plot_stsp_variables(axis, frame)
            ),
            "stsp_state": lambda _fig, axis, frame, _panel: _plot_stsp_state(
                axis, frame
            ),
        },
        svg_asset=asset,
    )
    qa = _qa_report(
        bundle=bundle,
        spec=spec,
        png_path=rendered["png"],
        pdf_path=rendered["pdf"],
        svg_path=rendered["svg"],
        plot_bboxes=rendered["plot_bboxes"],
        layout_passes=rendered["layout_passes"],
    )
    return {
        "figure_id": spec["figure_id"],
        "png": str(rendered["png"]),
        "pdf": str(rendered["pdf"]),
        "svg": str(rendered["svg"]),
        "qa": qa,
    }


def render_paper_fig1_fig2_redesign(*, input_dir: Path) -> dict[str, Any]:
    bundle = input_dir.resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(f"redesign source bundle is missing: {bundle}")
    reader = BundleReader(bundle)
    summary_path = reader._resolve("summary.json", suffixes={".json"})
    reader._resolve("run_config.json", suffixes={".json"})

    stsp = reader.read_csv("data/fig1_facilitating_stsp_probe.csv")
    required_stsp = {
        "time_ms",
        "presynaptic_rate_hz",
        "u",
        "x",
        "stsp_state_value",
        "baseline_state_value",
    }
    missing_stsp = required_stsp.difference(stsp.columns)
    if missing_stsp:
        raise ValueError(
            f"facilitating STSP probe is missing columns: "
            f"{sorted(missing_stsp)}"
        )
    if not np.isfinite(stsp[list(required_stsp)].to_numpy(dtype=float)).all():
        raise ValueError("facilitating STSP probe contains non-finite values")

    fig1_spec = _fig1_spec()
    fig1_frames = {panel_id: stsp for panel_id in ("a", "b", "c", "d")}
    fig1_result = _render_one(
        bundle=bundle,
        spec=fig1_spec,
        frames=fig1_frames,
        architecture_svg=reader.svg_path("data/fig1_panel_a_architecture.svg"),
    )

    fig2_spec = _fig2_spec()
    fig2_frames = {
        panel_id: reader.read_csv(f"data/fig2_panel_{panel_id}_plot_data.csv")
        for panel_id in ("a", "b", "c", "d")
    }
    fig2_result = _render_one(
        bundle=bundle,
        spec=fig2_spec,
        frames=fig2_frames,
    )

    summary = reader.read_json("summary.json")
    summary["status"] = "plot_ready"
    summary["plotting"] = {
        "plot_version": REDESIGN_PLOT_VERSION,
        "figures": {
            "fig1": {
                "png": "figures/fig1.png",
                "pdf": "figures/fig1.pdf",
                "svg": "figures/fig1.svg",
            },
            "fig2": {
                "png": "figures/fig2.png",
                "pdf": "figures/fig2.pdf",
                "svg": "figures/fig2.svg",
            },
        },
        "qa_status": {
            "fig1": fig1_result["qa"]["status"],
            "fig2": fig2_result["qa"]["status"],
        },
    }
    _write_json(summary_path, summary)
    (bundle / "logs" / "plot.log").write_text(
        f"plot_ready version={REDESIGN_PLOT_VERSION}\n", encoding="utf-8"
    )
    reader.write_access_log()
    _write_artifact_manifest(bundle)
    return {
        "status": "plot_ready",
        "input_dir": str(bundle),
        "plot_version": REDESIGN_PLOT_VERSION,
        "figures": {"fig1": fig1_result, "fig2": fig2_result},
    }


__all__ = ["REDESIGN_PLOT_VERSION", "render_paper_fig1_fig2_redesign"]
