from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from fontTools.pens.recordingPen import RecordingPen
from fontTools.svgLib.path import parse_path
from matplotlib.axes import Axes
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


_ICON_ROOT = Path(__file__).resolve().parent / "assets" / "tabler-icons-v3.46.0"


@lru_cache(maxsize=None)
def _load_tabler_icon(name: str) -> tuple[MplPath, tuple[float, float, float, float]]:
    manifest_path = _ICON_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(record["name"]): record for record in manifest.get("icons", [])
    }
    if name not in records:
        raise ValueError(f"Unknown bundled Tabler icon: {name}")
    record = records[name]
    asset_path = _ICON_ROOT / str(record["file"])
    asset_bytes = asset_path.read_bytes()
    observed_hash = hashlib.sha256(asset_bytes).hexdigest()
    if observed_hash != str(record["sha256"]):
        raise ValueError(f"Bundled Tabler icon hash changed: {asset_path}")

    root = ET.fromstring(asset_bytes)
    view_box = tuple(float(value) for value in root.attrib["viewBox"].split())
    if len(view_box) != 4 or view_box[2] <= 0.0 or view_box[3] <= 0.0:
        raise ValueError(f"Invalid SVG viewBox for {asset_path}: {view_box}")

    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "path":
            continue
        path_data = element.attrib.get("d")
        if not path_data:
            continue
        pen = RecordingPen()
        parse_path(path_data, pen)
        subpath_start: tuple[float, float] | None = None
        for command, points in pen.value:
            if command == "moveTo":
                point = tuple(float(value) for value in points[0])
                vertices.append(point)
                codes.append(MplPath.MOVETO)
                subpath_start = point
            elif command == "lineTo":
                for raw_point in points:
                    vertices.append(tuple(float(value) for value in raw_point))
                    codes.append(MplPath.LINETO)
            elif command == "curveTo":
                for raw_point in points:
                    vertices.append(tuple(float(value) for value in raw_point))
                    codes.append(MplPath.CURVE4)
            elif command == "qCurveTo":
                for raw_point in points:
                    if raw_point is None:
                        continue
                    vertices.append(tuple(float(value) for value in raw_point))
                    codes.append(MplPath.CURVE3)
            elif command == "closePath":
                if subpath_start is None:
                    raise ValueError(f"Closed SVG path has no start point: {asset_path}")
                vertices.append(subpath_start)
                codes.append(MplPath.CLOSEPOLY)
            elif command != "endPath":
                raise ValueError(
                    f"Unsupported SVG pen operation {command!r}: {asset_path}"
                )
    if not vertices:
        raise ValueError(f"Bundled Tabler icon has no drawable paths: {asset_path}")
    return MplPath(vertices, codes), view_box


def draw_tabler_icon(
    axis: Axes,
    name: str,
    bounds: tuple[float, float, float, float] | list[float],
    *,
    color: str,
    linewidth: float = 0.8,
    zorder: int = 6,
) -> PathPatch:
    path, view_box = _load_tabler_icon(name)
    x, y, width, height = (float(value) for value in bounds)
    min_x, min_y, source_width, source_height = view_box
    scale = min(width / source_width, height / source_height)
    drawn_width = source_width * scale
    drawn_height = source_height * scale
    offset_x = x + (width - drawn_width) / 2.0
    offset_y = y + (height - drawn_height) / 2.0
    transformed = [
        (
            offset_x + (vertex_x - min_x) * scale,
            offset_y + (source_height - (vertex_y - min_y)) * scale,
        )
        for vertex_x, vertex_y in path.vertices
    ]
    patch = PathPatch(
        MplPath(transformed, path.codes),
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch
