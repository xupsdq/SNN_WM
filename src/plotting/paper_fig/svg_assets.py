from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree

import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.plotting.paper_fig.utils import repo_root_from_here


def parse_svg_viewbox(path: str | Path) -> dict[str, float]:
    """Return SVG viewBox dimensions in source units."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")[:8192]
    match = re.search(r'viewBox\s*=\s*["\']([^"\']+)["\']', text)
    if match:
        values = [float(v) for v in re.split(r"[\s,]+", match.group(1).strip()) if v]
        if len(values) == 4 and values[2] > 0 and values[3] > 0:
            return {"x": values[0], "y": values[1], "width": values[2], "height": values[3]}
    width = _svg_length(text, "width")
    height = _svg_length(text, "height")
    if width and height:
        return {"x": 0.0, "y": 0.0, "width": width, "height": height}
    raise ValueError(f"Cannot parse SVG viewBox or width/height: {path}")


def render_svg_asset_panel(ax, spec: Mapping[str, Any], *, dpi: int = 300) -> None:
    """Draw an external SVG as a contained raster image while preserving aspect ratio."""
    ax.set_axis_off()
    path = resolve_svg_asset_path(spec)
    viewbox = parse_svg_viewbox(path)
    panel = spec.get("position_mm") or spec.get("size_mm") or {}
    panel_w = float(panel.get("w", panel.get("width", 1.0)))
    panel_h = float(panel.get("h", panel.get("height", 1.0)))
    aspect = viewbox["width"] / viewbox["height"]
    box_w, box_h = _contained_size(panel_w, panel_h, aspect)
    png = _rasterize_svg(path, box_w, box_h, dpi=dpi)
    img = mpimg.imread(png)
    frac_w = box_w / panel_w if panel_w else 1.0
    frac_h = box_h / panel_h if panel_h else 1.0
    x0 = (1.0 - frac_w) / 2.0
    y0 = (1.0 - frac_h) / 2.0
    ax.imshow(img, extent=(x0, x0 + frac_w, y0, y0 + frac_h), transform=ax.transAxes, aspect="auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.paper_fig_plot_form = "manual_svg_asset"
    ax.paper_fig_svg_asset = str(path)
    ax.paper_fig_svg_viewbox = viewbox
    ax.paper_fig_svg_aspect_ratio = float(aspect)
    ax.paper_fig_svg_rendered_size_mm = {"w": float(box_w), "h": float(box_h)}
    ax.paper_fig_svg_raster_cache = str(png)


def load_embedded_square_pngs(spec: Mapping[str, Any], *, count: int = 2) -> list[Any]:
    """Load the square digit PNGs embedded in a draw.io SVG asset."""
    path = resolve_svg_asset_path(spec)
    root = ElementTree.parse(path).getroot()
    images: list[Any] = []
    for node in root.iter("{http://www.w3.org/2000/svg}image"):
        width = float(node.get("width", "0"))
        height = float(node.get("height", "0"))
        href = node.get("{http://www.w3.org/1999/xlink}href", "") or node.get("href", "")
        if width < 50 or height < 50 or abs(width - height) > 1 or not href.startswith("data:image/png;base64,"):
            continue
        payload = base64.b64decode(href.split(",", 1)[1])
        images.append(mpimg.imread(BytesIO(payload), format="png"))
    if len(images) != count:
        raise ValueError(f"Expected {count} embedded square PNGs in {path}, found {len(images)}")
    return images


def setup_programmatic_schematic(ax, spec: Mapping[str, Any]) -> tuple[float, float]:
    """Set up a millimeter-coordinate schematic inside its fixed panel slot."""
    size = spec.get("size_mm") or spec.get("position_mm") or {}
    width = float(size.get("w", size.get("width", 50.0)))
    height = float(size.get("h", size.get("height", 50.0)))
    insets = spec.get("schematic_content_insets_mm") or {}
    left = float(insets.get("left", 0.5))
    right = float(insets.get("right", 0.5))
    top = float(insets.get("top", 1.0))
    bottom = float(insets.get("bottom", 1.0))
    ax.set_axis_off()
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.set_aspect("auto")
    ax.paper_fig_schematic_artists = []
    ax.paper_fig_schematic_content_box_mm = {
        "x": left,
        "y": top,
        "w": width - left - right,
        "h": height - top - bottom,
    }
    return width, height


def schematic_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    text: str = "",
    text_color: str = "#253041",
    fontsize: float = 6.4,
    fontweight: str = "semibold",
    linestyle: str | tuple[Any, ...] = "-",
    linewidth: float = 0.8,
    radius: float = 1.4,
    role: str | None = None,
    zorder: float = 2.0,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.12,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    _register_schematic_artist(ax, patch, role)
    if text:
        label = ax.text(
            x + width / 2.0,
            y + height / 2.0,
            text,
            ha="center",
            va="center",
            color=text_color,
            fontsize=fontsize,
            fontweight=fontweight,
            linespacing=1.05,
            zorder=zorder + 1,
        )
        _register_schematic_artist(ax, label, f"{role}_text" if role else None)
    return patch


def schematic_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    fontsize: float = 7.0,
    role: str | None = None,
):
    artist = ax.text(x, y, text, ha="center", va="center", color=color, fontsize=fontsize, fontweight="bold", zorder=6)
    _register_schematic_artist(ax, artist, role)
    return artist


def schematic_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#8B95A3",
    linewidth: float = 0.75,
    role: str | None = None,
):
    artist = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=linewidth,
        color=color,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=4,
    )
    ax.add_patch(artist)
    _register_schematic_artist(ax, artist, role)
    return artist


def schematic_digit(
    ax,
    image,
    x: float,
    y: float,
    size: float,
    *,
    edgecolor: str,
    role: str,
):
    frame = schematic_box(
        ax,
        x,
        y,
        size,
        size,
        facecolor="#050505",
        edgecolor=edgecolor,
        linewidth=1.05,
        radius=1.35,
        role=role,
        zorder=3,
    )
    inset = 0.8
    # The source is a 28 x 28 MNIST-style digit enlarged by an integer
    # factor. Reconstruct its lit pixels as vector rectangles: this preserves
    # the intentional pixelated appearance and avoids raster blur or image
    # compositing artefacts in full multi-panel PDF/SVG exports.
    rgb = np.asarray(image)[..., :3]
    target_cells = 28
    height, width = rgb.shape[:2]
    if height % target_cells == 0 and width % target_cells == 0:
        block_h = height // target_cells
        block_w = width // target_cells
        pixels = rgb.reshape(
            target_cells,
            block_h,
            target_cells,
            block_w,
            3,
        ).mean(axis=(1, 3))
    else:
        rows = np.linspace(0, height - 1, target_cells).round().astype(int)
        cols = np.linspace(0, width - 1, target_cells).round().astype(int)
        pixels = rgb[np.ix_(rows, cols)]

    inner_size = size - 2.0 * inset
    cell = inner_size / target_cells
    for row in range(target_cells):
        for col in range(target_cells):
            color = pixels[row, col]
            if float(np.max(color)) <= 0.02:
                continue
            ax.add_patch(
                Rectangle(
                    (
                        x + inset + col * cell,
                        y + inset + (target_cells - 1 - row) * cell,
                    ),
                    cell * 1.01,
                    cell * 1.01,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=3.2,
                )
            )
    return frame


def _register_schematic_artist(ax, artist, role: str | None) -> None:
    if role:
        artist.paper_fig_role = role
    ax.paper_fig_schematic_artists.append(artist)


def resolve_svg_asset_path(spec: Mapping[str, Any]) -> Path:
    raw = spec.get("source") or (spec.get("source_mapping") or {}).get("manual_asset")
    if not raw:
        raise ValueError(f"{spec.get('figure_id', '?')}{spec.get('panel_id', '?')}: SVG source missing")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    root = repo_root_from_here()
    for candidate in (root / path, Path(__file__).resolve().parent / path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(root / path)


def _contained_size(panel_w: float, panel_h: float, aspect: float) -> tuple[float, float]:
    if panel_w / panel_h > aspect:
        return panel_h * aspect, panel_h
    return panel_w, panel_w / aspect


def _rasterize_svg(path: Path, width_mm: float, height_mm: float, *, dpi: int) -> Path:
    root = repo_root_from_here()
    cache_dir = root / ".codex" / "tmp" / "paper_fig_svg_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    width_px = max(16, int(round(width_mm / 25.4 * dpi)))
    height_px = max(16, int(round(height_mm / 25.4 * dpi)))
    out = cache_dir / f"{path.stem}_{digest}_{width_px}x{height_px}.png"
    if out.exists():
        return out
    browser = _browser_exe()
    html_path = cache_dir / f"{path.stem}_{digest}_{width_px}x{height_px}.html"
    html_path.write_text(
        "<html><body style='margin:0;background:white;overflow:hidden'>"
        f"<img src='{html.escape(path.as_uri())}' style='width:{width_px}px;height:{height_px}px;object-fit:contain;display:block'>"
        "</body></html>",
        encoding="utf-8",
    )
    user_data_dir = cache_dir / "chrome_profile"
    user_data_dir.mkdir(exist_ok=True)
    result = subprocess.run(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--allow-file-access-from-files",
            f"--user-data-dir={user_data_dir}",
            f"--window-size={width_px},{height_px}",
            f"--screenshot={out}",
            html_path.as_uri(),
        ],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"SVG rasterization failed for {path}: {result.stderr.strip()}")
    return out


def _svg_length(text: str, attr: str) -> float | None:
    match = re.search(rf'{attr}\s*=\s*["\']([0-9.]+)', text)
    return float(match.group(1)) if match else None


def _node_exe() -> Path:
    candidates = []
    if os.environ.get("NODE_EXE"):
        candidates.append(Path(os.environ["NODE_EXE"]))
    candidates.append(Path(r"C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    from shutil import which

    found = which("node")
    if found:
        return Path(found)
    raise RuntimeError("Node.js is required to rasterize SVG assets for paper figures.")


def _bundled_node_modules() -> Path | None:
    path = Path(r"C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules")
    return path if path.exists() else None


def _browser_exe() -> Path:
    candidates = [
        Path(os.environ.get("CHROME_EXE", "")) if os.environ.get("CHROME_EXE") else None,
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists() and candidate.is_file():
            return candidate
    raise RuntimeError("Chrome or Edge is required to rasterize SVG assets for paper figures.")
