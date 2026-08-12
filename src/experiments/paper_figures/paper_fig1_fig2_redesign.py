from __future__ import annotations

import hashlib
import inspect
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.core.network import BaseLIFLayer, stsp_dynamics_jit


REDESIGN_VERSION = "paper_fig1_fig2_redesign_source_v1.0.0"
EXPECTED_SEEDS = tuple(range(1000, 1020))
FIG2_PANEL_SOURCES = {
    "a": "panel_b_plot_data.csv",
    "b": "panel_c_plot_data.csv",
    "c": "panel_d_plot_data.csv",
    "d": "panel_e_plot_data.csv",
}
FIG2_STATISTICS_SOURCES = {
    "a": "panel_b_statistics.csv",
    "b": "panel_c_statistics.csv",
    "c": "panel_d_statistics.csv",
    "d": "panel_e_statistics.csv",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
            "source_version": REDESIGN_VERSION,
            "files": files,
        },
    )


def _verified_parent_files(source_bundle: Path) -> dict[str, dict[str, Any]]:
    manifest_path = source_bundle / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing parent artifact manifest: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("figure_id") != "fig1":
        raise ValueError(
            f"expected a canonical Fig.1 parent bundle, "
            f"got {manifest.get('figure_id')!r}"
        )
    records = {
        str(row["path"]): dict(row) for row in manifest.get("files", [])
    }
    required = [f"data/{name}" for name in FIG2_PANEL_SOURCES.values()]
    required.extend(
        f"metrics/{name}" for name in FIG2_STATISTICS_SOURCES.values()
    )
    required.extend(
        [
            "metrics/panel_c_time_bin_validation.csv",
            "metrics/panel_e_composition_audit.csv",
            "meta/panel_a_asset_manifest.csv",
        ]
    )
    for relative in required:
        path = source_bundle / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"required parent artifact is missing: {path}"
            )
        record = records.get(relative)
        if record is None:
            raise ValueError(
                f"required parent artifact is absent from its manifest: "
                f"{relative}"
            )
        observed = _sha256(path)
        expected = str(record.get("sha256") or "")
        if observed != expected:
            raise ValueError(
                f"parent hash mismatch for {relative}: expected {expected}, "
                f"observed {observed}"
            )
    return records


def _stsp_defaults() -> dict[str, float]:
    parameters = inspect.signature(BaseLIFLayer.__init__).parameters
    values = {
        "U": float(parameters["stsp_U"].default),
        "tau_D_s": float(parameters["stsp_tau_D"].default),
        "tau_F_s": float(parameters["stsp_tau_F"].default),
        "dt_s": float(parameters["dt"].default),
    }
    if values != {"U": 0.2, "tau_D_s": 0.1, "tau_F_s": 1.0, "dt_s": 0.001}:
        raise ValueError(f"unexpected mainline STSP defaults: {values}")
    return values


def _calculate_facilitating_probe() -> tuple[pd.DataFrame, dict[str, Any]]:
    parameters = _stsp_defaults()
    dt_ms = parameters["dt_s"] * 1000.0
    if not math.isclose(dt_ms, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"the illustrative probe requires the mainline 1-ms step, "
            f"got {dt_ms}"
        )

    start_ms = -100
    stop_ms = 1200
    event_ms = 0
    rate_bin_ms = 50
    rate_hz = 1000.0 / rate_bin_ms
    decay_x = math.exp(-parameters["dt_s"] / parameters["tau_D_s"])
    decay_u = math.exp(-parameters["dt_s"] / parameters["tau_F_s"])
    u = torch.tensor([parameters["U"]], dtype=torch.float64)
    x = torch.ones(1, dtype=torch.float64)
    rows: list[dict[str, Any]] = []

    for time_ms in range(start_ms, stop_ms + 1):
        input_spike = 1.0 if time_ms == event_ms else 0.0
        spike_tensor = torch.tensor([input_spike], dtype=torch.float64)
        u, x, pre_event_support = stsp_dynamics_jit(
            u,
            x,
            spike_tensor,
            float(parameters["U"]),
            float(decay_x),
            float(decay_u),
        )
        u_value = float(u.item())
        x_value = float(x.item())
        rows.append(
            {
                "time_ms": float(time_ms),
                "presynaptic_rate_hz": (
                    rate_hz
                    if event_ms <= time_ms < event_ms + rate_bin_ms
                    else 0.0
                ),
                "input_spike": int(input_spike),
                "u": u_value,
                "x": x_value,
                "pre_event_support": float(pre_event_support.item()),
                "stsp_state_value": u_value * x_value,
                "baseline_u": parameters["U"],
                "baseline_x": 1.0,
                "baseline_state_value": parameters["U"],
            }
        )

    frame = pd.DataFrame(rows)
    peak_index = frame["stsp_state_value"].idxmax()
    peak_row = frame.loc[peak_index]
    probe = {
        "time_start_ms": start_ms,
        "time_stop_ms": stop_ms,
        "event_time_ms": event_ms,
        "display_rate_bin_ms": rate_bin_ms,
        "display_rate_hz": rate_hz,
        "event_count": 1,
        "U": parameters["U"],
        "tau_D_ms": parameters["tau_D_s"] * 1000.0,
        "tau_F_ms": parameters["tau_F_s"] * 1000.0,
        "dt_ms": dt_ms,
        "peak_state_value": float(peak_row["stsp_state_value"]),
        "peak_state_time_ms": float(peak_row["time_ms"]),
    }
    return frame, probe


def _validate_fig2_frame(frame: pd.DataFrame, panel_id: str) -> None:
    observed_seeds = tuple(
        sorted(
            pd.to_numeric(frame["network_seed"], errors="raise")
            .astype(int)
            .unique()
        )
    )
    if observed_seeds != EXPECTED_SEEDS:
        raise ValueError(
            f"candidate Fig.2{panel_id} requires seeds 1000-1019; "
            f"observed={observed_seeds}"
        )
    if pd.to_numeric(frame["value"], errors="coerce").isna().all():
        raise ValueError(
            f"candidate Fig.2{panel_id} has no finite plotted values"
        )


def _copy_fig2_sources(
    source_bundle: Path, staging: Path
) -> list[dict[str, Any]]:
    source_rows: list[dict[str, Any]] = []
    for new_panel, source_name in FIG2_PANEL_SOURCES.items():
        source = source_bundle / "data" / source_name
        frame = pd.read_csv(source)
        _validate_fig2_frame(frame, new_panel)
        frame["figure_id"] = "fig2_activity_silent_state_candidate"
        frame["panel_id"] = new_panel
        destination = (
            staging / "data" / f"fig2_panel_{new_panel}_plot_data.csv"
        )
        frame.to_csv(destination, index=False)
        source_rows.append(
            {
                "role": f"candidate_fig2_panel_{new_panel}",
                "path": str(source.resolve()),
                "sha256": _sha256(source),
                "bytes": source.stat().st_size,
            }
        )

    for new_panel, source_name in FIG2_STATISTICS_SOURCES.items():
        source = source_bundle / "metrics" / source_name
        frame = pd.read_csv(source)
        if "figure_id" in frame.columns:
            frame["figure_id"] = "fig2_activity_silent_state_candidate"
        if "panel_id" in frame.columns:
            frame["panel_id"] = new_panel
        frame.to_csv(
            staging / "metrics" / f"fig2_panel_{new_panel}_statistics.csv",
            index=False,
        )
        source_rows.append(
            {
                "role": f"candidate_fig2_panel_{new_panel}_statistics",
                "path": str(source.resolve()),
                "sha256": _sha256(source),
                "bytes": source.stat().st_size,
            }
        )

    auxiliary = {
        "metrics/panel_c_time_bin_validation.csv": (
            "metrics/fig2_panel_b_time_bin_validation.csv"
        ),
        "metrics/panel_e_composition_audit.csv": (
            "metrics/fig2_panel_d_composition_audit.csv"
        ),
    }
    for source_relative, destination_relative in auxiliary.items():
        source = source_bundle / source_relative
        frame = pd.read_csv(source)
        if "figure_id" in frame.columns:
            frame["figure_id"] = "fig2_activity_silent_state_candidate"
        if "panel_id" in frame.columns:
            frame["panel_id"] = {"c": "b", "e": "d"}.get(
                str(frame["panel_id"].iloc[0]), frame["panel_id"]
            )
        frame.to_csv(staging / destination_relative, index=False)
        source_rows.append(
            {
                "role": Path(destination_relative).stem,
                "path": str(source.resolve()),
                "sha256": _sha256(source),
                "bytes": source.stat().st_size,
            }
        )
    return source_rows


def build_paper_fig1_fig2_redesign_source_data(
    *,
    output_dir: Path,
    source_bundle: Path,
    command: str,
) -> dict[str, Any]:
    repo_root = _repo_root().resolve()
    source_bundle = source_bundle.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing redesign bundle: {output_dir}"
        )
    if not source_bundle.is_dir():
        raise FileNotFoundError(
            f"canonical Fig.1 source bundle is missing: {source_bundle}"
        )
    _verified_parent_files(source_bundle)

    asset_manifest = pd.read_csv(
        source_bundle / "meta" / "panel_a_asset_manifest.csv"
    )
    if len(asset_manifest) != 1:
        raise ValueError(
            "canonical Fig.1 panel-a asset manifest must contain "
            "exactly one row"
        )
    asset_relative = Path(str(asset_manifest.iloc[0]["asset_path"]))
    architecture_svg = (repo_root / asset_relative).resolve()
    if not architecture_svg.is_file():
        raise FileNotFoundError(
            f"registered architecture SVG is missing: {architecture_svg}"
        )
    expected_asset_hash = str(asset_manifest.iloc[0]["asset_sha256"])
    observed_asset_hash = _sha256(architecture_svg)
    if observed_asset_hash != expected_asset_hash:
        raise ValueError(
            f"architecture SVG hash mismatch: expected {expected_asset_hash}, "
            f"observed {observed_asset_hash}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".paper_fig1_fig2_redesign_", dir=str(output_dir.parent)
        )
    )
    try:
        for relative in ("data", "figures", "logs", "metrics", "meta"):
            (staging / relative).mkdir(parents=True, exist_ok=True)

        stsp_frame, probe = _calculate_facilitating_probe()
        stsp_frame.to_csv(
            staging / "data" / "fig1_facilitating_stsp_probe.csv",
            index=False,
        )
        shutil.copyfile(
            architecture_svg,
            staging / "data" / "fig1_panel_a_architecture.svg",
        )
        source_rows = _copy_fig2_sources(source_bundle, staging)
        source_rows.extend(
            [
                {
                    "role": "candidate_fig1_architecture",
                    "path": str(architecture_svg),
                    "sha256": observed_asset_hash,
                    "bytes": architecture_svg.stat().st_size,
                },
                {
                    "role": "stsp_equation_source",
                    "path": str((repo_root / "src/core/network.py").resolve()),
                    "sha256": _sha256(repo_root / "src/core/network.py"),
                    "bytes": (
                        (repo_root / "src/core/network.py").stat().st_size
                    ),
                },
                {
                    "role": "canonical_fig1_parent_manifest",
                    "path": str(
                        (source_bundle / "artifact_manifest.json").resolve()
                    ),
                    "sha256": _sha256(
                        source_bundle / "artifact_manifest.json"
                    ),
                    "bytes": (
                        (source_bundle / "artifact_manifest.json")
                        .stat()
                        .st_size
                    ),
                },
            ]
        )
        pd.DataFrame(source_rows).sort_values(
            ["role", "path"], kind="mergesort"
        ).to_csv(
            staging / "meta" / "source_manifest.csv", index=False
        )

        selected_times = [0, 50, 200, 400, 800, 1200]
        summary_rows = [
            {
                "metric": "baseline_state_value",
                "time_ms": -100.0,
                "value": probe["U"],
                "unit": "u_x",
            },
            {
                "metric": "peak_state_value",
                "time_ms": probe["peak_state_time_ms"],
                "value": probe["peak_state_value"],
                "unit": "u_x",
            },
        ]
        for time_ms in selected_times:
            row = stsp_frame.loc[
                stsp_frame["time_ms"].eq(float(time_ms))
            ].iloc[0]
            summary_rows.append(
                {
                    "metric": "stsp_state_value",
                    "time_ms": float(time_ms),
                    "value": float(row["stsp_state_value"]),
                    "unit": "u_x",
                }
            )
        pd.DataFrame(summary_rows).to_csv(
            staging / "metrics" / "fig1_stsp_probe_summary.csv", index=False
        )

        _write_json(
            staging / "run_config.json",
            {
                "bundle_id": "paper_fig1_fig2_redesign_20260811",
                "source_version": REDESIGN_VERSION,
                "command": command,
                "reuse_artifacts": "require",
                "source_bundle": str(source_bundle),
                "stsp_probe": probe,
                "stsp_source": {
                    "module": "src.core.network",
                    "symbol": "stsp_dynamics_jit",
                    "parameter_defaults": "BaseLIFLayer.__init__",
                },
                "figure_layouts": {
                    "fig1": "1+3 on 165 x 102 mm",
                    "fig2": "2+2 on 165 x 102 mm",
                },
            },
        )
        _write_json(
            staging / "summary.json",
            {
                "bundle_id": "paper_fig1_fig2_redesign_20260811",
                "source_version": REDESIGN_VERSION,
                "status": "source_ready",
                "figures": ["fig1", "fig2"],
                "source_fig2_panels": (
                    "canonical Fig.1 b-e relabelled a-d without "
                    "changing values"
                ),
                "stsp_probe_scope": (
                    "deterministic mechanism illustration; not a "
                    "network-level inferential result"
                ),
            },
        )
        (staging / "logs" / "source_build.log").write_text(
            f"source_ready version={REDESIGN_VERSION}\n", encoding="utf-8"
        )
        _write_artifact_manifest(staging)
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": "source_ready",
        "output_dir": str(output_dir),
        "source_version": REDESIGN_VERSION,
        "figures": ["fig1", "fig2"],
        "stsp_parameters": {
            "U": probe["U"],
            "tau_D_ms": probe["tau_D_ms"],
            "tau_F_ms": probe["tau_F_ms"],
            "dt_ms": probe["dt_ms"],
        },
    }


__all__ = ["REDESIGN_VERSION", "build_paper_fig1_fig2_redesign_source_data"]
