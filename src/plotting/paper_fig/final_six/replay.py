from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


FIGURE_IDS = tuple(f"fig{index}" for index in range(1, 7))
REPLAY_VERSION = "final_six_plot_replay_v2.0.0"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_exports(bundle_root: Path) -> dict[str, dict[str, str]]:
    return {
        figure_id: {
            suffix: _sha256(
                bundle_root / figure_id / "figures" / f"{figure_id}.{suffix}"
            )
            for suffix in ("png", "svg", "pdf")
        }
        for figure_id in FIGURE_IDS
    }


def replay(bundle_root: Path) -> dict[str, Any]:
    repo_root = _repo_root().resolve()
    bundle_root = bundle_root.resolve()
    expected_parent = (
        repo_root / "results/paper_figure_multi_seed"
    ).resolve()
    valid_bundle_name = (
        bundle_root.name == "final_six_figures"
        or bundle_root.name.startswith("final_six_figures_")
    )
    if bundle_root.parent != expected_parent or not valid_bundle_name:
        raise ValueError(
            "replay accepts only a canonical or versioned final-six bundle under "
            f"{expected_parent}; got {bundle_root}"
        )
    before = _hash_exports(bundle_root)
    package_dir = Path(__file__).resolve().parent
    removed: list[str] = []
    commands: list[dict[str, Any]] = []
    for figure_id in FIGURE_IDS:
        command = [
            sys.executable,
            "-m",
            f"src.plotting.paper_fig.final_six.{figure_id}_plot",
            "--input-dir",
            str(bundle_root / figure_id),
        ]
        result = subprocess.run(
            command,
            cwd=str(repo_root),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )
        commands.append(
            {
                "figure_id": figure_id,
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{figure_id} plot-only replay failed: {result.stderr[-2000:]}"
            )
    after = _hash_exports(bundle_root)
    deterministic = {
        figure_id: {
            "png_sha256_equal": before[figure_id]["png"] == after[figure_id]["png"],
            "svg_sha256_equal": before[figure_id]["svg"] == after[figure_id]["svg"],
            "pdf_sha256_equal": before[figure_id]["pdf"] == after[figure_id]["pdf"],
        }
        for figure_id in FIGURE_IDS
    }
    status = (
        "pass"
        if all(
            values["png_sha256_equal"] and values["svg_sha256_equal"]
            for values in deterministic.values()
        )
        else "fail"
    )
    report = {
        "schema": "final_six_plot_replay_validation_v1",
        "replay_version": REPLAY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "cache_scope": str(package_dir),
        "cache_directories_removed": removed,
        "cache_policy": "no cache deletion; subprocesses disable bytecode writes",
        "commands": commands,
        "before_sha256": before,
        "after_sha256": after,
        "deterministic_exports": deterministic,
        "consistency_rule": (
            "PNG and SVG must be byte-identical; PDF is dimension/content checked "
            "separately because the browser exporter may write volatile metadata."
        ),
    }
    output = bundle_root / "meta/plot_replay_validation.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if status != "pass":
        raise ValueError(f"plot-only replay was not deterministic: {deterministic}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clear final-six plot caches and replay all six CSV-only figures."
    )
    parser.add_argument(
        "--input-dir",
        default="results/paper_figure_multi_seed/final_six_figures",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    path = Path(args.input_dir)
    if not path.is_absolute():
        path = _repo_root() / path
    report = replay(path)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
