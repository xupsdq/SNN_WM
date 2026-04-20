from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.smoke.registry import get_experiment_spec, get_smoke_spec


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, check=False)


def run_one(experiment_id: str, results_root: Path) -> dict[str, Any]:
    smoke_spec = get_smoke_spec(experiment_id)
    experiment_spec = get_experiment_spec(experiment_id)
    base_dir = _ensure_dir(results_root / experiment_id)
    calc_dir = _ensure_dir(base_dir / "calc")
    plot_dir = _ensure_dir(base_dir / "plot")
    repo_root = Path(__file__).resolve().parents[2]

    calc_cmd = [
        sys.executable,
        "-m",
        smoke_spec.runner_module,
        "--output-dir",
        str(calc_dir),
        "--model-path",
        str((repo_root / "results" / "sdnn_deep_final" / "net_final.pth").resolve()),
        "--dataset-root",
        str((repo_root / "MNIST").resolve()),
        "--device",
        "cpu",
        "--seed",
        "7",
        "--smoke",
    ]
    calc_started = time.time()
    calc_result = _run(calc_cmd, cwd=repo_root)
    calc_ok = calc_result.returncode == 0
    calc_duration = time.time() - calc_started

    plot_cmd = [
        sys.executable,
        "-m",
        smoke_spec.plot_module,
        "--input-dir",
        str(calc_dir),
        "--output-dir",
        str(plot_dir),
    ]
    plot_started = time.time()
    plot_result = _run(plot_cmd, cwd=repo_root) if calc_ok else None
    plot_ok = bool(plot_result is not None and plot_result.returncode == 0)
    plot_duration = time.time() - plot_started if calc_ok else 0.0

    log_lines = [
        f"[experiment] {experiment_id}",
        f"[runner] {subprocess.list2cmdline(calc_cmd)}",
        f"[runner_returncode] {calc_result.returncode}",
        "[runner_stdout]",
        calc_result.stdout.rstrip(),
        "[runner_stderr]",
        calc_result.stderr.rstrip(),
    ]
    if plot_result is not None:
        log_lines.extend(
            [
                f"[plot] {subprocess.list2cmdline(plot_cmd)}",
                f"[plot_returncode] {plot_result.returncode}",
                "[plot_stdout]",
                plot_result.stdout.rstrip(),
                "[plot_stderr]",
                plot_result.stderr.rstrip(),
            ]
        )
    (base_dir / "smoke.log").write_text("\n".join(log_lines).rstrip() + "\n", encoding="utf-8")

    missing_artifacts = [name for name in smoke_spec.expected_artifacts if not (calc_dir / name).exists()]
    missing_plot_files = [name for name in smoke_spec.expected_plot_files if not (plot_dir / name).exists()]
    status = {
        "experiment_id": experiment_id,
        "title": experiment_spec.title,
        "calc_ok": bool(calc_ok and not missing_artifacts),
        "plot_ok": bool(plot_ok and not missing_plot_files),
        "missing_artifacts": missing_artifacts,
        "missing_plot_files": missing_plot_files,
        "calc_duration_sec": round(calc_duration, 3),
        "plot_duration_sec": round(plot_duration, 3),
        "calc_dir": str(calc_dir),
        "plot_dir": str(plot_dir),
        "error": "",
    }
    if not status["calc_ok"]:
        status["error"] = "calc failed or artifacts missing"
    elif not status["plot_ok"]:
        status["error"] = "plot failed or figure outputs missing"
    _write_json(base_dir / "status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Run calc+plot smoke for one experiment.")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--results-root", type=str, default=str(Path(__file__).resolve().parent / "results"))
    args = parser.parse_args()
    run_one(args.experiment, Path(args.results_root).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
