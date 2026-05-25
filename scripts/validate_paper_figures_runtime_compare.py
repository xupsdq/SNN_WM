from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ValidationRow:
    fig_id: str
    before_run_dir: str
    after_run_dir: str
    regression_status: str
    layout_status: str
    build_status: str
    regression_seconds: float
    layout_seconds: float
    build_seconds: float
    notes: str = ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_report_dir(root: Path) -> Path:
    if (root / "run_manifest.csv").is_file():
        return root
    batch_root = root / "_batch_runs"
    candidates = [item for item in batch_root.glob("*") if (item / "run_manifest.csv").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No batch report found under {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _successful_runs(report_dir: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in _read_csv(report_dir / "run_manifest.csv"):
        fig_id = str(row.get("fig_id", ""))
        if fig_id and str(row.get("status", "")) == "success":
            out[fig_id] = row
    return out


def _run_command(command: Sequence[str], *, cwd: Path) -> tuple[str, float, str]:
    start = datetime.now(timezone.utc)
    result = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    output = result.stdout or ""
    status = "pass" if result.returncode == 0 else "fail"
    return status, elapsed, output.strip()


def _write_csv(path: Path, rows: Sequence[ValidationRow]) -> None:
    fieldnames = [
        "fig_id",
        "before_run_dir",
        "after_run_dir",
        "regression_status",
        "layout_status",
        "build_status",
        "regression_seconds",
        "layout_seconds",
        "build_seconds",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "fig_id": row.fig_id,
                    "before_run_dir": row.before_run_dir,
                    "after_run_dir": row.after_run_dir,
                    "regression_status": row.regression_status,
                    "layout_status": row.layout_status,
                    "build_status": row.build_status,
                    "regression_seconds": f"{row.regression_seconds:.6f}",
                    "layout_seconds": f"{row.layout_seconds:.6f}",
                    "build_seconds": f"{row.build_seconds:.6f}",
                    "notes": row.notes,
                }
            )


def _write_markdown(path: Path, rows: Sequence[ValidationRow], payload: Mapping[str, Any]) -> None:
    lines = [
        "# Paper Figures Runtime Compare Validation",
        "",
        f"Created: {payload['created_at']}",
        f"Before root: {payload['before_root']}",
        f"After root: {payload['after_root']}",
        "",
        "| Fig | Regression | Layout | Build | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.fig_id} | {row.regression_status} | {row.layout_status} | {row.build_status} | {row.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate serial-vs-optimized paper-figure runtime compare outputs.")
    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd()
    before_root = Path(args.before_root).resolve()
    after_root = Path(args.after_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    before_runs = _successful_runs(_resolve_report_dir(before_root))
    after_runs = _successful_runs(_resolve_report_dir(after_root))
    rows: list[ValidationRow] = []
    all_ok = True

    for fig_id in sorted(set(before_runs) & set(after_runs)):
        before_dir = before_root / before_runs[fig_id]["run_dir"]
        after_dir = after_root / after_runs[fig_id]["run_dir"]
        experiment_root = after_dir.parent
        notes: list[str] = []
        regression_status, regression_seconds, regression_output = _run_command(
            [
                str(args.python),
                "scripts/regression_compare_fig_outputs.py",
                "--old-root",
                str(before_dir),
                "--new-root",
                str(after_dir),
                "--atol",
                str(float(args.atol)),
                "--rtol",
                str(float(args.rtol)),
            ],
            cwd=repo_root,
        )
        layout_status, layout_seconds, layout_output = _run_command(
            [str(args.python), "scripts/validate_results_layout.py", "--input-dir", str(after_dir)],
            cwd=repo_root,
        )
        build_status, build_seconds, build_output = _run_command(
            [
                str(args.python),
                "-m",
                "src.plotting.paper_fig.build",
                "--fig",
                fig_id,
                "--experiment-root",
                str(experiment_root),
                "--check-only",
            ],
            cwd=repo_root,
        )
        for status, label, output in (
            (regression_status, "regression", regression_output),
            (layout_status, "layout", layout_output),
            (build_status, "build", build_output),
        ):
            if status != "pass":
                all_ok = False
                notes.append(f"{label} failed: {output[:200].replace(chr(10), ' ')}")
        rows.append(
            ValidationRow(
                fig_id=fig_id,
                before_run_dir=str(before_dir),
                after_run_dir=str(after_dir),
                regression_status=regression_status,
                layout_status=layout_status,
                build_status=build_status,
                regression_seconds=regression_seconds,
                layout_seconds=layout_seconds,
                build_seconds=build_seconds,
                notes="; ".join(notes),
            )
        )

    missing = sorted((set(before_runs) | set(after_runs)) - (set(before_runs) & set(after_runs)))
    if missing:
        all_ok = False
        for fig_id in missing:
            rows.append(
                ValidationRow(
                    fig_id=fig_id,
                    before_run_dir=str((before_root / before_runs[fig_id]["run_dir"]) if fig_id in before_runs else ""),
                    after_run_dir=str((after_root / after_runs[fig_id]["run_dir"]) if fig_id in after_runs else ""),
                    regression_status="missing",
                    layout_status="missing",
                    build_status="missing",
                    regression_seconds=0.0,
                    layout_seconds=0.0,
                    build_seconds=0.0,
                    notes="fig present in only one arm",
                )
            )

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "before_root": str(before_root),
        "after_root": str(after_root),
        "rows": [row.__dict__ for row in rows],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "validation_by_fig.csv", rows)
    _write_markdown(output_dir / "validation_summary.md", rows, payload)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'validation_by_fig.csv'}")
    print(f"Wrote {output_dir / 'validation_summary.md'}")
    print(f"Wrote {output_dir / 'validation_summary.json'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
