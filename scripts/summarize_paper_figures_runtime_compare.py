from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BatchReport:
    label: str
    root: Path
    report_dir: Path
    summary: dict[str, Any]
    runs: list[dict[str, str]]
    builds: list[dict[str, str]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_report_dir(path: Path) -> Path:
    if (path / "run_manifest.csv").is_file():
        return path
    batch_root = path / "_batch_runs"
    candidates = [item for item in batch_root.glob("*") if (item / "run_manifest.csv").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No batch report found under {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _load_report(root: Path, *, label: str) -> BatchReport:
    report_dir = _resolve_report_dir(root)
    return BatchReport(
        label=label,
        root=root,
        report_dir=report_dir,
        summary=_read_json(report_dir / "summary.json"),
        runs=_read_csv(report_dir / "run_manifest.csv"),
        builds=_read_csv(report_dir / "build_manifest.csv"),
    )


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _unique_join(values: Sequence[Any]) -> str:
    items = [str(value) for value in values if str(value) != ""]
    return ";".join(dict.fromkeys(items))


def _group_runs(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        fig_id = str(row.get("fig_id", ""))
        if not fig_id:
            continue
        item = grouped.setdefault(fig_id, {"seconds": 0.0, "statuses": [], "seeds": [], "profiles": []})
        item["seconds"] += _to_float(row.get("elapsed_seconds"))
        item["statuses"].append(row.get("status", ""))
        item["seeds"].append(row.get("network_seed", ""))
        item["profiles"].append(row.get("benchmark_profile", ""))
    return grouped


def _group_builds(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        fig_id = str(row.get("fig_id", ""))
        if not fig_id:
            continue
        item = grouped.setdefault(fig_id, {"seconds": 0.0, "statuses": [], "build_fig_ids": []})
        item["seconds"] += _to_float(row.get("elapsed_seconds"))
        item["statuses"].append(row.get("status", ""))
        item["build_fig_ids"].append(row.get("build_fig_id", ""))
    return grouped


def _ratio(before: float, after: float) -> str:
    if before <= 0.0 or after <= 0.0:
        return ""
    return f"{before / after:.6f}"


def _seconds(value: float) -> str:
    return f"{float(value):.6f}"


def _runtime_rows(before: BatchReport, after: BatchReport) -> list[dict[str, str]]:
    before_runs = _group_runs(before.runs)
    after_runs = _group_runs(after.runs)
    before_builds = _group_builds(before.builds)
    after_builds = _group_builds(after.builds)
    fig_ids = sorted(set(before_runs) | set(after_runs) | set(before_builds) | set(after_builds))

    rows: list[dict[str, str]] = []
    total_before_run = total_after_run = 0.0
    total_before_build = total_after_build = 0.0
    for fig_id in fig_ids:
        br = before_runs.get(fig_id, {})
        ar = after_runs.get(fig_id, {})
        bb = before_builds.get(fig_id, {})
        ab = after_builds.get(fig_id, {})
        before_run = float(br.get("seconds", 0.0))
        after_run = float(ar.get("seconds", 0.0))
        before_build = float(bb.get("seconds", 0.0))
        after_build = float(ab.get("seconds", 0.0))
        before_total = before_run + before_build
        after_total = after_run + after_build
        total_before_run += before_run
        total_after_run += after_run
        total_before_build += before_build
        total_after_build += after_build
        rows.append(
            {
                "fig_id": fig_id,
                "network_seeds": _unique_join([*br.get("seeds", []), *ar.get("seeds", [])]),
                "before_statuses": _unique_join(br.get("statuses", [])),
                "after_statuses": _unique_join(ar.get("statuses", [])),
                "before_profiles": _unique_join(br.get("profiles", [])),
                "after_profiles": _unique_join(ar.get("profiles", [])),
                "before_run_seconds": _seconds(before_run),
                "after_run_seconds": _seconds(after_run),
                "run_speedup": _ratio(before_run, after_run),
                "run_seconds_saved": _seconds(before_run - after_run),
                "before_build_fig_ids": _unique_join(bb.get("build_fig_ids", [])),
                "after_build_fig_ids": _unique_join(ab.get("build_fig_ids", [])),
                "before_build_statuses": _unique_join(bb.get("statuses", [])),
                "after_build_statuses": _unique_join(ab.get("statuses", [])),
                "before_build_seconds": _seconds(before_build),
                "after_build_seconds": _seconds(after_build),
                "before_total_seconds": _seconds(before_total),
                "after_total_seconds": _seconds(after_total),
                "total_speedup": _ratio(before_total, after_total),
                "total_seconds_saved": _seconds(before_total - after_total),
            }
        )

    before_all = total_before_run + total_before_build
    after_all = total_after_run + total_after_build
    rows.append(
        {
            "fig_id": "TOTAL",
            "network_seeds": "",
            "before_statuses": "",
            "after_statuses": "",
            "before_profiles": "",
            "after_profiles": "",
            "before_run_seconds": _seconds(total_before_run),
            "after_run_seconds": _seconds(total_after_run),
            "run_speedup": _ratio(total_before_run, total_after_run),
            "run_seconds_saved": _seconds(total_before_run - total_after_run),
            "before_build_fig_ids": "",
            "after_build_fig_ids": "",
            "before_build_statuses": "",
            "after_build_statuses": "",
            "before_build_seconds": _seconds(total_before_build),
            "after_build_seconds": _seconds(total_after_build),
            "before_total_seconds": _seconds(before_all),
            "after_total_seconds": _seconds(after_all),
            "total_speedup": _ratio(before_all, after_all),
            "total_seconds_saved": _seconds(before_all - after_all),
        }
    )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    fieldnames = [
        "fig_id",
        "network_seeds",
        "before_statuses",
        "after_statuses",
        "before_profiles",
        "after_profiles",
        "before_run_seconds",
        "after_run_seconds",
        "run_speedup",
        "run_seconds_saved",
        "before_build_fig_ids",
        "after_build_fig_ids",
        "before_build_statuses",
        "after_build_statuses",
        "before_build_seconds",
        "after_build_seconds",
        "before_total_seconds",
        "after_total_seconds",
        "total_speedup",
        "total_seconds_saved",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _fmt_seconds(raw: str) -> str:
    value = _to_float(raw)
    return f"{value:.1f}s"


def _fmt_speedup(raw: str) -> str:
    return f"{float(raw):.2f}x" if raw else "NA"


def _write_markdown(path: Path, *, before: BatchReport, after: BatchReport, rows: Sequence[Mapping[str, str]]) -> None:
    total = rows[-1] if rows else {}
    lines = [
        "# Paper Figures Runtime Compare",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        f"Before label: {before.label}",
        f"After label: {after.label}",
        f"Before report: {before.report_dir}",
        f"After report: {after.report_dir}",
        "",
        "## Batch Context",
        "",
        f"- Before profile: {before.summary.get('benchmark_profile', '')}",
        f"- After profile: {after.summary.get('benchmark_profile', '')}",
        f"- Scope: {after.summary.get('scope', before.summary.get('scope', ''))}",
        f"- Device: {after.summary.get('device', before.summary.get('device', ''))}",
        f"- Seeds: {after.summary.get('network_seeds', before.summary.get('network_seeds', []))}",
        "",
        "## Totals",
        "",
        f"- Run time: {_fmt_seconds(total.get('before_run_seconds', '0'))} -> {_fmt_seconds(total.get('after_run_seconds', '0'))} ({_fmt_speedup(total.get('run_speedup', ''))})",
        f"- Build time: {_fmt_seconds(total.get('before_build_seconds', '0'))} -> {_fmt_seconds(total.get('after_build_seconds', '0'))}",
        f"- End-to-end: {_fmt_seconds(total.get('before_total_seconds', '0'))} -> {_fmt_seconds(total.get('after_total_seconds', '0'))} ({_fmt_speedup(total.get('total_speedup', ''))})",
        "",
        "## By Figure",
        "",
        "| Fig | Before run | After run | Run speedup | Before status | After status |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        if row.get("fig_id") == "TOTAL":
            continue
        lines.append(
            "| {fig} | {before_run} | {after_run} | {speedup} | {before_status} | {after_status} |".format(
                fig=row.get("fig_id", ""),
                before_run=_fmt_seconds(row.get("before_run_seconds", "0")),
                after_run=_fmt_seconds(row.get("after_run_seconds", "0")),
                speedup=_fmt_speedup(row.get("run_speedup", "")),
                before_status=row.get("before_statuses", ""),
                after_status=row.get("after_statuses", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_json(path: Path, *, before: BatchReport, after: BatchReport, rows: Sequence[Mapping[str, str]]) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "before": {"label": before.label, "root": str(before.root), "report_dir": str(before.report_dir), "summary": before.summary},
        "after": {"label": after.label, "root": str(after.root), "report_dir": str(after.report_dir), "summary": after.summary},
        "rows": list(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize before/after paper-figure batch runtime reports.")
    parser.add_argument("--before-root", required=True, help="Before/control output root or exact _batch_runs/<timestamp> report directory.")
    parser.add_argument("--after-root", required=True, help="After/optimized output root or exact _batch_runs/<timestamp> report directory.")
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--after-label", default="after")
    parser.add_argument("--output-dir", default=None, help="Directory for runtime_by_fig.csv and runtime_summary.md. Defaults to <after-root>/_runtime_compare.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    before_root = Path(args.before_root).resolve()
    after_root = Path(args.after_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else after_root / "_runtime_compare"

    before = _load_report(before_root, label=str(args.before_label))
    after = _load_report(after_root, label=str(args.after_label))
    rows = _runtime_rows(before, after)
    _write_csv(output_dir / "runtime_by_fig.csv", rows)
    _write_markdown(output_dir / "runtime_summary.md", before=before, after=after, rows=rows)
    _write_summary_json(output_dir / "runtime_compare.json", before=before, after=after, rows=rows)
    print(f"Wrote {output_dir / 'runtime_by_fig.csv'}")
    print(f"Wrote {output_dir / 'runtime_summary.md'}")
    print(f"Wrote {output_dir / 'runtime_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
