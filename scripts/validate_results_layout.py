from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_DIRS = ("data", "figures", "logs", "metrics", "meta")
REQUIRED_RUN_INFO_FIELDS = ("experiment_name", "git_commit", "started_at", "status", "output_dir")
COMPAT_FILES = ("summary.json", "run_config.json", "artifact_manifest.json")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_result_directory(input_dir: Path, *, strict: bool = False) -> dict[str, Any]:
    passes: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []

    if not input_dir.exists():
        failures.append(f"input directory does not exist: {input_dir}")
        return {"passes": passes, "warnings": warnings, "failures": failures}
    if not input_dir.is_dir():
        failures.append(f"input path is not a directory: {input_dir}")
        return {"passes": passes, "warnings": warnings, "failures": failures}

    for name in REQUIRED_DIRS:
        path = input_dir / name
        if path.is_dir():
            passes.append(f"{name} directory exists")
        else:
            failures.append(f"{name} directory missing")

    run_info_path = input_dir / "meta" / "run_info.json"
    run_info_payload: dict[str, Any] | None = None
    if run_info_path.is_file():
        passes.append("meta/run_info.json exists")
        try:
            run_info_payload = _load_json(run_info_path)
        except Exception as exc:
            failures.append(f"meta/run_info.json unreadable: {exc}")
    else:
        failures.append("meta/run_info.json missing")

    if run_info_payload is not None:
        for field in REQUIRED_RUN_INFO_FIELDS:
            if field not in run_info_payload:
                failures.append(f"run_info.json missing field: {field}")
                continue
            value = run_info_payload.get(field)
            if field != "git_commit" and (value is None or value == ""):
                failures.append(f"run_info.json field empty: {field}")
            else:
                passes.append(f"run_info.json field present: {field}")

    for filename in COMPAT_FILES:
        compat_path = input_dir / filename
        if compat_path.is_file():
            passes.append(f"compatibility file exists: {filename}")
        else:
            message = f"compatibility file missing: {filename}"
            if strict:
                failures.append(message)
            else:
                warnings.append(message)

    return {"passes": passes, "warnings": warnings, "failures": failures}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a normalized experiment results directory.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    input_dir = Path(args.input_dir).resolve()
    report = validate_result_directory(input_dir, strict=bool(args.strict))
    ok = not report["failures"]

    if args.json_output:
        print(
            json.dumps(
                {
                    "input_dir": str(input_dir),
                    "ok": ok,
                    "strict": bool(args.strict),
                    **report,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        for message in report["passes"]:
            print(f"PASS: {message}")
        for message in report["warnings"]:
            print(f"WARN: {message}")
        for message in report["failures"]:
            print(f"FAIL: {message}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
