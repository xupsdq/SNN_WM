from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.yaml_loader import load_yaml_file, nested_get

CATALOG_PATH = REPO_ROOT / "src" / "experiments" / "catalog.py"
RUNNERS_DIR = REPO_ROOT / "src" / "experiments" / "runners"
PLOTS_DIR = REPO_ROOT / "src" / "plotting" / "experiments"
EXPERIMENT_CONFIG_DIR = REPO_ROOT / "configs" / "experiment"

IGNORED_RUNNER_MODULES = {"__init__", "_common", "multi_network_batch"}
UTILITY_PLOT_ENTRYPOINTS = {"multi_network_summary"}
MAIN_FOR_RE = re.compile(r"main_for\(\s*['\"]([^'\"]+)['\"]")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _literal_keyword(call: ast.Call, keyword_name: str) -> Any:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            try:
                return ast.literal_eval(keyword.value)
            except Exception:
                return None
    return None


def _catalog_specs() -> dict[str, dict[str, Any]]:
    tree = ast.parse(_read_text(CATALOG_PATH), filename=str(CATALOG_PATH))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            is_specs_node = any(isinstance(target, ast.Name) and target.id == "EXPERIMENT_SPECS" for target in node.targets)
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            is_specs_node = isinstance(node.target, ast.Name) and node.target.id == "EXPERIMENT_SPECS"
            value_node = node.value
        else:
            continue
        if not is_specs_node:
            continue
        if not isinstance(value_node, ast.Dict):
            raise ValueError("EXPERIMENT_SPECS must be a dict literal for entrypoint auditing.")
        specs: dict[str, dict[str, Any]] = {}
        for key_node, item_node in zip(value_node.keys, value_node.values):
            if key_node is None:
                continue
            key = ast.literal_eval(key_node)
            if not isinstance(key, str):
                continue
            payload: dict[str, Any] = {"catalog_key": key}
            if isinstance(item_node, ast.Call):
                payload["experiment_id"] = _literal_keyword(item_node, "experiment_id")
                payload["legacy_module"] = _literal_keyword(item_node, "legacy_module")
            specs[key] = payload
        return specs
    raise ValueError(f"Could not find EXPERIMENT_SPECS in {CATALOG_PATH}")


def _module_path(module_name: str) -> Path:
    parts = module_name.split(".")
    return REPO_ROOT.joinpath(*parts).with_suffix(".py")


def _runner_entrypoints() -> dict[str, dict[str, Any]]:
    runners: dict[str, dict[str, Any]] = {}
    for path in sorted(RUNNERS_DIR.glob("*.py")):
        if path.stem in IGNORED_RUNNER_MODULES:
            continue
        declared = _declared_main_for_id(path)
        runners[path.stem] = {"path": path, "declared_id": declared}
    return runners


def _plot_entrypoints() -> dict[str, dict[str, Any]]:
    plots: dict[str, dict[str, Any]] = {}
    for path in sorted(PLOTS_DIR.glob("*_plot.py")):
        experiment_id = path.stem.removesuffix("_plot")
        declared = _declared_main_for_id(path)
        plots[experiment_id] = {
            "path": path,
            "declared_id": declared,
            "utility": experiment_id in UTILITY_PLOT_ENTRYPOINTS,
        }
    return plots


def _declared_main_for_id(path: Path) -> str | None:
    match = MAIN_FOR_RE.search(_read_text(path))
    return match.group(1) if match else None


def _experiment_configs() -> set[str]:
    return {path.stem for path in EXPERIMENT_CONFIG_DIR.glob("*.yaml")}


def _experiment_config_payloads() -> tuple[dict[str, dict[str, Any]], list[str]]:
    payloads: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for path in sorted(EXPERIMENT_CONFIG_DIR.glob("*.yaml")):
        try:
            payload = dict(load_yaml_file(path))
        except Exception as exc:
            failures.append(f"experiment YAML config is unreadable: {path.name}: {exc}")
            continue
        payloads[path.stem] = payload
    return payloads, failures


def _top_level_path_name(raw_path: Any) -> str | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    return path.parts[0] if path.parts else None


def build_report() -> dict[str, Any]:
    specs = _catalog_specs()
    registered = set(specs)
    runners = _runner_entrypoints()
    runner_ids = set(runners)
    plots = _plot_entrypoints()
    plot_ids = {name for name, meta in plots.items() if not meta["utility"]}
    configs = _experiment_configs()
    config_payloads, config_parse_failures = _experiment_config_payloads()

    failures: list[str] = list(config_parse_failures)
    warnings: list[str] = []
    passes: list[str] = []

    missing_runners = sorted(registered - runner_ids)
    extra_runners = sorted(runner_ids - registered)
    missing_plots = sorted(registered - plot_ids)
    extra_plots = sorted(plot_ids - registered)
    missing_configs = sorted(registered - configs)
    extra_configs = sorted(configs - registered)

    for item in missing_runners:
        failures.append(f"registered experiment has no runner wrapper: {item}")
    for item in extra_runners:
        failures.append(f"runner wrapper is not registered in catalog: {item}")
    for item in missing_plots:
        failures.append(f"registered experiment has no plot wrapper: {item}")
    for item in extra_plots:
        failures.append(f"plot wrapper is not registered in catalog: {item}")

    for name, meta in sorted(runners.items()):
        declared = meta["declared_id"]
        if declared and declared != name:
            failures.append(f"runner wrapper id mismatch: {name}.py calls main_for({declared!r})")
    for name, meta in sorted(plots.items()):
        declared = meta["declared_id"]
        if meta["utility"]:
            warnings.append(f"utility plot entrypoint is intentionally outside catalog: {name}")
            continue
        if declared and declared != name:
            failures.append(f"plot wrapper id mismatch: {name}_plot.py calls main_for({declared!r})")

    for key, payload in sorted(specs.items()):
        experiment_id = payload.get("experiment_id")
        if experiment_id != key:
            failures.append(f"catalog key/id mismatch: key={key!r}, experiment_id={experiment_id!r}")
        legacy_module = payload.get("legacy_module")
        if not legacy_module:
            failures.append(f"{key}: legacy_module missing")
            continue
        if not _module_path(str(legacy_module)).is_file():
            failures.append(f"{key}: legacy_module source missing: {legacy_module}")

    if missing_configs:
        warnings.append("registered experiments without sample YAML configs: " + ", ".join(missing_configs))
    if extra_configs:
        warnings.append("experiment YAML configs without catalog entry: " + ", ".join(extra_configs))

    for name, payload in sorted(config_payloads.items()):
        if name not in registered:
            continue
        experiment_name = payload.get("experiment_name")
        if experiment_name != name:
            failures.append(f"{name}: config experiment_name mismatch: {experiment_name!r}")
        output_dir = payload.get("output_dir")
        if _top_level_path_name(output_dir) != "results":
            failures.append(f"{name}: config output_dir must be under results/: {output_dir!r}")
        dataset_root = nested_get(payload, "data", "dataset_root")
        if not isinstance(dataset_root, str) or not dataset_root.strip():
            failures.append(f"{name}: config data.dataset_root missing")
        model_path = nested_get(payload, "model", "path")
        if not isinstance(model_path, str) or not model_path.strip():
            failures.append(f"{name}: config model.path missing")
        device = nested_get(payload, "runtime", "device")
        if device not in {"auto", "cpu", "cuda"}:
            failures.append(f"{name}: config runtime.device must be one of auto/cpu/cuda: {device!r}")
        plotting_output_dir = nested_get(payload, "plotting", "output_dir")
        if plotting_output_dir is not None and _top_level_path_name(plotting_output_dir) != "results":
            failures.append(f"{name}: config plotting.output_dir must be under results/: {plotting_output_dir!r}")

    if not failures:
        passes.append("catalog, runner wrappers, and plot wrappers are consistent")
    if not extra_configs:
        passes.append("no unregistered experiment YAML configs found")
    if not missing_configs and not extra_configs and not config_parse_failures:
        passes.append("sample YAML configs cover all registered experiments")

    return {
        "ok": not failures,
        "counts": {
            "registered": len(registered),
            "runner_wrappers": len(runner_ids),
            "plot_wrappers": len(plot_ids),
            "utility_plot_entrypoints": sum(1 for meta in plots.values() if meta["utility"]),
            "experiment_configs": len(configs),
        },
        "registered": sorted(registered),
        "runner_wrappers": sorted(runner_ids),
        "plot_wrappers": sorted(plot_ids),
        "utility_plot_entrypoints": sorted(name for name, meta in plots.items() if meta["utility"]),
        "missing_runners": missing_runners,
        "extra_runners": extra_runners,
        "missing_plots": missing_plots,
        "extra_plots": extra_plots,
        "missing_configs": missing_configs,
        "extra_configs": extra_configs,
        "passes": passes,
        "warnings": warnings,
        "failures": failures,
    }


def _print_text_report(report: dict[str, Any]) -> None:
    print("Entrypoint audit")
    print(f"registered={report['counts']['registered']}")
    print(f"runner_wrappers={report['counts']['runner_wrappers']}")
    print(f"plot_wrappers={report['counts']['plot_wrappers']}")
    print(f"utility_plot_entrypoints={report['counts']['utility_plot_entrypoints']}")
    print(f"experiment_configs={report['counts']['experiment_configs']}")
    for message in report["passes"]:
        print(f"PASS: {message}")
    for message in report["warnings"]:
        print(f"WARN: {message}")
    for message in report["failures"]:
        print(f"FAIL: {message}")
    print(f"RESULT: {'PASS' if report['ok'] else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit experiment catalog, runner wrappers, plot wrappers, and sample configs.")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--strict", action="store_true", help="Return a non-zero exit code when failures are present.")
    args = parser.parse_args()

    report = build_report()
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text_report(report)
    return 1 if args.strict and not report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
