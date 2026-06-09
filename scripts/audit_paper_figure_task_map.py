from __future__ import annotations

import argparse
import ast
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FIGS = [f"fig{i}" for i in range(1, 7)]
GENERATED_DATE = "2026-05-27"

FILE_RE = re.compile(r"[\"']([^\"']+\.(?:csv|json|npz|npy|pkl|parquet|yaml|yml|md|txt))[\"']")
WRITE_KW = (
    "_save_csv",
    "save_csv_with_registry",
    "write_json",
    "_write_json",
    "write_json_file",
    "np.save",
    "savez",
    "to_csv",
    "write_artifact_manifest",
    "write_text",
    "write_adapter_outputs",
    "write_empty_csv",
    "copy_csv_alias",
    "ctx.output_files",
)
READ_KW = (
    "pd.read_csv",
    "read_csv",
    "np.load",
    "read_json",
    "_read_csv_if_exists",
    "_read_required",
    "first_existing",
    "source_priority",
    "source:",
    "required_outputs",
    "legacy_alias_outputs",
)
RISK_KW = (
    "state_bank",
    "sequence_bank",
    "boundary",
    "snapshot",
    "stsp",
    "rollout",
    "trace",
    "spikes",
    "v_mem",
    "encoded",
    "overlap",
    "reentry",
)
REUSE_KW = (
    "state_bank",
    "sequence_bank",
    "boundary_state",
    "encoded",
    "spike_lookup",
    "rollout_trace",
    "rollout",
    "boundary",
)
ALLOWED_TASK_STATUS = {
    "canonical-candidate",
    "proxy-to-legacy",
    "subexperiment-wrapper",
    "missing-file-or-legacy-only",
}


def md(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|")
    return text if text else "-"


def list_text(items: list[str] | tuple[str, ...] | set[str], limit: int = 6) -> str:
    values = [str(item) for item in dict.fromkeys(items) if str(item)]
    if not values:
        return "-"
    if len(values) > limit:
        return "; ".join(values[:limit]) + f"; +{len(values) - limit} more"
    return "; ".join(values)


def rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def exec_registry(path: Path) -> dict[str, Any]:
    ns: dict[str, Any] = {}
    code = path.read_text(encoding="utf-8")
    exec(compile(code, str(path), "exec"), ns, ns)
    return ns


def top_defs(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def normalize_path_from_context(raw: str, context: str) -> str:
    raw = raw.replace("\\", "/")
    if raw.startswith(
        (
            "data/",
            "metrics/",
            "figures/",
            "logs/",
            "meta/",
            "config/",
            "panel_data/",
            "stats/",
            "source_manifests/",
        )
    ):
        return raw
    if "ctx.raw_dir" in context:
        return f"data/raw/{raw}"
    if "ctx.metrics_dir" in context:
        return f"data/metrics/{raw}"
    if "ctx.trial_specs_dir" in context:
        return f"data/trial_specs/{raw}"
    if "ctx.config_dir" in context:
        return f"config/{raw}"
    if "ctx.seed_dir" in context:
        return raw
    if 'output_dir / "panel_data"' in context or 'paths["panel_data"]' in context:
        return f"panel_data/{raw}"
    if 'output_dir / "stats"' in context or 'paths["stats"]' in context:
        return f"stats/{raw}"
    if 'output_dir / "source_manifests"' in context or 'paths["sources"]' in context:
        return f"source_manifests/{raw}"
    return raw


def paths_by_context(text: str) -> tuple[list[str], list[str], list[str]]:
    lines = text.splitlines()
    outs: set[str] = set()
    ins: set[str] = set()
    all_paths: set[str] = set()
    for i, line in enumerate(lines):
        matches = FILE_RE.findall(line)
        if not matches:
            continue
        context = "\n".join(lines[max(0, i - 2) : min(len(lines), i + 3)])
        for match in matches:
            norm = normalize_path_from_context(match, context)
            all_paths.add(norm)
            if any(keyword in context for keyword in WRITE_KW):
                outs.add(norm)
            if any(keyword in context for keyword in READ_KW):
                ins.add(norm)
    return sorted(outs), sorted(ins), sorted(all_paths)


def classify_subexperiment(path: Path | None) -> tuple[str, str, bool, bool, list[str]]:
    if path is None or not path.exists():
        return "missing-file-or-legacy-only", "missing-file-or-legacy-only", False, False, []
    text = path.read_text(encoding="utf-8")
    defs = top_defs(text)
    has_legacy = " as _legacy" in text or "vars(_legacy)" in text
    wrapper = "main_for_current_subexperiment" in text
    if wrapper:
        return "subexperiment-wrapper", "subexperiment-wrapper", True, has_legacy, defs
    if has_legacy:
        detail = "mixed-proxy" if defs else "proxy-to-legacy"
        return "proxy-to-legacy", detail, False, True, defs
    return "canonical-candidate", "canonical-candidate", False, has_legacy, defs


def infer_inputs(text: str, path_inputs: list[str]) -> list[str]:
    inferred = set(path_inputs)
    hints = (
        ("ctx.dataset", "dataset samples"),
        ("ctx.net", "loaded SDNN model"),
        ("ctx.encoder", "encoded input stream"),
        ("model_path", "model checkpoint"),
        ("dataset_root", "dataset root"),
        ("sequence_trials", "sequence trial specs"),
        ("pair_trials", "pair trial specs"),
        ("probe_trials", "probe trials"),
        (" bank", "state/sequence bank object"),
        ("bank.", "state/sequence bank object"),
        ("boundaries", "boundary state"),
        ("boundary", "boundary state"),
        ("pd.read_csv", "CSV artifacts"),
        ("np.load", "NPZ/NPY artifacts"),
        ("read_json", "JSON artifacts"),
        ("encode_images", "encoded input bank"),
        ("_encode_cached", "encoded input bank"),
    )
    for needle, label in hints:
        if needle in text:
            inferred.add(label)
    return sorted(inferred)


def artifact_type(path: str, fig: str = "", task: str = "", context: str = "") -> str:
    stem = Path(path).stem.lower()
    text = f"{path} {fig} {task} {context}".lower()
    if any(token in text for token in ("artifact_manifest", "run_info", "run_config", "source_manifest", "sources.json")):
        return "manifest"
    if "boundary_state" in text or ("boundary" in text and any(token in text for token in ("stsp", "state", "snapshot", "s_final", "s0"))):
        return "stsp_boundary_state"
    if "sequence_bank" in text or "sequence_trials" in text or ("sequence" in stem and "trial" in text):
        return "sequence_bank"
    if "state_bank" in text or "snapshot" in text:
        return "state_bank"
    if "spike_lookup" in text or "encoded" in text or "spike_count_lookup" in text:
        return "encoded_input_bank"
    if any(token in text for token in ("rollout", "trace", "spikes", "v_mem", "state_traces")):
        return "rollout_trace"
    if "panel_data" in text or stem.startswith("panel_") or "/panel_" in text:
        return "panel_table"
    if any(token in text for token in ("metric", "summary", "readout", "audit")) or stem.startswith("supp_"):
        return "readout_metrics"
    if path.endswith((".yaml", ".yml", ".md", ".txt")):
        return "manifest"
    return "readout_metrics"


def artifact_risk(atype: str, path: str, fig: str = "", task: str = "") -> str:
    text = f"{path} {fig} {task}".lower()
    if atype in {"stsp_boundary_state", "state_bank", "rollout_trace"}:
        return "high"
    if any(token in text for token in ("overlap_gated", "stsp", "reentry", "boundary")):
        return "high"
    if atype in {"sequence_bank", "encoded_input_bank"}:
        return "medium"
    return "low"


def human_reason(atype: str, cross: bool, risk: str) -> str:
    if atype == "stsp_boundary_state":
        return "yes: STSP boundary shape/time/phase/seed equivalence must be checked manually"
    if atype == "state_bank":
        return "yes: state bank scientific equivalence and mutation history must be checked manually"
    if atype == "rollout_trace":
        return "yes: rollout traces depend on dynamic state and endpoint definition"
    if cross:
        return "yes: cross-figure reuse requires human confirmation"
    if risk == "high":
        return "yes: STSP/overlap/reentry semantics are high risk"
    return "no"


def cache_candidate(atype: str) -> str:
    if atype in {"state_bank", "stsp_boundary_state", "rollout_trace"}:
        return "conditional"
    if atype in {"sequence_bank", "encoded_input_bank", "readout_metrics", "panel_table"}:
        return "yes"
    return "no"


def load_spec_consumers(plot_root: Path) -> dict[str, set[str]]:
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # type: ignore

    consumer_by_path: dict[str, set[str]] = defaultdict(set)
    for spec_path in sorted((plot_root / "specs").glob("*.yaml")):
        text = spec_path.read_text(encoding="utf-8")
        if yaml is None:
            fig_match = re.search(r"^figure_id:\s*(\S+)", text, re.M)
            fig_id = fig_match.group(1) if fig_match else spec_path.stem
            for source in re.findall(r"data/(?:metrics|raw)/[^,\]\s]+\.(?:csv|json|npz|npy)", text):
                consumer_by_path[source].add(f"plot:{fig_id}")
            continue

        try:
            data = yaml.safe_load(text) or {}
        except Exception:
            data = {}
        fig_id = str(data.get("figure_id") or spec_path.stem)
        for key, label in (("required_outputs", "required_output"), ("legacy_alias_outputs", "legacy_alias")):
            for item in data.get(key) or []:
                consumer_by_path[str(item)].add(f"plot:{fig_id}:{label}")
        panels = data.get("panels") or {}
        for panel_id, panel in panels.items():
            if not isinstance(panel, dict):
                continue
            sources: list[str] = []
            for key in ("source", "source_priority"):
                value = panel.get(key)
                if isinstance(value, list):
                    sources.extend(str(item) for item in value)
                elif isinstance(value, str):
                    sources.append(value)
            for source in sources:
                consumer_by_path[source].add(f"plot:{fig_id}.{panel_id}")
    return consumer_by_path


def load_adapter_consumers(plot_root: Path, consumer_by_path: dict[str, set[str]]) -> None:
    for adapter_path in sorted((plot_root / "adapters").glob("*.py")):
        text = adapter_path.read_text(encoding="utf-8")
        _outputs, inputs, all_paths = paths_by_context(text)
        for path in set(inputs) | {item for item in all_paths if item.startswith("data/")}:
            consumer_by_path[path].add(f"adapter:{adapter_path.stem}")


def scan_tasks(root: Path, base: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    registries: dict[str, dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []
    for fig in FIGS:
        ns = exec_registry(base / fig / "registry.py")
        registries[fig] = ns
        flags: dict[str, tuple[str, ...]] = ns["SUBEXPERIMENT_FLAGS"]
        main = set(ns.get("MAIN_SUBEXPERIMENTS", ()))
        supp = set(ns.get("SUPPLEMENT_SUBEXPERIMENTS", ()))
        for name, flag_values in flags.items():
            path = base / fig / "subexperiments" / f"{name}.py"
            current_file = path if path.exists() else None
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            status, detail, wrapper, has_legacy, defs = classify_subexperiment(current_file)
            outputs, input_paths, _all_paths = paths_by_context(text)
            inputs = infer_inputs(text, input_paths)
            if name in main and name in supp:
                scope = "main+supplement"
            elif name in main:
                scope = "main"
            elif name in supp:
                scope = "supplement"
            else:
                scope = "declared-only"
            reuse_context = " ".join([name, *outputs, *input_paths]).lower()
            risk_context = " ".join([name, *outputs, *input_paths, *inputs]).lower()
            tasks.append(
                {
                    "figure": fig,
                    "experiment_id": ns["EXPERIMENT_ID"],
                    "task_id": f"{fig}.{name}",
                    "task_name": name,
                    "legacy_module": ns["LEGACY_MODULE"],
                    "flags": " ".join(flag_values),
                    "scope": scope,
                    "current_file": rel(root, current_file) if current_file else "-",
                    "current_status": status,
                    "identity_detail": detail,
                    "standalone_cli": "yes" if wrapper else "no",
                    "legacy_dependency": "yes" if has_legacy else "no",
                    "likely_inputs": inputs,
                    "likely_outputs": outputs,
                    "reuse_candidate": "yes" if any(token in reuse_context for token in REUSE_KW) else "no",
                    "human_confirm": "yes" if any(token in risk_context for token in RISK_KW) else "no",
                    "top_level_defs": defs,
                }
            )
    return registries, tasks


def scan_artifacts(
    root: Path,
    base: Path,
    plot_root: Path,
    registries: dict[str, dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    consumer_by_path = load_spec_consumers(plot_root)
    load_adapter_consumers(plot_root, consumer_by_path)

    artifacts: dict[str, dict[str, Any]] = {}
    producer_name_index: dict[str, set[str]] = defaultdict(set)

    def add_artifact(
        *,
        artifact_id: str,
        atype: str,
        producer: str,
        consumers: set[str] | list[str],
        file_pattern: str,
        producer_fig: str,
        producer_task: str,
    ) -> None:
        consumer_set = set(consumers)
        cross = any(c.startswith("plot:") and producer_fig and not c.split(":", 2)[1].startswith(producer_fig) for c in consumer_set)
        if atype in {"state_bank", "sequence_bank", "stsp_boundary_state", "encoded_input_bank"} and producer_fig in {"fig2", "fig3", "fig6"}:
            cross = True
        risk = artifact_risk(atype, file_pattern, producer_fig, producer_task)
        existing = artifacts.get(artifact_id)
        if existing is not None:
            old = set(existing["consumer_task_or_plot"] if existing["consumer_task_or_plot"] != ["-"] else [])
            merged = old | consumer_set
            existing["consumer_task_or_plot"] = sorted(merged) if merged else ["-"]
            if existing["producer_task"] == "-" and producer != "-":
                existing["producer_task"] = producer
            if existing["cross_figure_candidate"] == "no" and cross:
                existing["cross_figure_candidate"] = "yes"
                existing["human_confirm_reason"] = human_reason(atype, True, risk)
            return
        artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "artifact_type": atype,
            "producer_task": producer,
            "consumer_task_or_plot": sorted(consumer_set) if consumer_set else ["-"],
            "file_pattern": file_pattern,
            "cross_figure_candidate": "yes" if cross else "no",
            "cache_candidate": cache_candidate(atype),
            "risk": risk,
            "human_confirm_reason": human_reason(atype, cross, risk),
        }
        producer_name_index[Path(file_pattern).name].add(artifact_id)

    for task in tasks:
        if task["current_file"] == "-":
            continue
        task_path = root / task["current_file"]
        text = task_path.read_text(encoding="utf-8")
        outputs, _inputs, _all_paths = paths_by_context(text)
        for file_pattern in outputs:
            atype = artifact_type(file_pattern, task["figure"], task["task_name"])
            stem = Path(file_pattern).stem.replace("panel_", "p_")
            artifact_id = f"{task['figure']}.{task['task_name']}.{atype}.{stem}"
            consumers = set(consumer_by_path.get(file_pattern, set())) | set(consumer_by_path.get(Path(file_pattern).name, set()))
            add_artifact(
                artifact_id=artifact_id,
                atype=atype,
                producer=task["task_id"],
                consumers=consumers,
                file_pattern=file_pattern,
                producer_fig=task["figure"],
                producer_task=task["task_name"],
            )

    for fig, ns in registries.items():
        module_tail = str(ns["LEGACY_MODULE"]).split(".")[-1]
        legacy_path = base / f"{module_tail}.py"
        if not legacy_path.exists():
            continue
        text = legacy_path.read_text(encoding="utf-8")
        outputs, _inputs, _all_paths = paths_by_context(text)
        for file_pattern in outputs:
            name = Path(file_pattern).name
            if name in producer_name_index:
                continue
            atype = artifact_type(file_pattern, fig, "legacy-backend")
            stem = Path(file_pattern).stem.replace("panel_", "p_")
            consumers = set(consumer_by_path.get(file_pattern, set())) | set(consumer_by_path.get(name, set()))
            add_artifact(
                artifact_id=f"{fig}.legacy_backend.{atype}.{stem}",
                atype=atype,
                producer=f"legacy-backend:{fig}",
                consumers=consumers,
                file_pattern=file_pattern,
                producer_fig=fig,
                producer_task="legacy-backend",
            )

    for file_pattern, consumers in sorted(consumer_by_path.items()):
        matched_ids = set(producer_name_index.get(Path(file_pattern).name, set()))
        if matched_ids:
            for artifact_id in matched_ids:
                existing = artifacts[artifact_id]
                old = set(existing["consumer_task_or_plot"] if existing["consumer_task_or_plot"] != ["-"] else [])
                existing["consumer_task_or_plot"] = sorted(old | consumers)
            continue
        fig_guess = "unknown"
        for consumer in sorted(consumers):
            if consumer.startswith("plot:"):
                fig_guess = consumer.split(":", 2)[1].split(".", 1)[0]
                break
        atype = artifact_type(file_pattern, fig_guess)
        stem = Path(file_pattern).stem.replace("panel_", "p_")
        add_artifact(
            artifact_id=f"plot.{fig_guess}.{atype}.{stem}",
            atype=atype,
            producer="-",
            consumers=consumers,
            file_pattern=file_pattern,
            producer_fig="",
            producer_task="consumer-only",
        )

    return artifacts


def inventory_files(root: Path, base: Path) -> dict[str, list[str]]:
    legacy_import_files: list[str] = []
    wrapper_files: list[str] = []
    for path in sorted(base.glob("fig*/subexperiments/*.py")):
        text = path.read_text(encoding="utf-8")
        if " as _legacy" in text or "vars(_legacy)" in text:
            legacy_import_files.append(rel(root, path))
        if "main_for_current_subexperiment" in text:
            wrapper_files.append(rel(root, path))
    return {
        "legacy_import_files": legacy_import_files,
        "wrapper_files": wrapper_files,
        "thin_wrappers": [rel(root, base / fig / "run.py") for fig in FIGS if (base / fig / "run.py").exists()],
        "registry_files": [rel(root, base / fig / "registry.py") for fig in FIGS],
        "common_files": [rel(root, path) for path in sorted((base / "common").glob("*.py"))],
        "optimized_patch_files": [rel(root, path) for path in sorted((base / "optimized_issue_patch").glob("*.py"))],
    }


def render_task_doc(
    root: Path,
    base: Path,
    registries: dict[str, dict[str, Any]],
    tasks: list[dict[str, Any]],
    files: dict[str, list[str]],
) -> str:
    status_counts = Counter(task["current_status"] for task in tasks)
    detail_counts = Counter(task["identity_detail"] for task in tasks)
    missing_tasks = [task for task in tasks if task["current_status"] == "missing-file-or-legacy-only"]
    registry_parse_ok = all({"FIGURE_ID", "EXPERIMENT_ID", "LEGACY_MODULE", "SUBEXPERIMENT_FLAGS"}.issubset(registries[fig].keys()) for fig in FIGS)
    all_status_ok = all(task["current_status"] in ALLOWED_TASK_STATUS for task in tasks)
    legacy_backend_files = [rel(root, base / (str(registries[fig]["LEGACY_MODULE"]).split(".")[-1] + ".py")) for fig in FIGS]

    lines: list[str] = [
        "# Paper Figure Task Map",
        "",
        f"Generated: {GENERATED_DATE}",
        "",
        "Scope: this map covers only `src/experiments/paper_figures/fig1` through `fig6` and `src/plotting/paper_fig`. Root-level `src/experiments/*.py` files are intentionally excluded from this phase.",
        "",
        "## Method",
        "",
        "- Task truth source: `figN/registry.py` for `FIGURE_ID`, `EXPERIMENT_ID`, `LEGACY_MODULE`, `SUBEXPERIMENT_FLAGS`, and main/supplement scope.",
        "- `current_status` is limited to `canonical-candidate`, `proxy-to-legacy`, `subexperiment-wrapper`, and `missing-file-or-legacy-only` so every registry task has a stable status.",
        "- `identity_detail=mixed-proxy` marks files that import `_legacy` but also contain local compute/serialization logic. These are not considered canonical until manually reviewed.",
        "- `optimized_issue_patch` is retained as historical patch/provenance and is not part of the main task axis.",
        "",
        "## Validation Summary",
        "",
        f"- Registry parse: {'PASS' if registry_parse_ok else 'FAIL'} ({len(registries)}/6 registries).",
        f"- Registry task count: {len(tasks)}.",
        f"- Task status coverage: {'PASS' if all_status_ok else 'FAIL'}.",
        f"- Status counts: {', '.join(f'{key}={value}' for key, value in sorted(status_counts.items()))}.",
        f"- Identity detail counts: {', '.join(f'{key}={value}' for key, value in sorted(detail_counts.items()))}.",
        f"- `_legacy` import files: {len(files['legacy_import_files'])}.",
        f"- `main_for_current_subexperiment` wrapper files: {len(files['wrapper_files'])}.",
        f"- Missing current task files: {list_text([task['task_id'] for task in missing_tasks], limit=12)}.",
        "",
        "## File Identity Inventory",
        "",
        "### Thin Wrappers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in files["thin_wrappers"])
    lines.extend(["", "### Registries", ""])
    lines.extend(f"- `{item}`" for item in files["registry_files"])
    lines.extend(["", "### Legacy Backends", ""])
    lines.extend(f"- `{item}`" for item in legacy_backend_files)
    lines.extend(["", "### Shared Common", ""])
    lines.extend(f"- `{item}`" for item in files["common_files"])
    lines.extend(["", "### Optimized Patch / Provenance Only", ""])
    lines.extend(f"- `{item}`" for item in files["optimized_patch_files"])
    lines.extend(
        [
            "",
            "## Task Table",
            "",
            "| Figure | Task | Scope | Flags | Current status | Identity detail | Current file | Legacy dependency | Standalone CLI | Likely inputs | Likely outputs | Reuse candidate | Human confirm |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for task in tasks:
        values = (
            task["figure"],
            task["task_id"],
            task["scope"],
            task["flags"],
            task["current_status"],
            task["identity_detail"],
            task["current_file"],
            task["legacy_dependency"],
            task["standalone_cli"],
            list_text(task["likely_inputs"], limit=5),
            list_text(task["likely_outputs"], limit=5),
            task["reuse_candidate"],
            task["human_confirm"],
        )
        lines.append("| " + " | ".join(md(value) for value in values) + " |")
    lines.extend(["", "## `_legacy` Import Files", ""])
    lines.extend(f"- `{item}`" for item in files["legacy_import_files"])
    lines.extend(["", "## Standalone Subexperiment Wrapper Files", ""])
    lines.extend(f"- `{item}`" for item in files["wrapper_files"])
    lines.extend(["", "## Human Confirmation Queue", ""])
    queue = [task for task in tasks if task["human_confirm"] == "yes" or task["reuse_candidate"] == "yes" or task["identity_detail"] == "mixed-proxy"]
    for task in queue:
        reasons: list[str] = []
        if task["identity_detail"] == "mixed-proxy":
            reasons.append("legacy namespace plus local logic")
        if task["reuse_candidate"] == "yes":
            reasons.append("named reusable artifact candidate")
        if task["human_confirm"] == "yes":
            reasons.append("state/sequence/rollout/encoded/STSP-related static signal")
        lines.append(f"- `{task['task_id']}`: {', '.join(dict.fromkeys(reasons))}.")
    lines.extend(
        [
            "",
            "## Migration Priority",
            "",
            "1. Registry / task map / artifact map documentation.",
            "2. Pure JSON/config/summary helpers.",
            "3. Pure artifact writer/reader helpers.",
            "4. Pure metric helpers.",
            "5. Input/image/sequence bank helpers.",
            "6. State bank serialization.",
            "7. Rollout wrappers.",
            "8. Simulation inner loop / STSP update.",
            "",
            "Root-level `src/experiments/*.py` cleanup remains a separate archive-candidate workflow and is not mixed into this map.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_artifact_doc(artifacts: dict[str, dict[str, Any]]) -> str:
    items = sorted(
        artifacts.values(),
        key=lambda item: (
            0 if item["cross_figure_candidate"] == "yes" else 1,
            {"high": 0, "medium": 1, "low": 2}.get(item["risk"], 3),
            item["artifact_id"],
        ),
    )
    high_value = [
        item
        for item in items
        if item["cross_figure_candidate"] == "yes"
        or item["risk"] == "high"
        or item["artifact_type"] in {"state_bank", "sequence_bank", "stsp_boundary_state", "encoded_input_bank", "rollout_trace"}
    ]
    endpoint_ok = all(item["producer_task"] != "-" or item["consumer_task_or_plot"] != ["-"] for item in items)
    cross_ok = all(item["human_confirm_reason"].startswith("yes") for item in items if item["cross_figure_candidate"] == "yes")
    type_counts = Counter(item["artifact_type"] for item in items)
    risk_counts = Counter(item["risk"] for item in items)

    lines: list[str] = [
        "# Paper Figure Artifact Map",
        "",
        f"Generated: {GENERATED_DATE}",
        "",
        "Scope: this artifact map is inferred from `src/experiments/paper_figures/fig*/subexperiments/*.py`, paper-figure legacy backends, `paper_figures/common/bundle_io.py`, and `src/plotting/paper_fig` adapters/specs. Root-level `src/experiments/*.py` files are intentionally excluded.",
        "",
        "## Validation Summary",
        "",
        f"- Artifact entries: {len(items)}.",
        f"- Artifact endpoint coverage: {'PASS' if endpoint_ok else 'FAIL'}; every entry has a producer or consumer.",
        f"- Cross-figure confirmation gate: {'PASS' if cross_ok else 'FAIL'}; every `cross_figure_candidate=yes` entry has `human_confirm_reason` starting with `yes`.",
        f"- Type counts: {', '.join(f'{key}={value}' for key, value in sorted(type_counts.items()))}.",
        f"- Risk counts: {', '.join(f'{key}={value}' for key, value in sorted(risk_counts.items()))}.",
        f"- Consumer-only plot/spec entries: {sum(1 for item in items if item['producer_task'] == '-')}.",
        "",
        "## Artifact Type Rules",
        "",
        "| Artifact type | Static signal | Default handling |",
        "|---|---|---|",
        "| sequence_bank | `sequence_bank`, `sequence_trials` | Cache candidate; cross-figure use needs confirmation. |",
        "| state_bank | `state_bank`, `snapshot` | High-risk; requires state equivalence review. |",
        "| stsp_boundary_state | `boundary`, `S0`, `S_final`, STSP state names | Highest-risk; shape/time/phase/seed/model equivalence must be checked manually. |",
        "| encoded_input_bank | `encoded`, `spike_lookup` | Cache candidate; validate encoding config/model/dataset hash. |",
        "| rollout_trace | `rollout`, `trace`, `spikes`, `v_mem` | High-risk large object; prefer summary artifacts unless trace identity is required. |",
        "| readout_metrics | `metrics`, `summary`, `readout`, `audit`, `supp_*` | Usually reusable if provenance/config is stable. |",
        "| panel_table | panel CSV/spec/adapter output | Plot-only input/output; should not trigger compute. |",
        "| manifest | manifest/config/run metadata | Provenance required; not a scientific cache by itself. |",
        "",
        "## High-Risk / Reuse Confirmation Queue",
        "",
        "| Artifact ID | Type | Producer | Consumers | File pattern | Cross figure | Cache | Risk | Confirmation reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in high_value:
        values = (
            item["artifact_id"],
            item["artifact_type"],
            item["producer_task"],
            list_text(item["consumer_task_or_plot"], limit=6),
            item["file_pattern"],
            item["cross_figure_candidate"],
            item["cache_candidate"],
            item["risk"],
            item["human_confirm_reason"],
        )
        lines.append("| " + " | ".join(md(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Full Static Artifact Index",
            "",
            "| Artifact ID | Type | Producer task | Consumer task or plot | File pattern | Cross figure candidate | Cache candidate | Risk | Human confirm reason |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in items:
        values = (
            item["artifact_id"],
            item["artifact_type"],
            item["producer_task"],
            list_text(item["consumer_task_or_plot"], limit=6),
            item["file_pattern"],
            item["cross_figure_candidate"],
            item["cache_candidate"],
            item["risk"],
            item["human_confirm_reason"],
        )
        lines.append("| " + " | ".join(md(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Immediate Manual Checks",
            "",
            "- Confirm whether Fig.3 `state_bank` artifacts can become shared artifacts or must remain figure-local.",
            "- Confirm whether Fig.6 `sequence_bank` / `final_support_maps` / `update_history_matrix` can consume any Fig.3 state or sequence artifact without changing scientific meaning.",
            "- For every STSP boundary/state artifact, check shape, time index, phase, seed, model checkpoint, dataset split, and mutation history before reuse.",
            "- Treat Fig.4/Fig.6 rollout traces as high-risk until endpoint definitions and dynamic-state reset behavior are reviewed.",
            "- Keep plot adapters plot-only: they may consume `panel_table` and `readout_metrics`, but should not import compute loops or legacy experiment modules.",
            "",
            "Root-level `src/experiments/*.py` files are still outside this phase; archive handling should start with a separate candidate list and human approval.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_outputs(root: Path) -> tuple[str, str, dict[str, Any]]:
    base = root / "src" / "experiments" / "paper_figures"
    plot_root = root / "src" / "plotting" / "paper_fig"
    registries, tasks = scan_tasks(root, base)
    artifacts = scan_artifacts(root, base, plot_root, registries, tasks)
    files = inventory_files(root, base)
    summary = {
        "registry_count": len(registries),
        "task_count": len(tasks),
        "artifact_count": len(artifacts),
        "legacy_import_files": len(files["legacy_import_files"]),
        "wrapper_files": len(files["wrapper_files"]),
        "registry_parse_ok": all({"FIGURE_ID", "EXPERIMENT_ID", "LEGACY_MODULE", "SUBEXPERIMENT_FLAGS"}.issubset(registries[fig].keys()) for fig in FIGS),
        "task_status_coverage_ok": all(task["current_status"] in ALLOWED_TASK_STATUS for task in tasks),
        "artifact_endpoint_ok": all(item["producer_task"] != "-" or item["consumer_task_or_plot"] != ["-"] for item in artifacts.values()),
        "cross_confirm_ok": all(item["human_confirm_reason"].startswith("yes") for item in artifacts.values() if item["cross_figure_candidate"] == "yes"),
        "missing_tasks": [task["task_id"] for task in tasks if task["current_status"] == "missing-file-or-legacy-only"],
    }
    return render_task_doc(root, base, registries, tasks, files), render_artifact_doc(artifacts), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit paper figure task and artifact maps without running experiments.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--write-docs", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    task_doc, artifact_doc, summary = build_outputs(root)
    if args.write_docs:
        out_dir = root / "docs" / "paper_figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "task_map.md").write_text(task_doc, encoding="utf-8")
        (out_dir / "artifact_map.md").write_text(artifact_doc, encoding="utf-8")
        print("wrote docs/paper_figures/task_map.md")
        print("wrote docs/paper_figures/artifact_map.md")

    print(
        " ".join(
            [
                f"registries={summary['registry_count']}",
                f"tasks={summary['task_count']}",
                f"artifacts={summary['artifact_count']}",
                f"legacy_import_files={summary['legacy_import_files']}",
                f"wrappers={summary['wrapper_files']}",
            ]
        )
    )
    print(
        " ".join(
            [
                f"registry_parse={'PASS' if summary['registry_parse_ok'] else 'FAIL'}",
                f"task_status_coverage={'PASS' if summary['task_status_coverage_ok'] else 'FAIL'}",
                f"artifact_endpoint={'PASS' if summary['artifact_endpoint_ok'] else 'FAIL'}",
                f"cross_confirm={'PASS' if summary['cross_confirm_ok'] else 'FAIL'}",
            ]
        )
    )
    if summary["missing_tasks"]:
        print("missing_tasks=" + ",".join(summary["missing_tasks"]))
    return 0 if all(summary[key] for key in ("registry_parse_ok", "task_status_coverage_ok", "artifact_endpoint_ok", "cross_confirm_ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
