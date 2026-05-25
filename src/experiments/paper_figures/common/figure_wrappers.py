from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.experiments.paper_figures.common.registry import load_registry_module
from src.experiments.paper_figures.run_paper_figures import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH_GLOB,
    DEFAULT_OUTPUT_ROOT,
    discover_checkpoints,
    main as batch_main,
)


SEED_DIR_RE = re.compile(r"^seed[_-]?(\d+)$", re.IGNORECASE)


def _registry(fig_id: str) -> Any:
    return load_registry_module(fig_id)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (DEFAULT_PROJECT_DEFAULTS.paths.repo_root / path).resolve()


def _infer_seed_from_dir(path: Path) -> int | None:
    match = SEED_DIR_RE.match(path.name)
    return int(match.group(1)) if match else None


def _resolve_model_path(model_path: str | None, model_path_glob: str, network_seed: int) -> Path:
    if model_path:
        return _resolve_repo_path(model_path)
    checkpoints = discover_checkpoints(model_path_glob)
    by_seed = {int(item.seed): item.model_path for item in checkpoints}
    if int(network_seed) not in by_seed:
        known = ", ".join(str(seed) for seed in sorted(by_seed))
        raise SystemExit(f"No checkpoint for network seed {network_seed} matched --model-path-glob. Known seeds: {known}")
    return by_seed[int(network_seed)]


def _subexperiment_parser(fig_id: str, name: str, registry: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run paper-figure sub-experiment {fig_id}.{name}. Unknown options are forwarded to the figure implementation.",
        allow_abbrev=False,
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Exact seed directory, e.g. results/paper_experiments/<figure>/seed_1000.")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Batch output root used when --output-dir/--figure-root are omitted.")
    parser.add_argument("--figure-root", type=str, default=None, help="Exact figure root passed to the legacy figure implementation.")
    parser.add_argument("--network-seed", type=int, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-path-glob", type=str, default=DEFAULT_MODEL_PATH_GLOB)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_PROJECT_DEFAULTS.runtime.device)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the delegated legacy command and exit.")
    choices = ", ".join(sorted(registry.SUBEXPERIMENT_FLAGS))
    parser.epilog = f"Available sub-experiments for {fig_id}: {choices}"
    return parser


def _legacy_output_root(args: argparse.Namespace, registry: Any, parser: argparse.ArgumentParser) -> Path:
    if args.output_dir and args.figure_root:
        parser.error("--output-dir and --figure-root are mutually exclusive.")
    if args.output_dir:
        output_dir = _resolve_repo_path(args.output_dir)
        if _infer_seed_from_dir(output_dir) is None:
            parser.error("--output-dir must end in seed_XXXX; use --figure-root for a figure-level directory.")
        return output_dir
    if args.figure_root:
        return _resolve_repo_path(args.figure_root)
    return _resolve_repo_path(args.output_root) / str(registry.EXPERIMENT_ID)


def _network_seed(args: argparse.Namespace) -> int:
    if args.network_seed is not None:
        return int(args.network_seed)
    if args.output_dir:
        inferred = _infer_seed_from_dir(Path(args.output_dir))
        if inferred is not None:
            return int(inferred)
    return 1000


def _build_legacy_args(
    *,
    fig_id: str,
    name: str,
    args: argparse.Namespace,
    extra_args: Sequence[str],
    registry: Any,
    parser: argparse.ArgumentParser,
) -> list[str]:
    flags_by_name: Mapping[str, Sequence[str]] = registry.SUBEXPERIMENT_FLAGS
    if name not in flags_by_name:
        choices = ", ".join(sorted(flags_by_name))
        raise SystemExit(f"Unknown sub-experiment {fig_id}.{name}. Available: {choices}")
    network_seed = _network_seed(args)
    output_root = _legacy_output_root(args, registry, parser)
    model_path = _resolve_model_path(args.model_path, str(args.model_path_glob), network_seed)
    legacy_args = [
        "--model-path",
        str(model_path),
        "--dataset-root",
        str(_resolve_repo_path(args.dataset_root)),
        "--output-root",
        str(output_root),
        "--network-seed",
        str(int(network_seed)),
        "--device",
        str(args.device),
        "--split",
        str(args.split),
        *[str(flag) for flag in flags_by_name[name]],
    ]
    if args.smoke:
        legacy_args.append("--smoke")
    if args.save_debug_figures:
        legacy_args.append("--save-debug-figures")
    if args.no_progress:
        legacy_args.append("--no-progress")
    legacy_args.extend(str(item) for item in extra_args)
    return legacy_args


def main_for_figure(fig_id: str, argv: Sequence[str] | None = None) -> int:
    """Run one figure through the existing batch controller with --figs fixed."""
    forwarded = list(sys.argv[1:] if argv is None else argv)
    return int(batch_main(["--figs", fig_id, *forwarded]) or 0)


def main_for_subexperiment(fig_id: str, name: str, argv: Sequence[str] | None = None) -> int:
    registry = _registry(fig_id)
    parser = _subexperiment_parser(fig_id, name, registry)
    args, extra_args = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    legacy_args = _build_legacy_args(fig_id=fig_id, name=name, args=args, extra_args=extra_args, registry=registry, parser=parser)
    if args.dry_run:
        print(subprocess.list2cmdline([sys.executable, "-m", str(registry.LEGACY_MODULE), *legacy_args]))
        return 0
    module = importlib.import_module(str(registry.LEGACY_MODULE))
    return int(module.main(legacy_args) or 0)


def main_for_current_subexperiment(module_name: str, argv: Sequence[str] | None = None) -> int:
    if module_name == "__main__":
        main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        module_name = str(getattr(main_spec, "name", module_name))
    parts = module_name.split(".")
    if len(parts) < 2:
        raise SystemExit(f"Cannot infer paper-figure sub-experiment from module name: {module_name}")
    return main_for_subexperiment(parts[-3], parts[-1], argv)


__all__ = ["main_for_current_subexperiment", "main_for_figure", "main_for_subexperiment"]
