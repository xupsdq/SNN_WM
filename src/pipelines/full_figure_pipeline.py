from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.config.defaults import DEFAULT_PROJECT_DEFAULTS
from src.experiments.paper_figures.run_paper_figures import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH_GLOB,
    DEFAULT_OUTPUT_ROOT,
    SCOPES,
    main as run_paper_figures_main,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated compatibility wrapper. Delegates to "
            "python -m src.experiments.paper_figures.run_paper_figures."
        )
    )
    parser.add_argument("--figs", type=str, default="all", help="'all' or comma-separated figure ids like fig1,fig3.")
    parser.add_argument("--scope", choices=sorted(SCOPES), default="both")
    parser.add_argument("--seeds", type=str, default="1000")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--model-path-glob", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None, help="Deprecated alias for --model-path-glob.")
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--results-root", type=str, default=None, help="Deprecated; maps to <results-root>/paper_experiments when --output-root is omitted.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_PROJECT_DEFAULTS.runtime.device)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed-jobs", type=int, default=None)
    parser.add_argument("--experiment-batch-size", type=int, default=None)
    parser.add_argument("--fig1-dms-batch-size", type=int, default=None)
    parser.add_argument("--fig4-l3-region-batch-size", type=int, default=None)
    parser.add_argument("--enable-gpu-batching", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--force", dest="resume", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-build-paper-figures", action="store_true")
    parser.add_argument("--check-only-build", action="store_true")
    parser.add_argument("--save-debug-figures", action="store_true")
    parser.add_argument("--progress-mode", choices=["auto", "compact", "detailed", "off"], default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-path", type=str, default=None, help="Deprecated and ignored; batch logs are written under output-root/_batch_runs.")
    return parser


def _output_root(args: argparse.Namespace) -> str:
    if args.output_root:
        return str(args.output_root)
    if args.results_root:
        return str(Path(args.results_root) / "paper_experiments")
    return DEFAULT_OUTPUT_ROOT


def _model_path_glob(args: argparse.Namespace) -> str:
    return str(args.model_path_glob or args.model_path or DEFAULT_MODEL_PATH_GLOB)


def _forwarded_args(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--figs",
        str(args.figs),
        "--scope",
        str(args.scope),
        "--seeds",
        str(args.seeds),
        "--model-path-glob",
        _model_path_glob(args),
        "--dataset-root",
        str(args.dataset_root),
        "--output-root",
        _output_root(args),
        "--device",
        str(args.device),
        "--split",
        str(args.split),
        "--jobs",
        str(int(args.jobs)),
        "--progress-mode",
        str(args.progress_mode),
    ]
    if args.seed_jobs is not None:
        forwarded.extend(["--seed-jobs", str(int(args.seed_jobs))])
    if args.experiment_batch_size is not None:
        forwarded.extend(["--experiment-batch-size", str(int(args.experiment_batch_size))])
    if args.fig1_dms_batch_size is not None:
        forwarded.extend(["--fig1-dms-batch-size", str(int(args.fig1_dms_batch_size))])
    if args.fig4_l3_region_batch_size is not None:
        forwarded.extend(["--fig4-l3-region-batch-size", str(int(args.fig4_l3_region_batch_size))])
    if args.enable_gpu_batching:
        forwarded.append("--enable-gpu-batching")
    if args.all_seeds:
        forwarded.append("--all-seeds")
    if args.smoke:
        forwarded.append("--smoke")
    if not args.resume:
        forwarded.append("--force")
    if args.continue_on_error:
        forwarded.append("--continue-on-error")
    if args.no_build_paper_figures:
        forwarded.append("--no-build-paper-figures")
    if args.check_only_build:
        forwarded.append("--check-only-build")
    if args.save_debug_figures:
        forwarded.append("--save-debug-figures")
    if args.no_progress:
        forwarded.append("--no-progress")
    if args.dry_run:
        forwarded.append("--dry-run")
    return forwarded


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.log_path:
        print("--log-path is deprecated for src.pipelines.full_figure_pipeline and is ignored.")
    print("src.pipelines.full_figure_pipeline is deprecated; delegating to src.experiments.paper_figures.run_paper_figures.")
    return int(run_paper_figures_main(_forwarded_args(args)) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
