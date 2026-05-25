from __future__ import annotations

import argparse
from typing import Sequence

from src.plotting.paper_fig.build import main as build_paper_fig_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated compatibility wrapper. Delegates to "
            "python -m src.plotting.paper_fig.build."
        )
    )
    parser.add_argument("--fig", type=str, default=None, help="Single paper figure id, e.g. fig6.")
    parser.add_argument("--figs", type=str, default=None, help="'all' or comma-separated figure ids like fig1,fig6.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--panel", type=str, default=None)
    parser.add_argument("--experiment-root", type=str, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--results-root", type=str, default=None, help="Deprecated and ignored.")
    parser.add_argument("--output-dir", type=str, default=None, help="Deprecated; paper_fig output roots are spec-driven.")
    parser.add_argument("--model-path", type=str, default=None, help="Deprecated and ignored.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Deprecated and ignored.")
    return parser


def _requested_figures(args: argparse.Namespace) -> list[str] | None:
    if args.all or (args.figs and str(args.figs).strip().lower() == "all"):
        return None
    raw = args.figs or args.fig
    if not raw:
        return ["fig1"]
    figures: list[str] = []
    seen: set[str] = set()
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        fig_id = token if token.lower().startswith("fig") else f"fig{token}"
        fig_id = fig_id.lower()
        if fig_id in seen:
            continue
        figures.append(fig_id)
        seen.add(fig_id)
    return figures or ["fig1"]


def _build_args(args: argparse.Namespace, fig_id: str | None = None) -> list[str]:
    forwarded: list[str] = []
    if fig_id is None:
        forwarded.append("--all")
    else:
        forwarded.extend(["--fig", fig_id])
    if args.panel:
        forwarded.extend(["--panel", str(args.panel)])
    if args.experiment_root:
        forwarded.extend(["--experiment-root", str(args.experiment_root)])
    if args.check_only:
        forwarded.append("--check-only")
    return forwarded


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.results_root or args.output_dir or args.model_path or args.dataset_root:
        print("Legacy rebuild_figures path/model/output flags are ignored; paper_fig uses specs and --experiment-root.")
    print("src.pipelines.rebuild_figures is deprecated; delegating to src.plotting.paper_fig.build.")

    figures = _requested_figures(args)
    if figures is None:
        return int(build_paper_fig_main(_build_args(args, None)) or 0)

    exit_code = 0
    for fig_id in figures:
        exit_code = max(exit_code, int(build_paper_fig_main(_build_args(args, fig_id)) or 0))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
