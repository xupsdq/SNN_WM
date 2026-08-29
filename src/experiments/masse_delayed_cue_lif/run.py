"""DAG CLI for the Masse delayed-cue LIF experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

from .artifacts import (
    REQUIRED_PLOT_INPUTS,
    layout_for,
    load_run_config,
    record_run_info,
    require_files,
    required_plot_inputs,
    save_run_config,
    write_manifest,
    write_summary,
)
from .config import MasseDelayedCueConfig, PROFILE_CHOICES, profile_config
from .decode import decode_run
from .evaluate import evaluate_run
from .plot import plot_run
from .task import generate_trial_table, save_trial_table, validate_trial_table
from .train import train_run


def build_trials(
    run_directory: Path,
    config: MasseDelayedCueConfig,
    *,
    reuse_trials: Path | None = None,
) -> dict:
    layout = layout_for(run_directory)
    destination = layout.data_dir / "trials.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if reuse_trials is not None:
        source = Path(reuse_trials)
        if not source.is_file():
            raise FileNotFoundError(f"reuse-trials file does not exist: {source}")
        table_info = validate_trial_table(source, config)
        shutil.copyfile(source, destination)
    else:
        rows = generate_trial_table(config)
        save_trial_table(destination, rows)
        table_info = validate_trial_table(destination, config)
    save_run_config(run_directory, config)
    write_manifest(run_directory)
    record_run_info(run_directory, command="build-trials", config=config, status="complete")
    summary = {
        "status": "trials_built",
        "profile": config.profile,
        "n_rows": table_info["n_rows"],
        "n_train": table_info["n_train"],
        "n_val": table_info["n_val"],
        "n_test": table_info["n_test"],
        "trials_sha256": table_info["trials_sha256"],
    }
    write_summary(run_directory, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Masse delayed-cue DMS+DMRS recurrent LIF DAG."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--output-directory", type=Path, required=True)
    shared.add_argument("--profile", choices=PROFILE_CHOICES, default="smoke")
    shared.add_argument("--device", default="cuda")
    shared.add_argument("--max-epochs", type=int)
    shared.add_argument("--n-hidden", type=int)
    shared.add_argument("--n-train", type=int)
    shared.add_argument("--n-val", type=int)
    shared.add_argument("--n-test", type=int)
    shared.add_argument("--batch-size", type=int)
    shared.add_argument("--learning-rate", type=float)

    shared.add_argument("--reuse-trials", type=Path)
    subparsers.add_parser("build-trials", parents=[shared])
    subparsers.add_parser("train", parents=[shared])
    subparsers.add_parser("evaluate", parents=[shared])
    subparsers.add_parser("decode", parents=[shared])
    plot = subparsers.add_parser("plot", parents=[shared])
    plot.add_argument("--check-only", action="store_true")
    return parser


def config_from_args(arguments) -> MasseDelayedCueConfig:
    overrides = {"device": arguments.device}
    for field, attr in (
        ("max_epochs", "max_epochs"),
        ("n_hidden", "n_hidden"),
        ("n_train", "n_train"),
        ("n_val", "n_val"),
        ("n_test", "n_test"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
    ):
        value = getattr(arguments, attr, None)
        if value is not None:
            overrides[field] = value
    return profile_config(arguments.profile, **overrides)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_directory = arguments.output_directory
    config = config_from_args(arguments)
    if arguments.command == "build-trials":
        result = build_trials(run_directory, config, reuse_trials=arguments.reuse_trials)
    elif arguments.command == "train":
        result = train_run(run_directory, config)
    elif arguments.command == "evaluate":
        result = evaluate_run(run_directory, config)
    elif arguments.command == "decode":
        result = decode_run(run_directory, config)
    else:
        if arguments.check_only:
            require_files(run_directory, REQUIRED_PLOT_INPUTS)
            stored = load_run_config(run_directory)
            require_files(run_directory, required_plot_inputs(stored))
            result = {"plot_only": True, "check_only": True}
        else:
            result = plot_run(run_directory)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
