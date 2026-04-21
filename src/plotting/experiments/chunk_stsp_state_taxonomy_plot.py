from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from src.config.yaml_loader import load_yaml_file, nested_get
from src.plotting.experiments.chunk_stsp_state_taxonomy_plot_lib import load_plot_bundle, render_figure_groups


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot-only entrypoint for chunk_stsp_state_taxonomy.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def _resolve_from_config(config: dict[str, Any] | None, *path: str, default: Any = None) -> Any:
    if not config:
        return default
    value = nested_get(config, *path, default=None)
    if value is not None:
        return value
    if len(path) == 1:
        return config.get(path[0], default)
    return default


def _apply_plot_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config_payload = load_yaml_file(args.config) if args.config else {}
    args.input_dir = args.input_dir or _resolve_from_config(config_payload, "input_dir", default=_resolve_from_config(config_payload, "experiment", "output_dir"))
    args.output_dir = args.output_dir or _resolve_from_config(config_payload, "plotting", "output_dir", default=_resolve_from_config(config_payload, "output_dir"))
    if not args.input_dir:
        raise SystemExit("--input-dir is required (or provide it via --config).")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    args = _apply_plot_config_defaults(args)
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "figures"
    bundle = load_plot_bundle(input_dir)
    outputs = render_figure_groups(bundle, figures_dir=output_dir)
    print(
        f"[Output] Generated {len(outputs)} figure groups for chunk_stsp_state_taxonomy -> {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
