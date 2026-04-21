# Net_torch Experiment Workspace

This repository is organized around experiment runners and plot-only entrypoints.
All 10 mainline runner experiments now write more native normalized outputs under
`results/<experiment_name>/...` while keeping compatibility files at the result root.

## Directory layout

```text
configs/                    Minimal YAML configs and examples
results/                    Mainline experiment outputs
scripts/                    Utility scripts, including result validation
src/config/                 Paths, defaults, runtime config, YAML loader
src/experiments/            Experiment implementations and shared helpers
src/experiments/runners/    Mainline computation entrypoints
src/plotting/               Plotting implementations
src/plotting/experiments/   Mainline plot-only entrypoints
tests/                      Normalization and validator tests
archive/                    Historical material outside the mainline workflow
useful_fig_results/         Legacy figure cache, not the canonical results root
```

## Mainline experiment workflow

- Compute entrypoint: `python -m src.experiments.runners.<experiment_id>`
- Plot entrypoint: `python -m src.plotting.experiments.<experiment_id>_plot`
- Plotting reads an existing result bundle and does not rerun the experiment.
- All 10 runner experiments now follow the same normalized result layout.

## Example commands

Run directly:

```bash
python -m src.experiments.runners.similarity_bias_experiment --output-dir results/similarity_bias_experiment
```

Run with config:

```bash
python -m src.experiments.runners.similarity_bias_experiment --config configs/experiment/similarity_bias_experiment.yaml --output-dir results/similarity_bias_experiment
```

Replot from an existing bundle:

```bash
python -m src.plotting.experiments.similarity_bias_experiment_plot --input-dir results/similarity_bias_experiment
```

## Result layout

```text
results/<experiment_name>/
├── data/
├── figures/
├── logs/
├── metrics/
├── meta/
├── summary.json
├── run_config.json
└── artifact_manifest.json
```

- `meta/run_info.json` is created by the shared runner layer.
- `data/` stores intermediate tables and large arrays.
- `metrics/` stores primary summary CSV and JSON artifacts.
- `meta/` stores configuration snapshots and runtime metadata.
- See [results/README.md](results/README.md) for details.

## Configs and precedence

- `configs/experiment/`: experiment-level examples
- `configs/model/`: model checkpoint defaults
- `configs/data/`: dataset defaults
- `configs/plotting/`: plotting defaults

Precedence:

```text
CLI > YAML > code defaults
```

Current example experiment configs:

- `configs/experiment/similarity_bias_experiment.yaml`
- `configs/experiment/engram_decode.yaml`

All shared runner and plotting entrypoints support optional `--config`.

## Validate a result directory

```bash
python scripts/validate_results_layout.py --input-dir results/similarity_bias_experiment
```

Strict mode:

```bash
python scripts/validate_results_layout.py --input-dir results/similarity_bias_experiment --strict
```

## Compatibility layers kept on purpose

- Root compatibility files remain: `summary.json`, `run_config.json`, `artifact_manifest.json`
- Legacy experiment implementations remain in `src/experiments/*.py`
- The shared runner still normalizes old `figure/` and `log/` outputs when needed
- Plotting still resolves files from root, `data/`, `metrics/`, and `meta/`

## Old patterns that are no longer recommended

- Writing primary metrics only to the result root or only to `logs/`
- Adding new experiments that still treat `figure/` or `log/` as canonical directories
- Treating `archive/` or `useful_fig_results/` as the canonical results root
