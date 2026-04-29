# AGENTS.md

## Project Overview
- This repository is a PyTorch-based SDNN experiment workspace.
- The project is in an experiment-engineering normalization phase: mainline experiments use unified computation runners and plot-only entrypoints, while legacy experiments and historical outputs remain for compatibility.
- Prefer repo-local shared utilities and normalized workflows over copying patterns from archived scripts.

## Repository Map
- `src/core/`: SDNN network and monitoring primitives.
- `src/data/`: data encoding and dataset loader helpers.
- `src/config/`: default paths, runtime defaults, YAML loading, and units.
- `src/experiments/`: experiment implementations and shared experiment utilities.
- `src/experiments/common/`: shared APIs for datasets, model loading, monitored DMS, result layout, run info, statistics, seeds, and related utilities.
- `src/experiments/runners/`: mainline computation-only entrypoints.
- `src/plotting/experiments/`: mainline plot-only entrypoints.
- `src/plotting/common/`: plotting style, publication helpers, figure export, and tidy CSV helpers.
- `scripts/validate_results_layout.py`: validates normalized result bundles.
- `configs/`: minimal YAML configs for mainline runner and plotting entrypoints.
- `archive/`: historical scripts and outputs; do not treat this as the source of new mainline patterns.
- `results/`, `MNIST/`, caches, checkpoints, figures, and generated CSV/JSON artifacts are local/generated data unless explicitly tracked.

## Environment
- Work from the repository root.
- Preferred runtime is the Conda `torch_env` Python when available, especially `S:\pycharm\Anaconda\envs\torch_env\python.exe`.
- Main runtime dependencies include `torch`, `numpy`, `pandas`, `matplotlib`, `scipy`, `sklearn`, `tqdm`, and optional `PyYAML` for `--config`.
- If default `python` lacks PyTorch, rerun with the explicit `torch_env` interpreter instead of changing project code.
- Use `--device auto` or CPU smoke runs when CUDA availability is uncertain.
- `KMP_DUPLICATE_LIB_OK=TRUE` is set by runner/pipeline code where needed; preserve that behavior.

## Mainline Workflow
- Computation entrypoint pattern:
  `python -m src.experiments.runners.<experiment_id> --output-dir results/<experiment_id>`
- Plot-only entrypoint pattern:
  `python -m src.plotting.experiments.<experiment_id>_plot --input-dir results/<experiment_id>`
- Plot-only code must read an existing result bundle and must not rerun the experiment.
- The shared runner layer supports `--config`, `--output-dir`, `--model-path`, `--dataset-root`, `--device`, `--seed`, and `--smoke`.
- CLI arguments override YAML config values; YAML overrides code defaults.
- Experiment IDs and mainline metadata belong in `src/experiments/catalog.py`.

## Experiment Development Rules
- For new mainline experiments, add or update a computation entrypoint under `src/experiments/runners/` and a plot-only entrypoint under `src/plotting/experiments/` when figures are expected.
- Every new experiment script must be paired with a runner entrypoint and plot-only code: add or update `src/experiments/runners/<experiment_id>.py` and `src/plotting/experiments/<experiment_id>_plot.py` together with the experiment implementation.
- Experiment scripts must not directly call or import code from other experiment scripts. They may only depend on shared layers such as `src/experiments/common/`, domain-specific `shared/` modules, `src/config/`, `src/core/`, `src/data/`, and `src/plotting/common/`.
- If logic is needed by multiple experiments, move it into a shared module first; do not reach across from one experiment file into another experiment file.
- Keep new experiment scripts directly runnable when they are intended to support direct execution; add a repo-root `sys.path` bootstrap if direct `python path/to/script.py` execution would otherwise fail.
- Preserve `--smoke` and `--skip-figures` support for expensive experiments where practical.
- Prefer post-hoc reanalysis from existing trial-level outputs when the needed metrics already exist; do not rerun expensive network conditions unless the task explicitly requires it.
- Archived scripts may be used for historical context, but new code should follow current `src/experiments/common/`, runner, plotting, and result-bundle conventions.

## Result Bundle Contract
- New normalized experiment outputs should use:
  - `data/`
  - `figures/`
  - `logs/`
  - `metrics/`
  - `meta/`
  - root-level `summary.json`
  - root-level `run_config.json`
  - root-level `artifact_manifest.json`
- Use `prepare_result_layout()` from `src.experiments.common.results` to create bundle directories.
- Use shared helpers such as `save_run_config()`, `save_summary_json()`, `save_log_lines()`, and `save_tidy_csv()` instead of ad hoc file writing.
- `meta/run_info.json` should record experiment name, git commit, start/end time, status, output directory, entry script, command, dataset, seed, and model path when applicable.
- Preserve compatibility with existing root-level `summary.json`, `run_config.json`, and `artifact_manifest.json`.

## Plotting Rules
- Save publication figures in `png`, `pdf`, and `svg` using `save_figure_all_formats()` unless a task asks otherwise.
- Use `apply_publication_style()` and `src/plotting/common/theme_tokens.py` / `src/plotting/common/io.py` for shared figure styling.
- Plot scripts should validate required input files and columns rather than silently guessing missing data.
- Plot-only replay should write to the existing bundle `figures/` directory by default.

## Validation
- After changing experiment or runner behavior, run a smoke command for the affected experiment.
- After producing a normalized result bundle, validate it:
  `python scripts/validate_results_layout.py --input-dir results/<experiment_id>`
- Use strict validation when checking mainline-ready bundles:
  `python scripts/validate_results_layout.py --input-dir results/<experiment_id> --strict`
- If no automated tests exist for the touched area, state that explicitly and provide the smoke/layout command that was run.
- Do not consider a change complete based only on syntax checks when a smoke path is available.

## Coding Conventions
- Prefer small, explicit helpers in `src/experiments/common/` only when they are reused or encode a stable invariant.
- Keep experiment-specific analysis local to the experiment unless it is clearly shared across multiple experiments.
- Use structured JSON/CSV outputs for metrics; do not leave key results only in logs or printed text.
- Avoid unrelated refactors, path churn, and broad archive cleanup while implementing a focused experiment or plotting change.
- Preserve existing public CLI flags and output filenames unless the user explicitly requests a breaking change.

## Safety And Generated Files
- Do not commit or treat as source of truth: `results/`, `MNIST/`, `.pytest_cache/`, `.pytest_tmp/`, `cache/`, `cache_data/`, `__pycache__/`, checkpoints, generated figures, or generated CSV/NPY/NPZ artifacts.
- `.pytest_tmp/` may contain inaccessible temporary directories; ignore those during scans.
- Do not use `archive/` or historical result folders as the destination for new mainline outputs.
- Do not delete generated results or caches unless the user explicitly asks.
