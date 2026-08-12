# AGENTS.md

## Purpose
- This is a PyTorch spiking-neural-network workspace for simulations, experiments, analysis, plotting, and manuscript-related work.
- Work from the repository root. Prefer `S:\pycharm\Anaconda\envs\torch_env\python.exe` when the default Python lacks project dependencies.

## Project Structure
- Core code: `src/core/`, `src/data/`, and `src/config/`.
- Experiments: `src/experiments/`; shared experiment utilities: `src/experiments/common/`.
- Plotting: `src/plotting/`; shared plotting utilities: `src/plotting/common/`.
- Configurations: `configs/`. Historical code: `archive/`; do not use it as the source of new mainline patterns.

## Required Workflow
- Mainline computation uses `python -m src.experiments.runners.<experiment_id> --output-dir results/<experiment_id>`.
- Plot-only replay uses `python -m src.plotting.experiments.<experiment_id>_plot --input-dir results/<experiment_id>` and must never rerun simulations.
- New experiments that produce figures must provide both a runner and a plot-only entrypoint. Do not import one experiment script from another; move genuinely shared logic into a shared module.

## Paper Figures
- Fig.1-Fig.6 use `src/experiments/paper_figures/figX/run_task.py` as the canonical runtime entrypoint.
- Preserve the DAG: persisted source specs -> validated reusable artifacts -> downstream task outputs -> plot-only panels.
- `--reuse-artifacts require` is load-only: missing, stale, or corrupt parents must fail, and downstream runs must not regenerate or modify parent artifacts.
- Plotting must declare its producer tasks and remain a leaf consumer.
- Preserve scientific protocols unless the user explicitly approves a change.
- Reusable rollout artifacts belong under `results/multi_seed_rollout/`; manuscript-facing multi-seed bundles belong under `results/paper_figure_multi_seed/`.

## Confirmed Introduction Problem
- The unresolved problem is how an inherited, distributed STSP state shapes the processing of a later input and is then rewritten by that processing, enabling continual, history-dependent evolution of working-memory representations without sustained firing.

## Outputs and Validation
- Normalized result bundles use `data/`, `figures/`, `logs/`, `metrics/`, `meta/`, `summary.json`, `run_config.json`, and `artifact_manifest.json`.
- After runtime changes: compile changed files, run a smoke DAG, run a downstream task in `require` mode, confirm parent hashes are unchanged, confirm a broken parent fails loudly, and rerun plotting/check-only validation.
- Smoke or single-seed runs validate structure only; do not present them as manuscript-final evidence.

## Safety
- Preserve unrelated worktree changes and avoid unrelated refactors.
- Do not delete generated results or caches unless explicitly asked.
- Do not treat `results/`, datasets, caches, checkpoints, or generated figures/tables as source code or commit them by default.
