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

## Paper Figure Runtime DAG Contract
- Fig.1-Fig.6 are the active paper-figure experiment spine. Treat each figure package under `src/experiments/paper_figures/figX/` as the canonical place for figure-local runtime orchestration.
- The current canonical runtime pattern for paper figures is:
  `python -m src.experiments.paper_figures.figX.run_task --task <task_id> --reuse-artifacts <auto|require|off|force> --output-dir results/<run_root> ...`
- Each active paper figure should keep a figure-local DAG layer:
  - `schemas.py`: task ids, reuse modes, manifest columns, artifact schema constants.
  - `cache_keys.py`: stable cache key, digest, table/file hash, parent specs hash, model fingerprint when applicable.
  - `artifacts.py`: artifact save/load/copy/validate helpers.
  - `run_task.py`: task-level DAG dispatch and CLI.
  - `subexperiments/`: scientific implementation and figure-local task logic.
- Do not bypass the figure-local `run_task.py` layer for new runtime work unless the user explicitly asks for a legacy comparison or a narrow legacy fix.
- Do not treat the old `figX_*_experiment.py` modules as the preferred place for new runtime orchestration. They may remain scientific backends or compatibility baselines, but new DAG behavior belongs in the figure package.
- The runtime structure must preserve this dependency model:
  `source-of-truth specs -> reusable artifact bank -> downstream task -> raw/metrics outputs -> plot-only panel sink`.
- Source-of-truth specs include trial specs, sequence specs, pair specs, mask specs, perturbation specs, cue specs, ping target specs, overlap/region specs, and sweep/job specs. If rerunning such a table could change downstream identity or provenance, persist it as an artifact before downstream use.
- Heavy reusable artifacts include STSP/state boundary banks, sequence banks, feature banks, encoded input banks, support/overlap maps, perturbation baseline banks, rollout banks, trace banks, and reference state banks. Downstream tasks must consume these explicitly instead of implicitly rerunning parent simulation.
- Every reusable artifact should live under `data/intermediates/<task_id>/` and include `cache_key.json` plus a manifest such as `manifest.csv`, `array_manifest.csv`, or `boundary_manifest.csv`.
- Artifact loaders must validate cache key digest, required files, manifest schema, file hashes, row counts or array shapes, condition/delay/task membership, parent specs hash, model fingerprint when model-dependent, and dataset identity when data-dependent.
- `--reuse-artifacts require` is a hard contract: it may only load and validate existing parent artifacts. It must not regenerate specs, masks, sequence banks, state banks, rollouts, traces, or any parent computation. Missing, stale, or corrupt artifacts must fail loudly.
- `--reuse-artifacts auto` may build missing artifacts or reuse matching ones, but it must not hide schema/hash corruption. `force` is only for explicit producer rebuilds. `off` is for legacy/fresh comparison paths.
- A downstream task run in `require` mode must not modify parent artifact directories. If parent artifact hashes change after a downstream require run, the implementation is wrong.
- Plotting specs under `src/plotting/paper_fig/specs/` must declare `producer_task` for every data-backed panel. A panel may list multiple producer tasks, but the dependency must be explicit.
- Plotting code must remain plot-only: it may read `data/raw/`, `data/metrics/`, `summary.json`, resolved specs, and source manifests, but it must not run simulations or create runtime artifacts.
- These DAG rules are mandatory and self-contained. Do not replace them with a new interpretation, a new orchestration style, or a doc-driven redesign unless the user explicitly asks for a new architecture.
- A paper-figure runtime refactor is valid only if it preserves these invariants:
  - parent source specs are generated once, persisted, hashable, and reused by identity;
  - parent artifact banks are generated once, persisted, hashable, and loaded by downstream tasks;
  - every downstream task declares its producer task id, required parent artifacts, output files, and panel consumers;
  - `require` mode loads existing parent artifacts only and never regenerates or mutates them;
  - plot/build code is a leaf consumer and never performs runtime computation;
  - structural refactors must prove output equivalence against the pre-refactor or fresh baseline before being considered complete.
- Do not replace explicit constraints with external prose, memory, or informal prior context. If a rule matters, encode it directly in code, CLI behavior, cache validation, or this file.
- Do not create a new figure runtime pattern for a single figure. Fig.1-Fig.6 must share the same DAG semantics even when their scientific tasks, artifacts, and panels differ.
- Do not collapse multiple DAG nodes into a single all-in-one task for convenience. A task boundary is required whenever its output can be reused, inspected, validated, or rerun independently.
- Do not allow downstream task outputs to depend on freshly resampled trials, masks, pairs, sequences, pings, probes, or perturbations when an upstream spec artifact already exists.
- Do not silently regenerate a missing parent artifact in a downstream-only run. Missing parents in `require` mode are errors, not opportunities to rerun upstream work.
- Do not weaken equality checks to make a refactor pass. If floating-point tolerance is needed, keep row identity, column identity, file presence, shape, and scientific condition labels exact.

## Paper Figure Change Gates
- Before changing a paper-figure runtime path, map the affected DAG nodes: parent specs, reusable artifacts, downstream tasks, output files, and panel consumers.
- Preserve scientific protocol unless the user explicitly approves a protocol change. Do not change delay grids, sequence lengths, sampling rules, mask definitions, target scopes, readout endpoints, restore order, perturbation semantics, STSP mutation/update behavior, or spike encoding as part of a structural refactor.
- Do not extract shared/common code from a single figure just because similar names appear elsewhere. Shared extraction requires stable semantics, at least two real consumers, regression evidence, and human confirmation that the scientific protocol is identical.
- Do not add hidden cross-figure dependencies. If Fig.6 should reuse a Fig.3 artifact, that must become an explicit artifact contract with cache key and manifest validation, not an import from Fig.3 private task code.
- Prefer figure-local helpers until reuse is proven. Move logic into `src/experiments/paper_figures/common/` only when it encodes a stable invariant used by multiple figures.
- Keep legacy/fresh behavior available for regression comparison when refactoring runtime structure. The refactor is not complete until the DAG output matches the legacy/fresh smoke output for key files.
- Smoke/single-seed validation proves structure and equivalence only. Do not describe it as manuscript-final multi-seed evidence.
- If an output equivalence check fails, stop and diagnose. Do not adjust tolerances, drop comparison files, or relabel the change as harmless without evidence.
- If a required source spec or state capture boundary cannot be separated without ambiguity, stop and ask the user before implementing.

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
- After changing a paper-figure DAG runtime path, run the figure-specific gate:
  - compile changed files with `python -m compileall`;
  - run `run_task --task all --reuse-artifacts auto` on a smoke/small validation root;
  - run at least one high-value downstream `run_task --task <task_id> --reuse-artifacts require` from the produced artifact root;
  - compare fresh/legacy vs DAG outputs with `scripts/regression_compare_fig_outputs.py` using an explicit comparison file list and explicit tolerance;
  - compare standalone require vs full DAG outputs for the downstream task;
  - verify parent artifact hashes are unchanged by the require run;
  - run a broken-artifact guard and confirm `require` fails loudly;
  - run `scripts/validate_results_layout.py` on the DAG bundle;
  - run `python -m src.plotting.paper_fig.build --fig <fig_id> --check-only --experiment-root <dag_seed_root>` for affected main/supp figures.
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
