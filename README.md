# Net_torch Experiment Workspace

This repository uses a **split mainline workflow**:

- **Runners** generate normalized experiment result bundles in **computation-only** mode.
- **Plot-only entrypoints** rebuild figures from existing result bundles **without rerunning experiments**.

Mainline workflow:

```text
runner -> results bundle -> plot-only
Quick Start

Run a mainline experiment:

python -m src.experiments.runners.similarity_bias_experiment --output-dir results/similarity_bias_experiment

Run with config:

python -m src.experiments.runners.similarity_bias_experiment \
  --config configs/experiment/similarity_bias_experiment.yaml \
  --output-dir results/similarity_bias_experiment

Rebuild figures from an existing result bundle:

python -m src.plotting.experiments.similarity_bias_experiment_plot \
  --input-dir results/similarity_bias_experiment

Validate the result directory:

python scripts/validate_results_layout.py --input-dir results/similarity_bias_experiment
Repository Layout
configs/                    YAML configs and examples
results/                    Canonical mainline result bundles
scripts/                    Utility scripts, including result validation
src/config/                 Paths, defaults, YAML loader
src/experiments/            Experiment implementations and shared helpers
src/experiments/runners/    Mainline computation-only entrypoints
src/plotting/               Plotting implementations
src/plotting/experiments/   Mainline plot-only entrypoints
tests/                      Validation and plotting tests
archive/                    Historical material outside the mainline workflow
useful_fig_results/         Legacy figure cache, not the canonical results root
Result Layout

Mainline runner experiments write outputs under:

results/<experiment_name>/
├── data/
├── figures/
├── logs/
├── metrics/
├── meta/
├── summary.json
├── run_config.json
└── artifact_manifest.json

Key conventions:

data/: intermediate tables and larger bundle inputs
metrics/: primary summary CSV / JSON artifacts
meta/run_info.json: runtime metadata from the shared runner layer
meta/plot_bundle_manifest.json: plot-only bundle contract where applicable

See results/README.md
 for details.

Mainline Status

Current mainline experiments fall into two categories:

Fully separated

These experiments support:

computation-only runner execution
dedicated plot-only rendering
result-driven figure rebuilding

Examples:

ux_shuffle_memory_collapse
similarity_bias_experiment
dms_overlap_ux_support_mechanism_experiment
overlap_causal_input_perturbation_experiment
chunk_stsp_state_taxonomy
chunk_stsp_multiitem_sequence
chunk_stsp_layer3_anchor_drift_mechanism
Partially separated

These experiments support plot-only rebuilding only for an explicitly retained subset of outputs:

engram_decode — main figure only
l3_accumulator_mechanism_experiment — excludes case-grid
chunk_step2_fused_state_experiment — excludes Panel A

So the current state is:

mainline separation with explicit retained/excluded scope

Configs

Config directories:

configs/experiment/
configs/model/
configs/data/
configs/plotting/

Precedence:

CLI > YAML > code defaults

Shared runner and plotting entrypoints support optional --config.

Validation

Check a result bundle:

python scripts/validate_results_layout.py --input-dir results/<experiment_name>

Strict mode:

python scripts/validate_results_layout.py --input-dir results/<experiment_name> --strict
Compatibility Notes

Some compatibility layers are intentionally kept:

root compatibility files:
summary.json
run_config.json
artifact_manifest.json
legacy experiment code in src/experiments/*.py
normalization of older figure/ / log/ style outputs where needed

These do not change the recommended mainline workflow.

Recommended Rules

For new mainline work:

use runner entrypoints for computation
write normalized outputs to results/<experiment_name>/...
keep plotting result-driven
avoid rerunning experiments inside plot-only entrypoints
make retained vs excluded outputs explicit when coverage is partial
Not Recommended
treating figure/ or log/ as canonical output directories
writing primary metrics only to the result root
making plot-only pipelines rerun experiment computation
keeping critical plotting inputs only in runtime memory
treating archive/ or useful_fig_results/ as the canonical results root
presenting partially migrated experiments as if all outputs were fully covered