# Plot-Only Experiment Template

This document records the recommended pattern for experiments that should support
"compute once, replot many times" without rerunning the experiment.

## 1. Applicability

This migration pattern is a good fit when:

- the experiment already writes the data needed to rebuild figures
- figures can be reconstructed from CSV, JSON, or NPZ files without touching the model
- the plotting code is experiment-specific and a generic fallback plot is not enough
- replotting is expected to happen repeatedly during paper iteration

This pattern is usually a poor fit when:

- the figure depends on transient runtime objects that are not saved
- rebuilding the figure would require replaying the model or dataset sampling logic
- the experiment only has a trivial single summary plot and `_common.py` is already sufficient
- the missing plot bundle would be unreasonably large

## 2. Recommended Result Layout

All migrated experiments should continue using the normalized result layout:

```text
results/<experiment_name>/
|- data/
|- figures/
|- logs/
|- metrics/
`- meta/
```

Recommended placement:

- `data/`: raw trial-level tables, pair-level tables, large NPZ payloads, plot helper tables
- `figures/`: rendered figures only
- `logs/`: runtime logs
- `metrics/`: summary CSV/JSON used for interpretation and reporting
- `meta/`: run metadata, config snapshots, plot bundle manifest

## 3. Plot Bundle Design

### 3.1 Manifest-only bundle

Use a manifest-only bundle when all figure inputs are already saved as normal result files.

Recommended manifest path:

```text
meta/plot_bundle_manifest.json
```

Recommended structure:

```json
{
  "version": 1,
  "experiment_name": "example_experiment",
  "inputs": {
    "summary_metrics": {
      "path": "metrics/summary_metrics.csv",
      "required_columns": ["x", "y"],
      "purpose": "Primary summary curve"
    }
  },
  "outputs": [
    "figures/figure_main.png",
    "figures/figure_main.pdf",
    "figures/figure_main.svg"
  ],
  "notes": [
    "Plot-only reads the bundle and does not rerun the experiment."
  ]
}
```

### 3.2 When to add extra bundle files

Add an explicit `data/plot_bundle.npz`, `data/plot_inputs.csv`, or similar only when:

- figure inputs are not already saved elsewhere
- the missing inputs are small enough to save cheaply
- the saved payload is the minimal sufficient information for figure replay

Do not save large duplicated arrays just to avoid a small amount of plotting refactor.

## 4. Code Responsibilities

### `src/experiments/<experiment>.py`

Should:

- run the experiment
- save all figure-relevant data into `data/` and `metrics/`
- write `meta/plot_bundle_manifest.json`
- optionally call the plotting library when figures are explicitly enabled

Should not:

- require plotting for a successful compute run
- hide figure inputs in unsaved in-memory variables

### `src/plotting/experiments/<experiment>_plot.py`

Should:

- provide a dedicated CLI
- accept `--input-dir`, `--output-dir`, and optional `--config`
- load only result bundle files
- validate required inputs
- call the experiment-specific plotting library

Should not:

- call the experiment main function
- re-run the model
- delegate the core figure logic back to `_common.py`

### Optional plotting library

Use `src/plotting/experiments/<experiment>_plot_lib.py` when the figure logic is non-trivial.

Typical responsibilities:

- load plot bundle inputs
- validate required columns
- build plot-ready tables
- render and save figures

### `_common.py`

Keep `_common.py` for:

- generic fallback plotting
- generic CLI defaults
- small experiments that truly only need `summary.json` plus one CSV

Do not keep experiment-specific multi-panel logic in `_common.py`.

## 5. Migration Checklist

- [ ] Read the original in-experiment figure code.
- [ ] List every figure and its exact data dependencies.
- [ ] Separate inputs into:
  - already saved
  - missing and must be added to the bundle
- [ ] Save any missing minimal inputs to `data/` or `metrics/`.
- [ ] Add `meta/plot_bundle_manifest.json`.
- [ ] Extract figure rendering into a plotting library.
- [ ] Replace the thin `_common.main_for(...)` entry with a dedicated plot-only entry.
- [ ] Keep runner behavior computation-only by default.
- [ ] Smoke test compute-only output.
- [ ] Smoke test plot-only replay.
- [ ] Compare file set and visual intent against original in-run figure generation.

## 6. Validation Rules

Recommended validation flow:

1. Run the experiment via the runner in computation-only mode.
2. Check that `meta/plot_bundle_manifest.json` exists.
3. Check that all manifest inputs exist in `data/` or `metrics/`.
4. Run the dedicated plot-only entry.
5. Check that expected figure files appear in `figures/` or the chosen output dir.
6. If in-run plotting still exists, compare:
   - figure stems
   - number of generated figure groups
   - major content intent

"Sufficiently consistent" usually means:

- same figure count
- same file naming scheme
- same plotted quantities
- same overall style and layout intent

Byte-identical outputs are not required.

## 7. When Not To Force Migration

Do not force full migration yet when:

- figure replay would require recomputing hidden state traces
- the experiment only writes final scalar summaries but not figure inputs
- the missing payload would be too large to save reasonably
- the experiment is legacy or archive-only and not part of the mainline workflow

In those cases, prefer:

- partial migration
- TODO notes
- explicit manifest preparation without claiming full plot-only replay support

## 8. Existing Examples

Current examples in this repository:

- `chunk_stsp_multiitem_sequence`
- `ux_shuffle_memory_collapse`
- `similarity_bias_experiment`
- `engram_decode`
