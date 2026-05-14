# Paper Figure Generation

`src/plotting/paper_fig/` is a paper-specific figure generation layer. It starts from manuscript figure and panel specifications, then resolves existing experiment outputs into canonical panel data before rendering manuscript figures.

This is deliberately separate from `src/plotting/experiments/`. Experiment plotting scripts show experiment result bundles. `paper_fig` maps those results into final manuscript arguments, panel identities, source manifests, and QC reports.

## Current Status

Fig.1 is implemented as the first skeleton:

- Fig.1A is a manual schematic slot expecting `manual_assets/fig1a_architecture.svg`.
- Fig.1B reads overall recall from the 20-network ensemble summary when available.
- Fig.1C reads delay-decoding metrics from multi-network `engram_decode` output, with single-run fallback.
- Fig.1D/E read `ux_shuffle_memory_collapse` summaries, with single-run fallbacks.
- Missing inputs create placeholder data and QC warnings instead of unhandled exceptions.

Fig.2 is implemented as the first state-level result skeleton. It follows a three-band argument layout:

- Row 1: protocol/reference logic, rendered as a horizontal two-item episode schematic.
- Row 2: structural evidence from network-level canonical data: fusion dual score, true-vs-shuffled pair specificity, and WPRI.
- Row 3: functional evidence from existing assay tables: neutral-ping access and partial-cue completion.

Fig.2 adapters consume existing `chunk_step2_fused_state_experiment` and `fig4_chunk_interaction_assay` outputs. They map internal state labels such as `baseline`, `S_B`, and `S_AB` into manuscript labels: `No memory`, `Item 2 only`, and `Item 1->Item 2`.

Fig.3 is implemented as the state-level outcome skeleton. It follows a three-row argument layout:

- Row 1: pair-level directionality, with fusion imbalance and latent-bias/readout-preference relationship.
- Row 2: multi-item retention and accessibility, with latest-vs-earlier similarity mass and neutral-ping seen-item access.
- Row 3: shifting anchor expression, with state center-of-mass and ping center-of-mass trajectories across sequence stages.

Fig.3 intentionally does not map the old `stepwise_update_ratio`/SUR output into the main manuscript Fig.3F. That legacy stem is recorded as an explicitly unused source candidate for later supplement/mechanism use.

Fig.4 is implemented as the mechanism-entry skeleton for overlap-gated synaptic re-entry. It follows a three-band layout:

- Row 1: the sample-probe assay and overlap definition.
- Row 2: effect localization from similarity-bin accuracy drop to high-vs-low overlap comparison.
- Row 3: functional restoration via dynamic-probe index timecourse and final readout recovery.

Fig.4 adapters consume existing `similarity_bias_experiment` and `overlap_causal_input_perturbation_experiment` outputs. They map internal perturbation condition names to manuscript labels: `Overlap-preserving` and `Non-overlap control`. Fig.4 intentionally does not include Fig.5 spike-recruitment or local-competition metrics.

Fig.5 is implemented as the support-to-competition mechanism skeleton. It follows a two-band layout:

- Row 1: overlap-aligned pre-probe support and its conversion into early recruitment.
- Row 2: local winner-loser competition, with event-aligned voltage/inhibition traces and event-pattern fractions.

Fig.5 adapters consume existing `dms_overlap_ux_support_mechanism_experiment` outputs. Some source figures and historical stems are named `fig4_panel_*`; those names are recorded only in source mappings/manifests and are not used as manuscript panel labels. Fig.5 intentionally does not repeat Fig.4 similarity/DPI/readout-recovery panels or pre-stage Fig.6 peak/recency/anchor metrics.

Fig.6 is implemented as the causal-closure skeleton for recency-weighted STSP peaks. It follows a three-band layout:

- Row 1: anchor-peak linkage, with Fig.6A kept as an explicit data-gap-aware placeholder if no direct support-loss/anchor-retreat source is found.
- Row 2: update-history origin, with peak membership, repetition x recency gain, and update+recency model comparison.
- Row 3: functional causal closure, with peak flattening/intact/boosting and probe-peak overlap dependency.

Fig.6 adapters consume existing `chunk_stsp_layer1_overlap_peak_formation` outputs. Historical stems such as `fig6B_update_recency_final_g`, `fig6C_anchor_prediction_model_comparison`, `fig6D_peak_function_spiking`, and `fig6E_overlap_conditioned_spike_effect` are recorded only in source mappings/manifests because the manuscript panels are Fig.6A-F.

## Running

From the repository root:

```bash
python -m src.plotting.paper_fig.build --fig fig1 --check-only
python -m src.plotting.paper_fig.build --fig fig1
python -m src.plotting.paper_fig.build --fig fig1 --panel C
python -m src.plotting.paper_fig.build --all
```

Default output location:

```text
src/plotting/paper_fig/outputs/<figure_id>/
```

For Fig.1, expected outputs include:

- `fig1.pdf`, `fig1.svg`, `fig1.png`
- `fig1_resolved_spec.yaml`
- `fig1_source_manifest.json`
- `fig1_qc_report.md`
- `panel_data/`, `stats/`, and `source_manifests/`

## Adding A Figure

1. Add a figure entry to `specs/paper_figures.yaml`.
2. Add `specs/figX.yaml` with `figure_id`, `figure_title_draft`, `canvas_mm`, `reading_order`, `argument_bands`, `panels`, `expected_outputs`, and `qc_requirements`.
3. Add or update `layouts/figX_layout.py` if the figure needs a manuscript-specific mm layout.
4. Add adapters in `adapters/figX_adapters.py` and renderers in `panels/figX_panels.py`.

The spec is the source of truth. Do not hard-code manuscript panel decisions inside renderers.

## Adding A Panel

Each panel spec should define:

- `panel_id`
- `claim`
- `panel_type`
- `data_adapter`
- `renderer`
- `size_mm`
- `position_mm` or a layout slot
- `source_mapping`
- axis labels, reference lines, conditions, legend semantics, and stats expectations

Use explicit source mappings when old experiment stems do not match manuscript panel labels.

## Writing An Adapter

Adapters convert existing experiment outputs into canonical panel data. They do not draw figures.

Interface:

```python
def build_<adapter_name>(spec, repo_root, output_dir) -> AdapterResult:
    ...
```

Each adapter writes:

- `panel_data/figXa_panel_data.csv`
- `stats/figXa_stats.json`
- `source_manifests/figXa_sources.json`

Canonical panel data should include `figure_id`, `panel_id`, `metric`, `condition`, `layer`, `network_id` or `seed_id`, `value`, `unit`, and `source_file` where applicable.

## Writing A Renderer

Renderers consume canonical panel data and stats. They do not search experiment folders.

Interface:

```python
def render_<panel_name>(ax, panel_data, stats, spec, style=None):
    ...
```

Current supported skeleton renderer families include manual schematic slots, dot summaries, layerwise dot summaries, outcome profiles, paired attribution shifts, and generic placeholders.

## Style Policy

Do not hard-code final colors, fonts, marker sizes, or line widths in this stage. The renderers accept a `style` parameter so a unified paper style system can be added later.

All manuscript layout dimensions should be expressed in millimeters. Full figure export avoids tight bounding boxes so the requested canvas size is preserved.
