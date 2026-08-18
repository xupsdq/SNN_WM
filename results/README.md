# Results Layout

Updated: 2026-08-14

`results/` stores normalized experiment artifacts and manuscript-facing bundles. Paper authority is defined by `docs/paper/PAPER_AUTHORITY.json`; modification time alone never determines whether a result is active or archivable.

## Pre-submission preservation policy

The current policy is `full_dag` until a later explicit cold-archive decision. Keep these roots in place:

```text
results/
├── multi_seed_rollout/                  # reusable parent artifacts and historical DAG lineage
├── multi_snn/                           # independently trained network checkpoints
├── paper_figure_multi_seed/             # manuscript-facing and parent result bundles
├── paper_figures/                       # rendered artwork, manual assets, promotion records and archive
├── causal_closure_multi_seed_20260803/  # C5 20-network closure
└── archive/                             # results already classified as historical
```

Do not move a root merely because its files are older than one month. Active source manifests still point into old parent directories, and `require` mode depends on stable paths, schemas and hashes.

## Current manuscript-facing bundles

### Main manuscript Fig.1–Fig.7

- Formal artwork: `paper_figures/outputs/fig1/` through `paper_figures/outputs/fig7/`.
- Formal authority and hashes: `paper_figures/outputs/main_figures_promotion_manifest.json`.
- Default plot-only bundles: `paper_figures/outputs/provenance/fig1_fig2/` and `paper_figures/outputs/provenance/fig3/` through `fig7/`.
- Approved source origins remain preserved in `paper_figure_redesign_20260811/` and `paper_figure_candidates/`; the formal provenance copies are the default destinations for future redraws.
- The author-confirmed Fig.5d draft redraw is in `paper_figure_candidates/manuscript_fig5_reader_first_v3/`: the analytically zero passive series is retained in source data but omitted from artwork. The pre-change bundle is archived at `paper_figures/archive/backups/manuscript_fig5_reader_first_v3_before_passive_removal_20260818/`; formal promotion remains separate.

### Supplementary Fig. S1–S7

- S1–S6: `paper_figure_multi_seed/supplementary_v5_c5_revised_20260804_r2/`
- S7: `paper_figure_multi_seed/supplementary_v5_s7_complete_pairs_20260812_r1/`

The formal-output promotion is separate from DOCX embedding. The current embedded PNGs in `docs/paper/v6.docx` and `docs/paper/supplementary_information.docx` remain pinned by `docs/paper/PAPER_AUTHORITY.json` until a separate manuscript-integration decision.

## Current direct parent roots

The current main/supplementary source-manifest closure points into:

```text
causal_closure_multi_seed_20260803/
multi_seed_rollout/fig1_time_binned_firing/
multi_seed_rollout/fig2/
paper_figure_multi_seed/fig1_functional_stsp_substrate/
paper_figure_multi_seed/fig2_fixed_b_mechanism_confirmatory/
paper_figure_multi_seed/fig2_pair_fused_stsp_state/
paper_figure_multi_seed/fig3_multiitem_peak_landscape/
paper_figure_multi_seed/fig4_accumulated_history_statistics/
paper_figure_multi_seed/fig4_overlap_reentry/
paper_figure_multi_seed/fig5_local_support_competition/
paper_figure_multi_seed/fig6_peak_amplified_reentry/
paper_figure_multi_seed/new_results_reanalysis/
paper_figure_multi_seed/supplementary_v5/
paper_figures/outputs/
```

`data/MNIST/` is the canonical dataset source. Historical result metadata may still record `MNIST` or machine-specific absolute dataset roots, but those paths are provenance only. `multi_snn/` contains protected checkpoints. These roots remain active even when their latest file is older than the archive cutoff.

### Promoted Fig.4 parent producer

The producer formerly stranded at `.codex/tmp/fig4_candidate_statistics_20260731_KEEP/build_fig4_candidate_statistics.py` is now available as:

```bash
python -m src.experiments.runners.fig4_accumulated_history_statistics \
  --output-dir results/paper_figure_multi_seed/fig4_accumulated_history_statistics
```

The promotion leaves the scientific analysis unchanged. A SHA-256-pinned fallback resolves the moved formal-Fig.2b consistency-check input to the byte-identical panel in the current 20260810 bundle. A full replay to a temporary output reproduced all 10 data/metrics files byte-for-byte; only path-bearing metadata differed. See `archive/move_ledgers/temporary_provenance_promotion_20260814.json`.

## Open lineage gap

Current main and S1–S6 manifests reference 40 missing upstream files (1,660,149,353 bytes total):

- 20 × `panel_b_early_firing_transition_metrics.csv`;
- 20 × `panel_d_l1_stsp_perturbation_unit_transitions.csv`.

See `docs/paper/PARENT_ARTIFACT_GAP_REGISTER_20260814.json` for paths, expected sizes, hashes and manifest consumers. The derived bundles remain present and plot-only evidence is not automatically invalidated, but full upstream `require` replay is blocked until byte-identical parents are restored or a new versioned lineage contract is approved. Never mutate frozen bundles to hide the gap.

## Normalized result layout

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

Compatibility files at the result root remain intentional. Plotting must remain a leaf consumer and must not rerun simulation.

## Existing archives

- `results/archive/`: historical result bundles not referenced by the current authority.
- `paper_figure_multi_seed/archive/`: superseded manuscript bundle variants and internal audit results.
- `paper_figures/archive/`: old rendered packs, backups and build inputs.
- Root `archive/results_legacy_20260728/`: pre-reorganization legacy results.

Archived content is provenance, not current manuscript evidence. Do not consolidate `docs/archive/`, `results/archive/` and root `archive/` into a deeper physical tree: existing Windows paths already approach the 260-character boundary, and 19 archived symlinks point to active result parents.

## Archive gate

A result may move only when all conditions hold:

1. it is not named by `PAPER_AUTHORITY.json`;
2. no current source/artifact manifest points to it;
3. it is not required by the selected reproducibility level;
4. it is not a checkpoint, dataset, current manual asset or parent artifact;
5. a move ledger records source, destination, size, hashes and restore path;
6. post-move plot-only, link, manifest and parent-hash validation passes.
