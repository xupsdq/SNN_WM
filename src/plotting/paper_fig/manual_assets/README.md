# Manual Assets

Place hand-drawn or externally prepared paper assets here.

Current optional/expected asset:

- `fig1a_architecture.svg`: horizontal STSP-SNN architecture schematic for Fig.1A. If absent, Fig.1 renders a programmatic schematic and records a QC warning.
- Fig.2A is rendered programmatically as a two-item episode schematic in the current standalone paper-figure pipeline; a future manual asset can be added here without changing the experiment outputs.
- Fig.3A is rendered programmatically as a multi-item sequence schematic; a future manual asset can replace it without changing the Fig.3 experiment bundle.
- Fig.4A is rendered programmatically as a DMS sample-delay-probe overlap re-entry schematic. A future manual schematic may replace it, and absence of a manual asset should remain warning-only.
- Fig.5 is data-driven from the standalone local-support competition bundle. If a future local-neighborhood schematic is added, it should be supplemental/contextual only; Fig.5A-E paper panels must continue to read canonical experiment outputs rather than manual figure images.
- Fig.6F is rendered programmatically from `panel_f_global_mechanism_metadata.json` in the standalone Fig.6 peak-amplified re-entry bundle. A future manual schematic may be added as an optional visual replacement, but the route/gain language and causal-language warnings must continue to come from the experiment summary and paper_fig QC.

The build system treats missing manual assets as QC warnings rather than hard failures.
