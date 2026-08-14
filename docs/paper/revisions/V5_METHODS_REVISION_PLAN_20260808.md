# V5 Methods revision plan

Date: 2026-08-08

## Scope

This document records the revision strategy executed against the then-current `docs/paper/v5.docx` baseline, whose original path no longer exists; the current product is `docs/paper/v5_submission_ready_20260808.docx`. It is based on the completed Introduction and Results in v5, the Methods in `docs/archive/paper/v3.docx`, the current `src/` implementation, and Masse et al., *Circuit mechanisms for the maintenance and manipulation of information in working memory*.

The intended product is a scientific Methods section, not a code report. The manuscript must describe the model, experimental contrasts, intervention scope, central measurements and statistical design, while excluding runtime architecture, artifact management and implementation-facing identifiers.

## Main decision

Use the v3 Methods as the prose baseline. Preserve its four core sections, add only the two transition-method families required by the current Results, and refocus the morphology and functional-access sections on the current Fig. 5 and Fig. 6 claims.

The current v5 Methods contains 12 headings and approximately 2,714 extractable words, excluding equation objects. The v3 Methods contains 8 headings and approximately 2,124 extractable words. The proposed structure contains 9 headings and should remain close to 2,200–2,400 words, excluding equations.

## Scientific order

The Methods must support the following dependency order:

1. a functional activity-silent inherited STSP state;
2. conditioning of an identical later input by inherited history;
3. formation of a downstream successor state;
4. reuse of that successor by the next input;
5. recurrence across successive inputs;
6. two parallel outcome modules: accumulated-state morphology and conditional functional expression.

Morphology and functional expression are parallel outcomes. The Methods must not imply that the functional-access assays read out the morphology quantified in the morphology assays.

## Proposed Methods structure

### 1. Input encoding and network model

Retain the v3 organization and wording wherever possible:

- MNIST input;
- Difference-of-Gaussians ON/OFF preprocessing;
- response-rank temporal spike encoding;
- three-layer feedforward STSP-enabled spiking network;
- class-selective earliest-spike readout;
- retention of silent or no-response trials.

Detailed encoder constants belong in a parameter table or supplementary methods rather than the main prose.

### 2. Spiking and short-term synaptic plasticity dynamics

Retain the v3 model description and core equations:

- conductance-based leaky integrate-and-fire dynamics;
- refractory periods, local top-k competition and lateral inhibition;
- presynaptic utilization and resource variables;
- effective STSP support;
- dynamic STSP and static-frozen control conditions.

Do not repeat these dynamics in later assay sections.

### 3. Training and fixed-circuit simulations

Retain the v3 STDP and reward-modulated STDP framework, but make the actual regime explicit:

- dynamic STSP was disabled during training;
- the learned kernels were loaded into the STSP-enabled network after training;
- baseline gain compensation was applied when STSP was enabled;
- all long-term weights were fixed during working-memory assays;
- the main cohort comprised 20 independently trained networks, seeds 1000–1019.

Training-loop implementation and file-level checkpoint handling do not belong in the manuscript.

### 4. Episode simulation, state capture and restoration

Retain the v3 common episode logic, while distinguishing the state operations required by the current experiments:

- complete boundaries may contain STSP and fast network variables;
- functional cue/readout assays restore the selected STSP state after reinitializing fast variables;
- transition and transplantation assays either restore a matched complete boundary or equalize fast variables before replacing only the specified STSP layer;
- the Layer 3 decision state is cleared before a new readout period;
- static-frozen STSP and equal-time passive evolution are distinct controls.

### 5. Exact-input conditioning and successor formation

Add a compact section supporting Figs. 2–3:

1. Describe the identical-input design, aligned and mismatched histories, and the separate rescue and loss opportunity sets.
2. Define the common input-driven update and history-conditioned residual using the free and matched-event-replay branches with passive-evolution correction.
3. Describe overlap-aligned reset or attenuation, non-overlap and equal-size random controls, early firing-event classification, downstream write-back, and the selective inherited-state transfer used to test successor formation.

Retain only the central decomposition equations. Protocol hashes, frozen artifacts, audit gates and source-file identifiers are excluded.

### 6. Successor reuse and progressive transitions

Add a compact section supporting Fig. 4:

1. Describe selective transplantation of the post-input Layer 2 successor into a matched receiver, preservation of the receiver’s other retained states, fast-state equalization, and presentation of an identical next input.
2. Define donor-directed transfer for early Layer 2 processing and the downstream successor formed after the next input.
3. Describe observed-next-input versus equal-time passive displacement across stages 2–10 and the separate shallow-versus-deeper rescue/loss comparison.

The obsolete early-versus-late stage average that does not support a current main panel should not remain in the main Methods.

### 7. Accumulated-state morphology assays

Refocus the v3 morphology section on Fig. 1 and Fig. 5:

- delay-period decoding from layer-specific joint STSP states;
- retention of both pair constituents;
- experienced-pair specificity against constituent-controlled shuffled pairs;
- non-negative decomposition over slot-matched singleton references;
- effective component number and latest-item contribution;
- Layer 1 effective-area and matched-versus-deranged morphology analyses.

Remove or move to supplementary methods:

- chunk or fused-state terminology;
- whole-pair residual indices not used by the current main figures;
- additive, convex and unconstrained alternative mixture models;
- exploratory cross-fitted interactions;
- saturating-versus-linear model comparisons.

### 8. Functional access and overlap-gated expression

Refocus the v3 functional-access and overlap sections on Fig. 6:

- pair-state partial-cue recovery;
- sequence-state access relative to cue-only and slot-matched singleton states;
- matched, same-label novel and unseen cue controls;
- sequence-length-by-delay dependence;
- high-overlap contribution tested against exact area-and-energy-matched removal;
- overlap-gated interaction between retained STSP and incoming pathways.

Neutral pings, region-gated pings, old/middle/recent ping composition, dynamic-pattern indices and decision-deflection indices should appear only in supplementary methods if the corresponding supplementary results remain active.

### 9. Statistics and reproducibility

Replace the generic v3 statistics language with the current design:

- the independently trained network is the inferential unit;
- lower-level observations are aggregated within network and condition;
- descriptive panels are separated from inferential endpoints;
- Student t tests and Benjamini–Hochberg adjustment are used only for the stated families;
- Fig. 4 uses exact sign-flip tests, bootstrap confidence intervals and Holm correction as specified;
- directional and two-sided endpoints are distinguished;
- silent/no-response trials remain in unconditional denominators;
- rescue and loss retain separate opportunity denominators;
- undefined quantities are not imputed.

## Main-text boundary

### Must remain in the main Methods

- core model and STSP equations;
- the training-time versus post-training STSP regime;
- fixed long-term weights;
- encoded-input identity in exact-input assays;
- restoration and fast-state controls;
- the scope of both selective state-transfer interventions;
- the distinction between static-frozen and passive controls;
- central endpoint definitions;
- inference unit, tests and multiplicity correction.

### Move to supplementary methods or source-data definitions

- complete thresholds, quantiles, windows and clipping settings;
- image and anchor identifiers;
- sampling recipes and sensitivity analyses;
- alternative mixture models;
- supplementary ping and trajectory assays;
- detailed edge-case tables.

### Exclude from manuscript prose

- runner or module names;
- cache and artifact-reuse behavior;
- protocol digests and hash checks;
- parent-artifact immutability checks;
- implementation-facing identifiers and audit labels.

## Equation policy

Preserve the v3 core model and training equations. Add only equations that define central current claims:

- exact-input update decomposition;
- donor-directed transfer;
- effective component number;
- state displacement if prose is insufficient.

Cue-gain normalization, regression implementation and edge-case formulas should be stated in prose or moved to supplementary methods unless required to interpret a main endpoint.

## Revision workflow

1. Copy the v3 Methods as the prose baseline.
2. Preserve the first four sections and make only code-grounded factual corrections.
3. Add the exact-input/successor-formation and successor-reuse/recurrence sections.
4. Reassign the retained morphology, functional-access and overlap prose to the current Fig. 5–6 roles.
5. Apply the formal terminology alignment before writing any new paragraph.
6. Cross-check every current Fig. 1–6 panel against one Methods paragraph.
7. Remove every main-text method that has neither a current main-panel role nor an explicitly retained supplementary role.
8. Reconcile the statistics wording with the current Results and captions.

## Acceptance criteria

- All six current main figures are covered.
- No implementation-facing identifier appears in manuscript-facing prose.
- The two selective state-transfer interventions cannot be confused.
- Static-frozen and equal-time passive controls cannot be confused.
- Training-time STSP status is explicit.
- Morphology and conditional function remain parallel modules.
- Structural effective component number is not described as accessible-item count.
- A static-frozen update opportunity is not described as an actual STSP mutation.
- No retired chunk/fusion claim is reintroduced.
- The main Methods remains close to the v3 level of detail.

## Authority files used for factual checks

- `docs/paper/v5.docx` (historical baseline path; identity retained in the V5 manifests)
- `docs/paper/v5_submission_ready_20260808.docx` (current product)
- `docs/archive/paper/v3.docx`
- `docs/paper/CORE_SCIENTIFIC_LOGIC_CONTRACT.md`
- `docs/paper/RESULTS_EVIDENCE_BOUNDARIES.md`
- `src/core/network.py`
- `src/training/train_sdnn.py`
- `src/experiments/common/model_io.py`
- `src/experiments/common/monitored_dms.py`
- `src/experiments/paper_figures/final_six/pipeline.py`
- current exact-input, successor-transfer, progressive-update, morphology, cue-specificity and overlap-gating experiment modules under `src/experiments/`
