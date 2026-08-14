# V5 Methods terminology and name alignment

Date: 2026-08-08  
Status: manuscript-facing terminology contract

## Purpose

This document separates implementation identifiers from scientific names before the Methods is rewritten. The internal-identifier columns below exist only to verify provenance. They must never be copied into the manuscript, figure labels, captions, equations or supplementary prose.

The manuscript must name a quantity by its scientific role, measurement and intervention scope, not by the variable, task, file, module or endpoint name used to produce it.

## Authority order

Terminology is aligned to:

1. `docs/paper/CORE_SCIENTIFIC_LOGIC_CONTRACT.md`;
2. `docs/paper/RESULTS_EVIDENCE_BOUNDARIES.md`;
3. the current v5 Introduction, Results and figure captions;
4. the current final-six panel claims;
5. `src/` only for determining what a stored quantity or intervention actually is.

Code names are provenance evidence, not prose authority.

## Hard naming rules

1. Define the scientific object first; introduce a mathematical symbol only if it is used repeatedly.
2. Never expose snake_case, CamelCase, task IDs, experiment IDs, module names, artifact keys or enum values in manuscript-facing prose.
3. Distinguish a state from its change: a successor state is a boundary state, whereas a successor update is the change that produced it.
4. Distinguish the joint utilization/resource state from its product: joint `u/x` is a two-variable state; `u x` is derived effective STSP support.
5. Distinguish effective STSP support from excitatory synaptic conductance. Only the model’s conductance variable may be called conductance.
6. Use history-conditioned for processing or updating, history-specific for state identity or morphology, and history-dependent only for broad consequences.
7. Use firing-silent for the measured absence of spikes and activity-silent for the inferred synaptic maintenance regime.
8. Use iterative updating across successive inputs rather than unqualified recurrent updating, which could be mistaken for recurrent connectivity in this feedforward network.
9. Define donor and receiver as experimental roles; never use native, swap or sham without a scientific qualifier.
10. Every intervention name must state what was changed and what was held fixed.

## Canonical conceptual hierarchy

| Scientific position | Canonical English term | Allowed short form after first use | Meaning |
|---|---|---|---|
| Before the current input | inherited STSP state | inherited state | Layer-specific joint utilization/resource configuration available at an input boundary |
| Current computation | history-conditioned processing of the current input | history-conditioned processing | Processing driven by the current input and modulated by inherited STSP |
| Downstream result | downstream successor STSP state | successor state | Downstream boundary state produced after the current input |
| Change producing that result | downstream successor update | successor update | Input-associated change from the preceding boundary to the successor state |
| Use at the next boundary | successor-state reuse | successor reuse | A successor state acting as an inherited condition for the next input |
| Repetition over inputs | iterative updating across successive inputs | iterative updating | Repeated application of the transition motif; does not imply recurrent connectivity |
| State after a sequence | terminal sequence state | terminal state or accumulated state | Retained STSP state after multiple inputs and the terminal delay |
| Later functional consequence | conditional effect of retained STSP on later processing | conditional effect | Influence expressed only under the tested cue/content/pathway conditions |

## Recommended Methods subsection names

| Working title | Canonical subsection title |
|---|---|
| Input encoding and network model | Input encoding and network model |
| Spiking and STSP dynamics | Spiking and short-term synaptic plasticity dynamics |
| Training and fixed circuit | Training and fixed-circuit simulations |
| Episode simulation and restore | State formation, boundary-state capture and controlled restoration |
| Exact input and formation | Exact-input history conditioning and downstream successor formation |
| Reuse and progressive transitions | Successor reuse and iterative updating across successive inputs |
| State morphology | Morphology of accumulated STSP states |
| Functional access and overlap | Conditional effects of retained STSP on later processing |
| Statistics | Statistics and reproducibility |

## Internal implementation to formal manuscript mapping

### Core model and state variables

| Internal-only identifier or pattern | Formal manuscript name | Do not write |
|---|---|---|
| `SDNN_Network` | three-layer feedforward spiking network with presynaptic STSP | SDNN object, network class |
| `DoGSpikeEncoder` | Difference-of-Gaussians response-rank temporal encoder | encoder class, DoGSpikeEncoder |
| `layer1`, `layer2`, `layer3` | Layer 1, Layer 2 and Layer 3 | lowercase layer labels in prose |
| `u_pre` or stored `u` | presynaptic utilization variable, \(u\) | u_pre |
| `x_pre` or stored `x` | available-resource variable, \(x\) | x_pre |
| paired `u` and `x` arrays | joint \(u/x\) STSP state | ux tensor, ux configuration without first definition |
| `gain`, stored `g`, `G_final`, `G_baseline`, or an explicit `u*x` product | effective STSP support, \(u x\) | conductance, g map, gain tensor |
| long-term kernel multiplied by `u*x` | STSP-scaled synaptic efficacy or STSP-scaled presynaptic drive, according to context | raw gain |
| `g_e` | excitatory synaptic conductance | STSP support |
| `v_mem` | membrane potential | v_mem |
| `res` | refractory state | res counter |
| `inh_trace` | lateral-inhibition state | inhibition trace variable |
| `v_mem`, `g_e`, `res`, `inh_trace` collectively | fast network state | non-ux state |
| `kernels` | fixed long-term feedforward weights | kernels, weight tensors |
| prediction sentinel `-1` | no-response outcome | negative prediction |

### Simulation and restoration conditions

| Internal-only identifier | Formal manuscript name | Required clarification |
|---|---|---|
| `dynamic` | dynamic-STSP condition | utilization and resources evolve with time and presynaptic spikes |
| `static_frozen` | static-frozen STSP control | STSP remains at baseline and no actual STSP mutation occurs |
| `full_boundary` | full boundary-state restoration | both retained STSP and matched fast state are restored |
| `stsp_only` | STSP-only restoration with fast-state reinitialization | only selected layer-wise STSP is retained as memory |
| `stsp_only_legacy_current_ux` | no manuscript term; legacy implementation mode | never manuscript-facing |
| `free` branch | unmanipulated dynamic branch | distinguishes it from the event-replay branch |
| `replay` branch | matched presynaptic-event replay branch | state the layer and replayed event sequence |
| `passive`, `natural_decay` | equal-time passive-evolution branch | do not imply that it is a fitted natural-decay model |
| `native` | intact receiver condition | define receiver history and held-fixed variables |
| `layer2_swap`, `l2_swap` | selective Layer 2 successor-transplant condition | only the post-current-input Layer 2 STSP successor is replaced |
| `own_sham` | identity-matched sham transplant | donor and receiver identities are the same |
| `S0` | no-memory reference state | S0 string in prose |
| `S_final` | terminal sequence state | S_final string in prose |
| singleton state keys | slot-matched single-item state | singleton tensor/reference key |
| pair-state keys | two-item retained state | fused state |

### Input and history roles

| Internal-only identifier | Formal manuscript name | Notes |
|---|---|---|
| fixed-B task or protocol name | exact-input assay or identical-current-input assay | never call the experiment fixed-B in the manuscript |
| `exact_b_spikes` | identical encoded current input \(B\) | identity is at the encoded-spike level |
| `c_input`, C anchor | identical encoded next input \(C\) | used in successor-reuse assay |
| B or C `anchor_id` | sampled current-input or next-input exemplar | anchor is an implementation sampling term |
| history conditions `A` and `C` | paired inherited-history conditions | use donor/receiver or history 1/history 2 when A/C would conflict with input labels |
| `donor_condition` | donor history | define the state supplied by the donor |
| `receiver_condition` | receiver history | define all receiver states retained under intervention |
| `history_family_id` | matched history pair | family ID is never manuscript-facing |
| `prefix_k` | history depth \(K\) | do not write prefix depth |
| `seq_len` | sequence length \(K\) | define locally; do not use bare \(K\) across sections without its meaning |
| `delay_ms` | retention delay | report units, not the code suffix |

### Exact-input conditioning and successor formation

| Internal-only endpoint or variable | Canonical manuscript name | Prohibited alternative |
|---|---|---|
| `same_B_common_update_cosine` | common-update cosine similarity | same-B cosine |
| common component in prose | common input-driven update | shared update as a second competing canonical name |
| `total`, \(T\) | total history contrast | total tensor difference |
| `local`, \(L\) | inherited-state contrast under matched event replay | local replay tensor |
| `gamma`, \(Γ\) | history-conditioned processing residual | gamma signal without definition |
| `processing_residual_gamma_energy_fraction` | history-conditioned residual norm ratio | energy fraction; the implemented numerator is a norm, not squared energy |
| event–residual endpoint | residual magnitude at history-differential events relative to size-matched random events | gamma enrichment score |
| behavioral rescue endpoint | history-enabled rescue rate | rescued accuracy |
| behavioral loss endpoint | history-associated loss rate | loss accuracy |
| `accuracy_drop` | accuracy decrease relative to the stated reference condition | accuracy-drop variable |
| overlap-aligned group | overlap-aligned sites | define overlap between retained support and the incoming pathway |
| probe-only group | input-engaged non-overlap sites | probe-only variable/group |
| non-overlap control | non-overlap STSP reset control | nonoverlap mask |
| random-matched control | equal-size random STSP reset control | random mask |
| `advance` | spike advancement relative to the static-frozen response | advance event code |
| `recruit` | spike recruitment relative to the static-frozen response | recruit flag |
| `loss` event | spike loss relative to the static-frozen response | loss flag without distinguishing behavioral loss |
| Layer 2 write-back endpoint | history-aligned downstream STSP updating | write-back score as the main formal name |
| Layer-1-only Layer-2 donor-transfer endpoint | Layer 2 successor donor-transfer index after selective substitution of inherited Layer 1 STSP | layer1_only endpoint string |
| early class-score transfer endpoint | donor-directed shift in the early Layer 3 class-score vector | early_class_score endpoint |

### Successor reuse and iterative updating

| Internal-only endpoint or variable | Canonical manuscript name | Notes |
|---|---|---|
| `post_b` | post-\(B\) Layer 2 successor state | a boundary state, not merely an update vector |
| `early_layer2_event_map` | early Layer 2 event pattern | event map may be retained only in a figure label after definition |
| `early_layer2_event_map_donor_transfer` | donor-transfer index for the early Layer 2 event pattern after successor transplantation | never expose endpoint name |
| `layer3_successor_ux` | post-\(C\) Layer 3 joint \(u/x\) successor state | not a Layer 3 output score |
| `layer3_successor_ux_donor_transfer` | donor-transfer index for the post-\(C\) Layer 3 successor state | never expose endpoint name |
| `donor_transfer_index` | donor-transfer index | do not abbreviate as DTI because of its established diffusion-imaging meaning |
| `state_displacement` | input-associated state displacement | state-displacement variable |
| `natural_decay_displacement` | equal-time passive displacement | natural-decay displacement |
| `observed_minus_natural_decay` | observed-minus-passive displacement | observed-minus-decay |
| stage index | transition stage | distinguish it from history depth and sequence length |

### Morphology of accumulated states

| Internal-only identifier or metric | Canonical manuscript name | Boundary |
|---|---|---|
| concatenated `u` and `x` feature vector | joint \(u/x\) STSP state vector | state, not effective support |
| `beta` | non-negative singleton-template coefficient | beta coefficient only after equation definition |
| `p_i` | normalized constituent weight, \(p_i\) | item probability |
| `N_eff` | effective component number, \(N_{\mathrm{eff}}\) | never effective item count, accessible-item count or capacity |
| `N_eff_fraction` | normalized effective component number | never memory-strength fraction |
| `latest_mass` | latest-item weight | latest mass |
| `latest_collapse_index` | latest-item dominance index | use only if the endpoint remains active |
| `serial_COM` | serial-position center of mass | supplementary unless explicitly retained |
| `reconstruction_R2` | reconstruction coefficient of determination | supplementary method term |
| `g_final` | terminal effective STSP support map | terminal conductance map |
| `g_baseline` | baseline effective STSP support map | baseline conductance map |
| `delta_support` | change in effective STSP support | delta-g |
| gain ratio | relative retained-support map, defined as terminal divided by baseline effective STSP support | conductance ratio |
| effective-area endpoint | effective area of retained STSP support | conductance area |
| matched-minus-deranged endpoint | history-matched morphology advantage over sequence-deranged composites | deranged score |
| coefficient-free endpoint | coefficient-free morphology specificity | mixture-free capacity |

### Conditional effects on later processing

| Internal-only identifier or metric | Canonical manuscript name | Required clarification |
|---|---|---|
| `keep_prob` | cue keep probability or retained cue fraction | state which cue elements were retained |
| cue type `matched` | matched-image cue | exact retained exemplar |
| cue type `mismatched` | same-label novel-image cue | never call it a mismatched cue; its class label matches the target |
| cue type `unseen` | unseen-class cue | define as a cue whose class was absent from the retained sequence |
| `P_target` | target-readout probability | not recall probability unless the task definition warrants it |
| `P_silent` | silent-response probability | not error rate |
| `target_memory_gain` | target-readout gain from the retained STSP state | memory gain without an explicit reference |
| normalized cue AUC endpoint | normalized area-under-the-curve gain across cue strength | AUC gain only after first definition |
| `rescued_fraction` | functionally rescued fraction | never effective component number, retained-item count or capacity |
| `local_score` | local retained-support score | local score without scientific definition |
| `overlap_map` | local input-pathway overlap | overlap score without the pathway definition |
| high/low STSP groups | high- and low-retained-support sites | STSP is a process, not a scalar group label |
| `interaction_delta` | overlap-gated STSP interaction | interaction score |
| high-STSP ablation task | targeted removal of the high-support, input-overlapping contribution | high-STSP ablation |
| matched ablation control | exact area- and input-energy–matched removal control | matched mask |
| load-by-delay product endpoint | standardized sequence-length-by-delay interaction | KxD score |

### Statistical units and summaries

| Internal-only identifier | Canonical manuscript name | Rule |
|---|---|---|
| `network_seed` | independently trained network | seed identifies a network; it is not itself the inferential unit name |
| paired cells, trials, events, sites | lower-level observations | aggregate within network before cross-network inference |
| `valid_coverage` | valid-case coverage | supplementary/protocol term unless it determines a stated exclusion |
| positive fraction across networks | fraction of networks with a positive effect | do not treat lower-level cells as independent replication |
| bootstrap draws | bootstrap resamples | state only when defining the confidence interval |

## First-use definitions for drafting

The following definitions may be adapted directly, without implementation names:

- “We refer to the layer-specific joint \(u/x\) configuration immediately before an input as the inherited STSP state.”
- “The downstream joint \(u/x\) configuration produced after that input is termed the successor STSP state.”
- “The product \(u x\) is termed effective STSP support; it scales presynaptic drive and is distinct from excitatory synaptic conductance.”
- “The common input-driven update denotes the component shared across inherited histories under an identical encoded input, whereas the history-conditioned residual denotes the additional component associated with history-dependent processing.”
- “Successor-state reuse was tested by selectively transplanting the post-\(B\) Layer 2 successor while preserving the receiver’s other retained STSP states, equalizing fast network state and presenting an identical encoded \(C\).”
- “Morphology and conditional function were assessed in separate protocols and were not treated as measurements of one common memory-strength variable.”

## Controlled adjective and hyphenation rules

Use consistently:

- activity-silent STSP state;
- firing-silent delay;
- history-conditioned processing or updating;
- history-specific state or morphology;
- input-driven update;
- inter-layer state transition;
- successor state as a noun;
- successor-state formation or successor-state reuse as compound modifiers;
- static-frozen STSP control;
- equal-time passive evolution;
- no-memory reference;
- single-item, two-item and multi-item;
- overlap-aligned sites;
- same-label novel-image cue;
- area- and input-energy–matched removal;
- Layer 1, Layer 2 and Layer 3.

Use working memory as a noun and working-memory as an adjective.

## Mandatory corrections to the current v5 wording

1. Replace Layer 1 “conductance map” wherever the quantity is the product \(u x\) with Layer 1 “effective STSP support map.” The code separately represents excitatory conductance, so the two names are not interchangeable.
2. Replace “processing-residual energy fraction” with “history-conditioned residual norm ratio.”
3. Replace “mismatched cue” with “same-label novel-image cue.”
4. Replace “natural-decay displacement” with “equal-time passive displacement.”
5. Replace unqualified “recurrent updating” with “iterative updating across successive inputs,” or explicitly state that recurrence refers to the transition motif rather than network connectivity.
6. Replace C5, fixed-B, native, swap and own-sham labels with the scientific intervention names above.
7. Use downstream STSP updating or successor-state formation as the primary term instead of write-back.
8. Do not abbreviate donor-transfer index as DTI.

## Retired or prohibited manuscript terms

Do not use as active v5 scientific names:

- chunk-like state;
- fused state or fusion;
- fixed-B protocol;
- C5 assay;
- re-entry as the main mechanism name;
- peak, valley, basin, score or mask without a scientific definition;
- conductance for \(u x\);
- energy fraction for a norm ratio;
- mismatched cue for the same-label novel-image condition;
- natural decay for the equal-time passive branch;
- effective item count, accessible-item count or capacity for \(N_{\mathrm{eff}}\);
- memory strength as a common label for morphology and conditional function;
- Fig. 6 access to the morphology defined by Fig. 5;
- static-frozen STSP update as an actual mutation;
- causal necessity, complete mediation or uniqueness for successor transplantation;
- recurrent network dynamics for the feedforward architecture;
- DTI for donor-transfer index.

## Manuscript-facing QA checklist

Before any Methods paragraph is accepted:

1. Search for underscores, class names, task IDs, endpoint strings and path fragments.
2. Confirm that every `g`-like quantity has been classified as either effective STSP support or excitatory conductance.
3. Confirm that every state-transfer sentence names the changed layer, retained layers, fast-state treatment and identical input.
4. Confirm that state and update are not used interchangeably.
5. Confirm that history-conditioned, history-specific and history-dependent follow the role rules above.
6. Confirm that firing-silent and activity-silent are used at the correct evidential level.
7. Confirm that equal-time passive and static-frozen controls are not merged.
8. Confirm that structural effective component number and functional rescued fraction remain separate.
9. Confirm that the same-label novel-image cue is never called merely mismatched.
10. Confirm that no retired chunk/fusion/re-entry label has returned through legacy code or figure names.
