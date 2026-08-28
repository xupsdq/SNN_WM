---
status: accepted
---

# Masse-first LIF-STSP student mainline

The recurrent working-memory extension will first reproduce the full Masse rate-STSP task as an independently usable behavioral baseline and teacher, then train a full-task sparse Dale LIF student with STSP active from the first training step; the project claims biological constraints for the student's forward circuit, not for its global teacher/BPTT learning rule. Single-direction recurrent-LIF work remains a kernel and causal tracer, while a full-task no-STSP network is deferred to a final failure-isolation fallback rather than treated as a parallel main baseline, because pretraining it would preferentially establish a persistent-activity solution that need not recruit STSP.
