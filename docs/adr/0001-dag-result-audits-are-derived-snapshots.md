---
status: accepted
---

# DAG/result audits are derived snapshots, not authority

DAG/result audit documents are dated, derived snapshots and do not replace runtime task registries, paper-authority records, or result manifests. This preserves discoverability while avoiding a new static task/artifact map that silently becomes stale; future audits must record their snapshot identity and continue to resolve conflicts through the established authority order.
