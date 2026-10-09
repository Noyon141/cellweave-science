---
trigger: always_on
description: "Prevents false production-ready claims and requires the system-level CellWeave production Definition of Done."
---

# Production Readiness

Read `docs/architecture/PRODUCTION-DEFINITION-OF-DONE.md`.

A deployment is not proof of production readiness.

Any repo-level release claim must account for the system invariants it can affect:
durability, deduplication, billing idempotency, tenant isolation, large-data routing, immutable provenance, reproducibility, exports, deletion, provider detection, restore, cancellation, scientific regression gates, diagnostics, reconciliation, cost ceilings.
