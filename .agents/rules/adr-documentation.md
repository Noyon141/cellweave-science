---
trigger: always_on
description: "Requires CellWeave ADR discipline and documentation updates as part of Definition of Done."
---

# ADR / Documentation Discipline

Maintain system documentation for:
architecture, data model, security, scientific methods, API, runbooks, deployment, incident response, billing, retention, pipeline validation.

Documentation changes are part of feature Definition of Done.

Maintain `/docs/adr`.

Baseline architectural decisions include:
- AnnData canonical representation
- R2 storage
- Temporal durable workflows
- Modal initial compute provider
- Polar billing
- processed-data-only
- pseudobulk condition DE
- no PHI
- separate repositories

Changing a baseline decision is not an implementation refactor. It requires explicit architectural review.

Never let an agent-generated ADR silently override the authoritative system design.
