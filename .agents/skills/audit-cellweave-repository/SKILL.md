---
name: audit-cellweave-repository
description: Performs a read-only pre-implementation audit of the current CellWeave repository against the authoritative system design, phase gates, ownership boundaries, and unresolved decisions.
---

# Audit CellWeave Repository

Do not edit code during this skill.

1. Read root `AGENTS.md`.
2. Read `docs/architecture/CellWeave-Production-Master-System-Design-v1.0.pdf` sections relevant to this repository.
3. Read `docs/architecture/IMPLEMENTATION-ORDER.md`.
4. Read `docs/architecture/DO-NOT-GUESS.md`.
5. Inspect current repository structure/configuration.
6. Identify:
   - repository-owned responsibilities already implemented
   - missing required responsibilities for current phase
   - architecture-boundary violations
   - cross-tenant/security risks
   - contract duplication/drift
   - missing tests/gates
   - unresolved design inputs
   - work that belongs in another CellWeave repository
7. Determine the highest fully satisfied implementation phase and the next unmet exit criterion.
8. Report findings without proposing product scope changes.
