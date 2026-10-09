---
trigger: always_on
description: "Enforces the CellWeave Phase 0–14 dependency order and phase exit criteria."
---

# Implementation Order / Phase Gates

Read `docs/architecture/IMPLEMENTATION-ORDER.md`.

Do not implement a later-phase dependency as though earlier foundations already exist when the required earlier exit criterion is unmet.

When a task belongs to a later phase:
- identify prerequisites;
- verify they exist and their exit criteria are satisfied;
- if a prerequisite is absent, do not fabricate it inside the wrong repository/task;
- report the dependency clearly.

The system-design phase order is authoritative.
