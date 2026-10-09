---
name: add-or-change-science-step
description: Implements or changes a CellWeave scientific pipeline step with versioned parameters, provenance, deterministic behavior, resource metadata, lineage, and scientific regression.
---

# Add or Change Scientific Step

1. Confirm the step is authorized by the design and belongs to V1.
2. Confirm the input/output contract and canonical AnnData expectations.
3. Preserve raw counts and immutable historical outputs.
4. Define/version parameters and deterministic seeds where applicable.
5. Record software/reference/model versions required by provenance.
6. Produce numerical outputs plus visualization artifacts where applicable.
7. Register artifact lineage and checksums through the output contract.
8. Respect logical resource profile and ceilings.
9. Emit scientific warnings rather than fabricating certainty.
10. Add unit tests.
11. Add golden/regression expectations with scientifically meaningful tolerances.
12. Benchmark runtime/memory/cost impact.
13. Create a new pipeline/step version when behavior changes; never mutate an already-used production version.
