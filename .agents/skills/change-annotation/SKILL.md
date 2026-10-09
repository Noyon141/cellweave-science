---
name: change-annotation
description: Changes automated or manual cell annotation while preserving confidence, Unassigned behavior, model/reference provenance, and revision history.
---

# Change Annotation

1. Confirm organism support and provider abstraction.
2. Record predicted label, confidence, method, model name/version, reference dataset, timestamp.
3. Low confidence must produce `Unassigned` rather than forced certainty.
4. Preserve manual accept/reject/replace/merge/split behavior.
5. Never erase automated provenance after manual correction.
6. Version model/reference changes.
7. Add accuracy/regression tests on authorized golden datasets.
8. Measure resource impact for scVI/scANVI/GPU paths.
9. Do not select an unspecified production model as canonical without an authorized decision.
