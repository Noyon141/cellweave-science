---
name: change-qc-filtering
description: Changes CellWeave QC, doublet detection, threshold recommendation, or filtering while preserving per-sample analysis and the mandatory human review gate.
---

# Change QC / Filtering

1. Verify exact scientific requirement and pipeline version.
2. Preserve per-sample QC behavior.
3. Preserve required metrics.
4. Preserve Scrublet default unless explicit authorized change exists.
5. Mark predicted doublets before any user-confirmed exclusion.
6. Keep suggestions separate from confirmed filtering decisions.
7. Do not introduce universal silent thresholds.
8. Ensure raw input remains intact.
9. Record algorithm versions/parameters.
10. Update QC visual artifact data contracts if needed.
11. Add regression tests across human, mouse, multi-sample, batch-effect datasets.
12. Treat changed scientific behavior as a new pipeline/step version.
