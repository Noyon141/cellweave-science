---
name: change-condition-de
description: Changes biological condition differential expression or pathway analysis with sample-aware/pseudobulk safeguards, reference versioning, warnings, and reproducibility.
---

# Change Condition DE / Pathways

1. Distinguish marker discovery from biological condition comparison.
2. Validate sample count, conditions, replicates, batches, and cells.
3. Default toward sample-level/pseudobulk when biological replicates exist.
4. Never use cells as independent biological replicates for condition inference.
5. Emit scientific warnings for weak design/confounding/insufficient data.
6. Record method/parameters/software versions.
7. For pathways record database/version/gene universe/multiple-testing correction.
8. Ensure commercial licensing review is represented before production use of external resources.
9. Add known-treatment golden regression tests for DE direction/pathway expectations.
10. Treat algorithm/reference changes as versioned behavior.
