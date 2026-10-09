---
trigger: model_decision
description: "Use for marker-vs-condition differential expression, pseudobulk safeguards, pathway/enrichment analysis, scientific warnings, or reference versioning."
---

# Differential Expression / Pathways

Distinguish:
- cluster-vs-cluster marker discovery
- biological condition comparison

Cell-level analysis is acceptable for explicit marker discovery.

For Treatment vs Control / Drug A vs Drug B / Disease vs Healthy:
default toward sample-level/pseudobulk when biological replicates exist.

Never treat thousands of cells as thousands of independent biological replicates.

Comparison validation considers:
- biological sample count
- condition assignment
- replicates
- batch structure
- cell counts

Weak comparison quality produces a scientific warning.

Pathway analysis uses `PathwayProvider` abstraction.
Every result records:
- database
- database version
- gene universe
- method
- parameters
- multiple-testing correction

External reference/pathway licensing requires review before commercial production.

A biological reference database must never silently update beneath an existing analysis.

Do not invent a DE/pathway tool or database if not already authorized.
