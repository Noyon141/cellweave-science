---
trigger: model_decision
description: "Use for dataset validation, QC metrics, Scrublet/doublet handling, QC artifacts, thresholds, or filtering."
---

# Validation / QC / Filtering

Stage 0 validates:
- dimensions
- sparse/dense representation
- cell identifiers
- gene identifiers
- metadata
- species
- sample assignment
- count availability

Malformed scientific input blocks analysis.

QC per cell:
- total counts
- genes detected
- mitochondrial percentage
- ribosomal metrics
- doublet score

QC is analyzed per sample, not only globally.

Default doublet implementation: Scrublet.

Store:
doublet_score, predicted_doublet, algorithm version, parameters.

Do not immediately delete predicted doublets.

QC review is human-gated.
The science engine may calculate/suggest thresholds, but must not silently apply universal destructive filtering.

Confirmed filtering produces immutable QC decision/provenance through the system contract and downstream branch.
