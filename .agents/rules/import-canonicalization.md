---
trigger: model_decision
description: "Use for H5AD/10x/matrix import, format detection, validation, canonical AnnData conversion, or future importer abstractions."
---

# Import / Canonicalization

V1 importers:
- H5AD
- 10x filtered H5
- grouped 10x MTX + features + barcodes
- CSV/TSV/MTX generic matrix with explicit mapping

Importer interface concept:
detect, validate, inspect, convert, metadata.

H5AD validation includes:
AnnData structure, X, obs, var, dimensions, identifiers, layers, species metadata, gene identifier format.

Generic matrix mapping must explicitly map:
cells, genes, sample identifiers, conditions, batches, replicates.

Ambiguous input is never silently guessed.

All supported input formats convert to canonical AnnData:
`canonical.h5ad`.

Downstream V1 science operates on canonical representation.

Future Seurat/Parse support remains future unless explicitly authorized. Seurat conversion must remain isolated from the core Python environment.
