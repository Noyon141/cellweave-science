# ADR-001: AnnData Canonical Internal Representation

## Status
Accepted

## Context
Downstream single-cell RNA-seq analysis involves multiple source formats (e.g. 10x cellranger h5, 10x mtx, generic matrix TSV/CSV, AnnData h5ad). Operating downstream pipelines directly against vendor-specific formats creates fragmentation, duplication, and inconsistent metadata tracking.

## Decision
All imported single-cell datasets in CellWeave must be canonicalized to `AnnData` (`canonical.h5ad`) upon ingest. Downstream scientific computation, quality control, normalization, embeddings, clustering, and differential expression operate exclusively against this canonical representation. Raw counts are strictly preserved in an AnnData layer or raw slot.

## Consequences
- Single downstream execution path for all scientific pipeline steps.
- Preservation of raw counts and observation/variable metadata in standardized data structures.
- Evolution path toward cloud-native chunked formats (such as Zarr) supported without rewriting pipeline algorithms.
