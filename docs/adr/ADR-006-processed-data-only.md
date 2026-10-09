# ADR-006: Processed Data Only Scope

## Status
Accepted

## Context
Raw sequencing processing (FASTQ alignment, read mapping, UMI counting) requires massive compute infrastructure and varies widely across sequencing technologies.

## Decision
CellWeave V1 strictly supports downstream analysis starting from processed count matrices (10x h5/mtx, h5ad, generic CSV/TSV). Processing raw FASTQ or unaligned sequencing data is an explicit non-goal for V1.

## Consequences
- Scope is focused on biological interpretation, QC, clustering, annotation, and comparative analysis.
- Faster turnaround times and bounded compute resource profiles.
