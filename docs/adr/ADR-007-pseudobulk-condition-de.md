# ADR-007: Pseudobulk Condition Differential Expression

## Status
Accepted

## Context
Treating individual single cells as independent biological replicates in condition-level comparisons leads to inflated false-positive rates and severe p-value distortion.

## Decision
CellWeave defaults to sample-level / pseudobulk differential expression whenever biological replicates are available across conditions. Cluster-level marker detection may use single-cell rank-sum tests, but condition comparisons (Treatment vs Control) require sample-aware modeling and biological replicate validation.

## Consequences
- Scientifically sound differential expression that guards against false discoveries.
- Clear distinction in the scientific engine between marker discovery and condition-level comparison.
