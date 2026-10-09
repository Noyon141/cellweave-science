---
trigger: model_decision
description: "Use for normalization, HVG, PCA, neighbors, UMAP, Leiden, marker discovery, artifact versioning, or recomputation behavior."
---

# Core Scientific Pipeline

Normalization:
- preserve raw counts
- library-size normalization
- log1p
- never overwrite raw counts

Feature selection:
- highly variable genes
- parameters versioned
- account for batch/sample where appropriate

Dimensional reduction:
PCA → neighbor graph → UMAP.

Store:
- numerical embeddings
- optimized visualization artifacts

Clustering:
- Leiden
- user-selectable validated resolution options
- each clustering change creates a new artifact/version
- never silently mutate historical assignments

Marker discovery:
validated non-parametric methods such as Wilcoxon may be used.
Store gene, cluster, score, log fold change, p-value, adjusted p-value, method, parameters.

Recompute:
- presentation/color changes do not rerun scientific stages
- annotation changes do not rerun UMAP
- filter changes invalidate downstream artifacts and branch a new run

Outputs form a DAG with lineage.
