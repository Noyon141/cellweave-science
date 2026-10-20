# CellWeave Science — Scientific Compute Engine

`cellweave-science` is the downstream single-cell RNA-seq scientific analysis engine for CellWeave.

## Architectural Boundaries

This repository owns:
- Canonical dataset validation & ingestion into AnnData (`canonical.h5ad`)
- Sample-aware Quality Control (QC) & doublet marking
- Normalization, Highly Variable Gene (HVG) selection, PCA, neighbors, and UMAP embeddings
- Leiden community clustering
- Cell-type annotation with strict confidence metrics and `Unassigned` categorization
- Cluster marker discovery & sample-aware pseudobulk differential expression
- Biological pathway & gene set enrichment analysis
- Browser-optimized visualization artifact generation
- Reproducibility bundles & immutable run manifests
- Modal application deployment harness

This repository **never**:
- Handles customer authentication (owned by `cellweave-web` / BetterAuth)
- Stores business billing records or interacts directly with Polar (owned by `cellweave-web`)
- Serves customer-facing REST APIs
- Receives primary PostgreSQL administrative credentials
- Executes arbitrary user-supplied code
- Logs raw scientific count matrices or expression vectors

## Directory Layout

```
cellweave_science/
  importers/                # DatasetImporter protocol, format detectors, & canonical publication
  publication.py            # Capability-based immutable canonical artifact publication boundary
  validation/               # Stage 0 canonical dataset validation protocol
  qc/                       # QC metrics & doublet detection protocols
  normalization/            # Library-size normalization & HVG protocols
  embeddings/               # PCA, neighbor graph, & UMAP protocols
  clustering/               # Leiden community detection protocols
  annotation/               # AnnotationProvider protocols (Human & Mouse)
  differential_expression/  # Marker discovery & pseudobulk DE protocols
  pathways/                 # PathwayProvider enrichment protocols
  visualization/            # Compact visualization artifact generator protocols
  exports/                  # Reproducibility bundle generator protocols
  provenance/               # Immutable RunManifest & lineage tracker protocols
  contracts/                # Consumed Pydantic v2 models generated from cellweave-contracts
  seeds/                    # Deterministic random seed utilities
  logging/                  # Structured JSON logging with scientific data sanitization
  adapters/                 # ComputeAdapter interface (LOCAL vs MODAL execution modes)

modal_app/                  # Modal App definition, container image, & healthcheck
benchmarks/                 # Data size benchmarking harness (10k, 50k, 100k, 500k, 1M cells)
reference_data/             # Golden reference dataset catalog and manifests
tests/                      # Pytest unit, golden dataset, and adapter test suites
docs/adr/                   # Architecture Decision Records (ADRs)
```

## Quality & Standards

- **Strict Type Hints**: PEP 561 typed package (`py.typed`), checked with `mypy` in strict mode.
- **Linting & Formatting**: Ruff configured with strict rules (`[tool.ruff]`).
- **Testing**: `pytest` test suite with deterministic fixtures.
- **Logging**: Structured JSON logging only; raw counts, matrices, and expression vectors are automatically sanitized.
