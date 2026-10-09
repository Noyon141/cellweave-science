# CellWeave Science — Agent Constitution

## Authority
`docs/architecture/CellWeave-Production-Master-System-Design-v1.0.pdf` is the scientific and architectural source of truth.

## Repository responsibility
This repository owns the Python scientific engine:
- importers
- validation
- QC
- normalization
- embeddings
- clustering
- annotation
- differential expression
- pathways
- visualization artifact generation
- exports
- provenance
- Modal application
- scientific tests
- benchmarks
- reference data

It is not the SaaS backend and owns no BetterAuth/Polar/customer API logic.

## Scientific principles
- canonical internal representation is AnnData / `.h5ad`
- raw counts are preserved
- QC is sample-aware
- predicted doublets are marked before deletion
- filtering is human-confirmed
- pipeline versions are immutable
- historical outputs are never silently mutated
- annotation uncertainty is explicit
- condition-level DE is sample-aware/pseudobulk by default when replicates are available
- reference/database versions are recorded
- stochastic methods use deterministic seeds where applicable
- every output is reproducible through provenance

## Security
Science jobs receive only scoped input/output capabilities, manifest fragment, and required parameters. They never receive primary database, BetterAuth, Polar, or R2 master credentials.

## Quality
Type hints, Pydantic, Ruff, pytest, pinned dependencies, deterministic seeds, structured logging, and scientific regression/golden datasets are mandatory.

A scientific algorithm change cannot ship merely because unit tests pass.
