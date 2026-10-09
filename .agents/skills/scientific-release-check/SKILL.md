---
name: scientific-release-check
description: Runs the CellWeave scientific release gate across typing, tests, golden datasets, regression, SBOM, vulnerability scan, benchmarks, provenance, and immutable pipeline versioning.
---

# Scientific Release Check

Block release unless applicable checks pass:
- Python lint
- type checks
- unit tests
- scientific regression
- golden datasets
- container build
- SBOM
- vulnerability scan
- benchmark
- provenance completeness
- deterministic seeds where applicable
- input/output contract compatibility
- resource ceiling behavior
- no secret/raw matrix logging
- pipeline version immutability
- benchmark-qualified data-size behavior
- scientific warnings preserved
- RUO boundary preserved

Unit tests alone are never enough for algorithm changes.
