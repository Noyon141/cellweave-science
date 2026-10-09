---
name: add-importer
description: Implements a CellWeave dataset importer through the DatasetImporter contract with explicit validation, no ambiguous guessing, safe canonical AnnData conversion, and golden importer tests.
---

# Add Importer

1. Verify importer is in current authorized scope; do not implement future Seurat/Parse without explicit authorization.
2. Implement the importer abstraction: detect, validate, inspect, convert, metadata.
3. Validate actual structure/signature, not only filename.
4. Reject malformed or scientifically invalid data.
5. Require explicit mapping for ambiguous generic matrices.
6. Convert to canonical AnnData / `canonical.h5ad`.
7. Preserve/import species, gene identifiers, sample metadata, counts, layers as required.
8. Produce deterministic metadata and checksums through the system contract.
9. Add malformed/security test fixtures.
10. Add golden importer fixtures and verify canonical AnnData.
11. Do not execute embedded/user code.
