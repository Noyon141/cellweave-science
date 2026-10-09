---
trigger: glob
description: "Applies mandatory Python typing, validation, testing, determinism, dependency, and logging rules."
globs: "**/*.py"
---

# Python Quality

Mandatory:
- type hints
- Pydantic schemas generated/derived from authoritative contracts where cross-repo
- Ruff
- mypy/pyright where practical
- pytest
- dependency pinning
- deterministic seeds
- structured logging

Scientific functions should be pure where practical.

Never `print(adata)` on customer datasets.
Never log:
- matrices
- expression vectors
- raw scientific content
- full filenames containing sensitive metadata

Prefer identifiers and aggregate metrics.
