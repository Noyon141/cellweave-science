"""Dataset validation interfaces and definitions.

Prescribes Stage 0 input validation per System Design Section 19:
- dimensions
- sparsity/density
- cell & gene identifiers
- sample assignment
- species compatibility
- raw count availability
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class DatasetValidator(Protocol):
    """Protocol for pre-pipeline scientific dataset validation."""

    def validate_canonical(self, canonical_path: str) -> Dict[str, Any]:
        """Validate canonical AnnData file before pipeline execution."""
        ...


__all__ = ["DatasetValidator"]
