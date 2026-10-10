"""Dataset importer interfaces and abstractions.

Prescribes the DatasetImporter protocol per System Design Section 121:
- detect()
- validate()
- inspect()
- convert() -> canonical.h5ad
- metadata()

Algorithm implementations belong to Phase 3.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DatasetImporter(Protocol):
    """Protocol implemented by format-specific dataset importers."""

    def detect(self, file_path: str) -> bool:
        """Determine if this importer supports the given source file."""
        ...

    def validate(self, file_path: str) -> bool:
        """Validate structural integrity of source dataset before import."""
        ...

    def inspect(self, file_path: str) -> dict[str, Any]:
        """Extract dataset dimensions, cell/gene identifiers, and metadata without full load."""
        ...

    def convert(self, file_path: str, output_path: str) -> str:
        """Convert source dataset to canonical AnnData (.h5ad)."""
        ...

    def metadata(self, file_path: str) -> dict[str, Any]:
        """Extract canonical dataset metadata."""
        ...


__all__ = ["DatasetImporter"]
