"""Pathway and enrichment analysis interfaces.

Prescribes PathwayProvider protocol per System Design Section 29.
Encapsulates biological reference database versions and multiple-testing corrections.
Implementations belong to Phase 8.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class PathwayProvider(Protocol):
    """Protocol for pathway/gene-set enrichment engines."""

    @property
    def database_name(self) -> str:
        """Name of the pathway reference database."""
        ...

    @property
    def database_version(self) -> str:
        """Version of the pathway database."""
        ...

    def run_enrichment(
        self,
        genes: List[str],
        organism: str,
        universe: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform enrichment analysis on a gene list."""
        ...


__all__ = ["PathwayProvider"]
