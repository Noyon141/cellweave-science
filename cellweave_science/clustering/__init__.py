"""Clustering protocols and abstractions.

Prescribes Leiden community detection clustering per System Design Section 24.
Algorithm implementations belong to Phase 6.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ClusteringEngine(Protocol):
    """Protocol for single-cell clustering methods."""

    def cluster_leiden(
        self,
        input_path: str,
        resolution: float = 1.0,
        output_path: str = "",
    ) -> dict[str, Any]:
        """Compute Leiden cluster assignments with deterministic seeds."""
        ...


__all__ = ["ClusteringEngine"]
