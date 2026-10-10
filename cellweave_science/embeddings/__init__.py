"""Dimensionality reduction and embedding protocols.

Prescribes PCA, neighbor graph construction, and UMAP computation
per System Design Section 23.
Algorithm implementations belong to Phase 6.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingEngine(Protocol):
    """Protocol for calculating dimensionality reductions and neighbor graphs."""

    def compute_pca(self, input_path: str, n_comps: int = 50) -> dict[str, Any]:
        """Compute PCA embedding on highly variable features."""
        ...

    def build_neighbors(self, input_path: str, n_neighbors: int = 15) -> dict[str, Any]:
        """Compute k-nearest neighbor graph."""
        ...

    def compute_umap(self, input_path: str, output_path: str) -> dict[str, Any]:
        """Compute 2D UMAP projection and store coordinates."""
        ...


__all__ = ["EmbeddingEngine"]
