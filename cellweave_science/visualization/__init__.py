"""Visualization artifact generation protocols.

Prescribes generation of lightweight browser-consumable visualization artifacts
(UMAP coordinates, cluster metadata, QC distributions) per System Design Section 74.
Ensures large raw matrices never need to be downloaded to client browsers.
Implementations belong to Phase 9.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class VisualizationArtifactGenerator(Protocol):
    """Protocol for extracting lightweight visualization tables from AnnData."""

    def generate_umap_artifact(
        self,
        input_path: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """Generate compact coordinates and cluster indices for client rendering."""
        ...

    def generate_qc_distribution_artifact(
        self,
        input_path: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """Generate QC histogram and violin distribution points."""
        ...


__all__ = ["VisualizationArtifactGenerator"]
