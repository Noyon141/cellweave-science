"""Marker gene discovery and differential expression protocols.

Prescribes MarkerGeneDiscovery and DifferentialExpressionEngine protocols
per System Design Sections 27 & 28.
Distinguishes cluster-level marker discovery from sample-level pseudobulk DE.
Implementations belong to Phase 8.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MarkerGeneDiscovery(Protocol):
    """Protocol for cluster-level marker gene identification."""

    def find_markers(
        self,
        input_path: str,
        group_key: str,
        method: str = "wilcoxon",
    ) -> dict[str, Any]:
        """Identify cluster marker genes."""
        ...


@runtime_checkable
class DifferentialExpressionEngine(Protocol):
    """Protocol for sample-aware condition-level differential expression."""

    def run_differential_expression(
        self,
        input_path: str,
        group_a: str,
        group_b: str,
        sample_key: str,
    ) -> dict[str, Any]:
        """Compute pseudobulk or sample-aware condition DE."""
        ...


__all__ = ["DifferentialExpressionEngine", "MarkerGeneDiscovery"]
