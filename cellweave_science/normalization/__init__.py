"""Normalization and feature selection protocols.

Prescribes library-size normalization and highly variable gene (HVG) selection
per System Design Sections 21 & 22.
Algorithm implementations belong to Phase 6.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class NormalizationEngine(Protocol):
    """Protocol for normalizing counts while strictly preserving raw count layers."""

    def normalize(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Perform library-size normalization and log1p transformation."""
        ...

    def select_hvg(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Identify highly variable genes accounting for batch/sample metadata."""
        ...


__all__ = ["NormalizationEngine"]
