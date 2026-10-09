"""Quality control (QC) and doublet detection protocols.

Prescribes QC metrics generation, doublet marking, and QC decision structures
per System Design Sections 19 & 20.
Algorithm implementations belong to Phase 5.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class QCEngine(Protocol):
    """Protocol for per-sample single-cell quality control calculation."""

    def compute_qc_metrics(self, canonical_path: str) -> Dict[str, Any]:
        """Compute per-cell and per-sample QC metrics without mutating raw data."""
        ...

    def run_doublet_detection(self, canonical_path: str) -> Dict[str, Any]:
        """Calculate doublet scores and mark candidate doublets without silent deletion."""
        ...


__all__ = ["QCEngine"]
