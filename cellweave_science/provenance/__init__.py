"""Provenance tracking and run manifest generation protocols.

Prescribes creation of immutable RunManifest and ArtifactManifest objects
per System Design Sections 33 & 34.
Guarantees full computational lineage and complete reproducibility.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from cellweave_science.contracts import ArtifactManifest, RunManifest


@runtime_checkable
class ProvenanceTracker(Protocol):
    """Protocol for recording immutable scientific lineage and execution metadata."""

    def record_step_execution(
        self,
        step_id: str,
        parameters: dict[str, Any],
        software_versions: dict[str, str],
        random_seeds: dict[str, int],
    ) -> dict[str, Any]:
        """Record step-level metadata and software environment."""
        ...

    def compile_run_manifest(self, run_id: str) -> RunManifest:
        """Compile complete immutable RunManifest upon run completion."""
        ...

    def create_artifact_manifest(
        self,
        artifact_id: str,
        run_id: str,
        producing_step_id: str,
    ) -> ArtifactManifest:
        """Generate provenance-linked ArtifactManifest for a derived output."""
        ...


__all__ = ["ProvenanceTracker"]
