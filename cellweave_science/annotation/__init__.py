"""Cell-type annotation interfaces and definitions.

Prescribes AnnotationProvider protocol per System Design Section 25:
- Human and Mouse support
- Strict confidence tracking
- Explicit 'Unassigned' category for low-confidence calls
- Provider metadata (model name, model version, reference dataset)

Implementations belong to Phase 7.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AnnotationProvider(Protocol):
    """Protocol for cell-type annotation engines."""

    @property
    def provider_name(self) -> str:
        """Name of the annotation provider."""
        ...

    @property
    def model_version(self) -> str:
        """Version of the underlying reference model."""
        ...

    def annotate(
        self,
        input_path: str,
        organism: str,
        min_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Perform automated cell-type annotation."""
        ...


__all__ = ["AnnotationProvider"]
