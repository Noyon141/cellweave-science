"""Dataset importer interfaces and abstractions.

Prescribes the DatasetImporter protocol per System Design Section 121 and Phase 3:
- detect()
- validate()
- inspect()
- canonicalize() -> CanonicalizationResult

Algorithm implementations belong to subsequent Phase 3 steps.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cellweave_science.contracts import (
    CanonicalizationResult,
    GeneIdScheme,
    ImportInspectionResult,
    MetadataMappingSubmission,
    Organism,
    ScientificIssue,
)
from cellweave_science.importers.sources import ImportSourceFile, ImportSourceSet


@runtime_checkable
class DatasetImporter(Protocol):
    """Protocol implemented by format-specific dataset importers."""

    def detect(self, source_set: ImportSourceSet) -> bool:
        """Determine if this importer supports the given source file set."""
        ...

    def validate(self, source_set: ImportSourceSet) -> list[ScientificIssue]:
        """Validate structural integrity and format cardinality before canonicalization."""
        ...

    def inspect(self, source_set: ImportSourceSet) -> ImportInspectionResult:
        """Extract dataset dimensions, detect candidate organism/gene schemes, and check if mapping is required."""
        ...

    def canonicalize(
        self,
        source_set: ImportSourceSet,
        output_path: str,
        organism: Organism,
        gene_id_scheme: GeneIdScheme,
        reference_build: str | None = None,
        mapping: MetadataMappingSubmission | None = None,
    ) -> CanonicalizationResult:
        """Convert source dataset to canonical AnnData (.h5ad) conforming to Policy B (counts in X)."""
        ...


__all__ = ["DatasetImporter", "ImportSourceFile", "ImportSourceSet"]
