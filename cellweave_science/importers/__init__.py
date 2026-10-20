"""Dataset importer interfaces, abstractions, and shared Step D security/checksum primitives.

Prescribes the DatasetImporter protocol per System Design Section 121 and Phase 3:
- detect()
- validate()
- inspect()
- canonicalize() -> CanonicalizationResult

Provides shared primitives for:
- streaming SHA-256 and deterministic aggregate sourceChecksum;
- physical file inspection and magic-byte detection;
- bounded decompression and path security;
- count matrix and sparse matrix validation;
- identifier validation and species detection;
- canonical AnnData construction and validation conforming to Policy B.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cellweave_science.contracts import (
    CanonicalizationResult,
    GeneIdScheme,
    ImportFileRole,
    ImportInspectionResult,
    MetadataMappingSubmission,
    Organism,
    ScientificIssue,
    SourceFormat,
)
from cellweave_science.importers.canonical import (
    build_canonical_anndata,
    validate_canonical_anndata,
)
from cellweave_science.importers.checksum import (
    DEFAULT_CHUNK_SIZE_BYTES,
    SourceChecksumFile,
    compute_file_sha256,
    compute_source_checksum,
)
from cellweave_science.importers.errors import (
    CanonicalValidationError,
    CountValidationError,
    DecompressionLimitError,
    ImporterSecurityError,
    MalformedArchiveError,
)
from cellweave_science.importers.security import (
    DEFAULT_MAX_COMPRESSION_RATIO,
    DEFAULT_MAX_UNCOMPRESSED_BYTES,
    GZIP_MAGIC,
    HDF5_MAGIC,
    MATRIX_MARKET_MAGIC,
    STREAM_CHUNK_SIZE,
    CompressionFormat,
    PhysicalContainerFormat,
    PhysicalInspectionResult,
    detect_compression,
    detect_physical_format,
    inspect_physical_file,
    open_bounded_gzip_stream,
    validate_trusted_local_path,
)
from cellweave_science.importers.sources import ImportSourceFile, ImportSourceSet
from cellweave_science.importers.validation import (
    FLOAT_INTEGER_TOLERANCE,
    detect_ensembl_organism_candidates,
    ensure_csr_matrix,
    validate_count_matrix,
    validate_identifiers,
    validate_import_source_set_cardinality,
)


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


__all__ = [
    # Contracts re-exports
    "ImportFileRole",
    "SourceFormat",
    # Protocols and sources
    "DatasetImporter",
    "ImportSourceFile",
    "ImportSourceSet",
    # Checksum primitives
    "DEFAULT_CHUNK_SIZE_BYTES",
    "SourceChecksumFile",
    "compute_file_sha256",
    "compute_source_checksum",
    # Security and container detection
    "HDF5_MAGIC",
    "GZIP_MAGIC",
    "MATRIX_MARKET_MAGIC",
    "DEFAULT_MAX_UNCOMPRESSED_BYTES",
    "DEFAULT_MAX_COMPRESSION_RATIO",
    "STREAM_CHUNK_SIZE",
    "CompressionFormat",
    "PhysicalContainerFormat",
    "PhysicalInspectionResult",
    "validate_trusted_local_path",
    "detect_compression",
    "detect_physical_format",
    "inspect_physical_file",
    "open_bounded_gzip_stream",
    # Validation primitives
    "FLOAT_INTEGER_TOLERANCE",
    "validate_count_matrix",
    "ensure_csr_matrix",
    "validate_identifiers",
    "detect_ensembl_organism_candidates",
    "validate_import_source_set_cardinality",
    # Canonical builder & validator
    "build_canonical_anndata",
    "validate_canonical_anndata",
    # Errors
    "ImporterSecurityError",
    "DecompressionLimitError",
    "MalformedArchiveError",
    "CountValidationError",
    "CanonicalValidationError",
]
