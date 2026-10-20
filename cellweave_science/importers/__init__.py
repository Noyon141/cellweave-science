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
- count matrix classification (VALID_COUNTS, NOT_COUNTS, AMBIGUOUS) and sparse safety;
- identifier validation and conservative species detection;
- canonical AnnData construction and validation conforming to Policy B.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cellweave_science.contracts import (
    CanonicalArtifactPublicationCapabilities,
    CanonicalArtifactPublicationResult,
    CanonicalArtifactPublicationStatus,
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
    MalformedCompressedStreamError,
)
from cellweave_science.importers.generic import (
    GenericMatrixConfig,
    GenericMatrixImporter,
    read_delimited_table,
    sniff_delimiter,
)
from cellweave_science.importers.h5ad import H5ADImporter
from cellweave_science.importers.publication import (
    CanonicalArtifactCapabilityExpiredError,
    CanonicalArtifactIntegrityError,
    CanonicalArtifactPublicationError,
    CanonicalArtifactTransportError,
    CanonicalPublicationTransport,
    HttpsCapabilityTransport,
    LocalFileStatSnapshot,
    PutDisposition,
    RemoteStreamVerification,
    publish_canonical_artifact,
    redact_capability_url,
    validate_local_canonical_h5ad_file,
    validate_publication_capabilities,
)
from cellweave_science.importers.security import (
    GZIP_MAGIC,
    HDF5_MAGIC,
    MATRIX_MARKET_MAGIC,
    STREAM_CHUNK_SIZE,
    CompressionFormat,
    DecompressionLimits,
    PhysicalContainerFormat,
    PhysicalInspectionResult,
    detect_compression,
    detect_physical_format,
    inspect_physical_file,
    open_bounded_gzip_stream,
    validate_trusted_local_path,
)
from cellweave_science.importers.sources import ImportSourceFile, ImportSourceSet
from cellweave_science.importers.tenx_h5 import TenxH5Importer
from cellweave_science.importers.tenx_mtx import TenxMtxImporter
from cellweave_science.importers.tenx_shared import (
    GENE_EXPRESSION_MODALITY,
    TenxFeature,
    TenxImporterConfig,
    parse_10x_barcodes,
    parse_10x_features_tsv,
    parse_10x_matrix_market,
    parse_matrix_market,
    validate_10x_features,
)
from cellweave_science.importers.validation import (
    CountClassification,
    check_ensembl_species_conflict,
    classify_count_matrix,
    detect_ensembl_organism_candidates,
    ensure_csr_matrix,
    has_opposing_ensembl_species,
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
    "CanonicalArtifactPublicationCapabilities",
    "CanonicalArtifactPublicationResult",
    "CanonicalArtifactPublicationStatus",
    "ImportFileRole",
    "SourceFormat",
    # Protocols and sources
    "DatasetImporter",
    "H5ADImporter",
    "TenxH5Importer",
    "TenxMtxImporter",
    "TenxImporterConfig",
    "GenericMatrixConfig",
    "GenericMatrixImporter",
    "read_delimited_table",
    "sniff_delimiter",
    "TenxFeature",
    "GENE_EXPRESSION_MODALITY",
    "parse_10x_barcodes",
    "parse_10x_features_tsv",
    "validate_10x_features",
    "parse_10x_matrix_market",
    "parse_matrix_market",
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
    "STREAM_CHUNK_SIZE",
    "CompressionFormat",
    "PhysicalContainerFormat",
    "DecompressionLimits",
    "PhysicalInspectionResult",
    "validate_trusted_local_path",
    "detect_compression",
    "detect_physical_format",
    "inspect_physical_file",
    "open_bounded_gzip_stream",
    # Validation primitives
    "CountClassification",
    "classify_count_matrix",
    "validate_count_matrix",
    "ensure_csr_matrix",
    "validate_identifiers",
    "detect_ensembl_organism_candidates",
    "check_ensembl_species_conflict",
    "has_opposing_ensembl_species",
    "validate_import_source_set_cardinality",
    # Canonical builder & validator
    "build_canonical_anndata",
    "validate_canonical_anndata",
    # Canonical publication
    "CanonicalArtifactPublicationError",
    "CanonicalArtifactCapabilityExpiredError",
    "CanonicalArtifactIntegrityError",
    "CanonicalArtifactTransportError",
    "CanonicalPublicationTransport",
    "HttpsCapabilityTransport",
    "LocalFileStatSnapshot",
    "PutDisposition",
    "RemoteStreamVerification",
    "publish_canonical_artifact",
    "redact_capability_url",
    "validate_local_canonical_h5ad_file",
    "validate_publication_capabilities",
    # Errors
    "ImporterSecurityError",
    "DecompressionLimitError",
    "MalformedCompressedStreamError",
    "CountValidationError",
    "CanonicalValidationError",
]
