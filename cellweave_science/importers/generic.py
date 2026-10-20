"""Generic matrix importer (GENERIC_MATRIX) conforming to Phase 3 System Design and Policy B.

Supports:
- CSV expression matrices (plain and bounded gzip).
- TSV expression matrices (plain and bounded gzip).
- Matrix Market (.mtx / .mtx.gz) expression matrices.
- Optional CELL_METADATA (CSV / TSV).
- Optional FEATURE_METADATA (CSV / TSV).

Invariants:
1. Orientation is never guessed: requires explicit MetadataMappingSubmission.orientation
   (CELLS_BY_GENES vs GENES_BY_CELLS).
2. Raw counts only: validates integer, non-negative, finite counts. Rejects fractional,
   negative, NaN/Inf, and ambiguous near-integral values.
3. Strict sample authority: source values never pass through to sample_key; all mapped
   cells must resolve through explicit SampleMappingEntry.sampleKey. Incomplete mapping fails closed.
4. Deterministic metadata joins: cell and feature metadata joined on explicit identifiers by value,
   not position. Duplicate metadata keys or missing matrix IDs strictly prohibited. Extra unreferenced
   metadata rows are safely ignored.
5. Slot hygiene: Policy B AnnData with SciPy CSR matrix in X, clean layers, empty obsm/varm/uns.
6. Content-based format and delimiter detection: file extensions (.csv/.tsv) are never used
   as authoritative format evidence.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellweave_science.contracts import (
    CanonicalizationResult,
    GeneIdScheme,
    ImportFileRole,
    ImportInspectionResult,
    InspectionStatus,
    MatrixOrientation,
    MetadataMappingSubmission,
    Organism,
    ScientificIssue,
    ScientificIssueCode,
    ScientificIssueSeverity,
    SourceFormat,
)
from cellweave_science.importers.canonical import (
    build_canonical_anndata,
    validate_canonical_anndata,
)
from cellweave_science.importers.checksum import compute_file_sha256
from cellweave_science.importers.errors import (
    CountValidationError,
    DecompressionLimitError,
    ImporterSecurityError,
)
from cellweave_science.importers.security import (
    CompressionFormat,
    DecompressionLimits,
    PhysicalContainerFormat,
    inspect_physical_file,
    open_bounded_gzip_stream,
    validate_trusted_local_path,
)
from cellweave_science.importers.sources import ImportSourceSet
from cellweave_science.importers.tenx_shared import (
    parse_10x_matrix_market,
    stream_text_lines,
)
from cellweave_science.importers.validation import (
    check_ensembl_species_conflict,
    detect_ensembl_organism_candidates,
    ensure_csr_matrix,
    has_opposing_ensembl_species,
    validate_count_matrix,
    validate_identifiers,
    validate_import_source_set_cardinality,
)


@dataclass(frozen=True)
class GenericMatrixConfig:
    """Configuration for generic matrix importer technical safety and resource limits (Fail-Closed).

    Invariants:
    1. decompression_limits is mandatory and must be an instance of DecompressionLimits.
    2. Zero permissive defaults: callers must provide explicit limits.
    """

    decompression_limits: DecompressionLimits

    def __post_init__(self) -> None:
        if not isinstance(self.decompression_limits, DecompressionLimits):
            raise TypeError(
                f"decompression_limits must be an instance of DecompressionLimits, got {type(self.decompression_limits).__name__}"
            )


def sniff_delimiter(file_path: Path | str, limits: DecompressionLimits) -> str:
    """Deterministically detect delimiter (comma or tab) from text stream content.

    Content-based detection invariants:
    1. Delimiters are inferred strictly from stream content, never from file extensions (.csv/.tsv).
    2. A line with commas and 0 tabs is comma-delimited (',').
    3. A line with tabs and 0 commas is tab-delimited ('\\t').
    4. Mixed lines require a >= 2:1 predominance ratio, otherwise FORMAT_MISMATCH is raised.
    5. Undelimited single-column content (0 commas and 0 tabs across content) returns a consistent
       single-column delimiter ('\\t') independent of filename extension.
    6. Empty files or comment-only files raise MALFORMED_INPUT.
    """
    path = validate_trusted_local_path(file_path)
    inspected_any = False

    for raw_line in stream_text_lines(path, limits=limits):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        inspected_any = True
        tab_count = line.count("\t")
        comma_count = line.count(",")
        if tab_count > 0 and comma_count == 0:
            return "\t"
        if comma_count > 0 and tab_count == 0:
            return ","
        if tab_count > 0 and comma_count > 0:
            if tab_count >= comma_count * 2:
                return "\t"
            if comma_count >= tab_count * 2:
                return ","
            raise CountValidationError(
                f"Ambiguous delimiter in {path.name}: contains both commas ({comma_count}) and tabs ({tab_count})",
                issue_code=ScientificIssueCode.FORMAT_MISMATCH,
            )

    if not inspected_any:
        raise CountValidationError(
            f"File {path.name} is empty or contains only comments",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        )

    # Content has no commas and no tabs in any inspected line (undelimited single-column table)
    # Return a deterministic delimiter ('\t') independent of filename extension
    return "\t"


def read_delimited_table(
    file_path: Path | str,
    limits: DecompressionLimits,
    expected_delim: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Read a delimited text table into a DataFrame with bounded decompression safety.

    All columns are initially read as strings to preserve identifier and metadata fidelity.
    """
    path = validate_trusted_local_path(file_path)
    inspection = inspect_physical_file(path)
    delim = expected_delim or sniff_delimiter(path, limits=limits)

    if inspection.compression == CompressionFormat.GZIP:
        raw_bytes = bytearray()
        for chunk in open_bounded_gzip_stream(path, limits=limits):
            raw_bytes.extend(chunk)
        bio = io.BytesIO(raw_bytes)
        try:
            df = pd.read_csv(bio, sep=delim, header=0, dtype=str, index_col=False)
        except Exception as exc:
            raise CountValidationError(
                f"Failed to parse delimited table from {path.name}: {exc}",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            ) from exc
    else:
        try:
            df = pd.read_csv(path, sep=delim, header=0, dtype=str, index_col=False)
        except Exception as exc:
            raise CountValidationError(
                f"Failed to parse delimited table from {path.name}: {exc}",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            ) from exc

    if df.empty or len(df.columns) == 0:
        raise CountValidationError(
            f"Delimited table {path.name} is empty or contains no columns",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        )

    return df, delim


class GenericMatrixImporter:
    """Importer for generic expression matrices (CSV, TSV, MTX) conforming to Policy B."""

    def __init__(self, config: GenericMatrixConfig) -> None:
        if not isinstance(config, GenericMatrixConfig):
            raise TypeError("config must be an instance of GenericMatrixConfig")
        self.config = config

    def detect(self, source_set: ImportSourceSet) -> bool:
        """Determine if this importer supports the given source file set.

        Requirements:
        1. source_format == SourceFormat.GENERIC_MATRIX.
        2. Cardinality: exactly 1 MATRIX file, <=1 CELL_METADATA, <=1 FEATURE_METADATA.
        3. MATRIX file is detectable as MTX (Matrix Market) or delimited text (CSV/TSV).
        """
        if source_set.source_format != SourceFormat.GENERIC_MATRIX:
            return False

        cardinality_issues = validate_import_source_set_cardinality(source_set)
        if cardinality_issues:
            return False

        try:
            matrix_file = source_set.get_single_file_by_role(ImportFileRole.MATRIX)
            path = validate_trusted_local_path(matrix_file.local_path)
            inspection = inspect_physical_file(path)

            if inspection.detected_format == PhysicalContainerFormat.MATRIX_MARKET:
                return True
            if inspection.compression == CompressionFormat.GZIP:
                for chunk in open_bounded_gzip_stream(
                    path, limits=self.config.decompression_limits
                ):
                    if chunk.startswith(b"%%MatrixMarket"):
                        return True
                    break

            # Sniff delimiter for plain or gzip text
            _ = sniff_delimiter(path, limits=self.config.decompression_limits)
            return True
        except (ImporterSecurityError, CountValidationError, OSError):
            return False

    def _detect_matrix_kind(self, path: Path) -> str:
        """Detect whether matrix file is MTX, TSV, or CSV strictly from file contents."""
        inspection = inspect_physical_file(path)
        if inspection.detected_format == PhysicalContainerFormat.MATRIX_MARKET:
            return "MTX"
        if inspection.compression == CompressionFormat.GZIP:
            for chunk in open_bounded_gzip_stream(path, limits=self.config.decompression_limits):
                if chunk.startswith(b"%%MatrixMarket"):
                    return "MTX"
                break

        delim = sniff_delimiter(path, limits=self.config.decompression_limits)
        return "TSV" if delim == "\t" else "CSV"

    def validate(self, source_set: ImportSourceSet) -> list[ScientificIssue]:
        """Validate cardinality, structural parsing, table consistency, duplicate IDs, and raw counts."""
        cardinality_issues = validate_import_source_set_cardinality(source_set)
        if cardinality_issues:
            return cardinality_issues

        try:
            matrix_file = source_set.get_single_file_by_role(ImportFileRole.MATRIX)
            matrix_path = validate_trusted_local_path(matrix_file.local_path)
        except (ValueError, ImporterSecurityError) as exc:
            return [
                ScientificIssue(
                    code=ScientificIssueCode.MALFORMED_INPUT,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Invalid matrix file in source set: {exc}",
                    userActionRequired=True,
                )
            ]

        # 1. Detect and validate MATRIX format
        try:
            matrix_kind = self._detect_matrix_kind(matrix_path)
        except DecompressionLimitError as exc:
            return [
                ScientificIssue(
                    code=ScientificIssueCode.RESOURCE_LIMIT,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Decompression limit exceeded in matrix file {matrix_path.name}: {exc}",
                    userActionRequired=True,
                )
            ]
        except CountValidationError as exc:
            return [
                ScientificIssue(
                    code=exc.issue_code,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Matrix format detection failed for {matrix_path.name}: {exc}",
                    userActionRequired=True,
                )
            ]
        except Exception as exc:  # noqa: BLE001
            return [
                ScientificIssue(
                    code=ScientificIssueCode.MALFORMED_INPUT,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Failed to read matrix file {matrix_path.name}: {exc}",
                    userActionRequired=True,
                )
            ]

        if matrix_kind == "MTX":
            try:
                _n_rows, _n_cols, _n_entries, _coo = parse_10x_matrix_market(
                    matrix_path, limits=self.config.decompression_limits
                )
            except DecompressionLimitError as exc:
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.RESOURCE_LIMIT,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Decompression limit exceeded in MTX file {matrix_path.name}: {exc}",
                        userActionRequired=True,
                    )
                ]
            except CountValidationError as exc:
                return [
                    ScientificIssue(
                        code=exc.issue_code,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Matrix Market validation failed in {matrix_path.name}: {exc}",
                        userActionRequired=True,
                    )
                ]
            except Exception as exc:  # noqa: BLE001
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.MALFORMED_INPUT,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Failed to parse Matrix Market file {matrix_path.name}: {exc}",
                        userActionRequired=True,
                    )
                ]
        else:
            # CSV / TSV validation
            try:
                delim = "\t" if matrix_kind == "TSV" else ","
                df, _ = read_delimited_table(
                    matrix_path,
                    limits=self.config.decompression_limits,
                    expected_delim=delim,
                )
            except DecompressionLimitError as exc:
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.RESOURCE_LIMIT,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Decompression limit exceeded in table {matrix_path.name}: {exc}",
                        userActionRequired=True,
                    )
                ]
            except CountValidationError as exc:
                return [
                    ScientificIssue(
                        code=exc.issue_code,
                        severity=ScientificIssueSeverity.ERROR,
                        message=str(exc),
                        userActionRequired=True,
                    )
                ]
            except Exception as exc:  # noqa: BLE001
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.MALFORMED_INPUT,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Failed to read delimited matrix {matrix_path.name}: {exc}",
                        userActionRequired=True,
                    )
                ]

            if len(df.columns) < 2:
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.MALFORMED_INPUT,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Expression matrix table must have at least 2 columns (found {len(df.columns)})",
                        userActionRequired=True,
                    )
                ]

            # Check duplicate row identifiers (column 0)
            raw_row_ids = df.iloc[:, 0].astype(str).tolist()
            if len(raw_row_ids) != len(set(raw_row_ids)):
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.DUPLICATE_CELL_IDS,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Duplicate identifiers found in row index column of {matrix_path.name}",
                        userActionRequired=True,
                    )
                ]

            # Check duplicate column identifiers in raw header (before pandas mangles them)
            for raw_line in stream_text_lines(matrix_path, limits=self.config.decompression_limits):
                line = raw_line.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                header_tokens = [h.strip() for h in line.split(delim)]
                if len(header_tokens) > 1:
                    raw_feature_tokens = header_tokens[1:]
                    if len(raw_feature_tokens) != len(set(raw_feature_tokens)):
                        return [
                            ScientificIssue(
                                code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                                severity=ScientificIssueSeverity.ERROR,
                                message=f"Duplicate identifiers found in column header of {matrix_path.name}",
                                userActionRequired=True,
                            )
                        ]
                break

            # Validate numeric expression body (columns 1..N)
            expr_df = df.iloc[:, 1:]
            try:
                arr = expr_df.to_numpy(dtype=np.float64)
            except (ValueError, TypeError):
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.MALFORMED_INPUT,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Expression matrix body in {matrix_path.name} contains non-numeric values",
                        userActionRequired=True,
                    )
                ]

            try:
                validate_count_matrix(arr)
            except CountValidationError as exc:
                return [
                    ScientificIssue(
                        code=exc.issue_code,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Raw count validation failed in {matrix_path.name}: {exc}",
                        userActionRequired=True,
                    )
                ]

        # 2. Validate CELL_METADATA if present
        cell_meta_files = source_set.get_files_by_role(ImportFileRole.CELL_METADATA)
        if cell_meta_files:
            try:
                cell_meta_path = validate_trusted_local_path(cell_meta_files[0].local_path)
                cell_df, _ = read_delimited_table(
                    cell_meta_path, limits=self.config.decompression_limits
                )
                if cell_df.empty or len(cell_df.columns) == 0:
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.MALFORMED_INPUT,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"CELL_METADATA table in {cell_meta_path.name} is empty",
                            userActionRequired=True,
                        )
                    ]
                if cell_df.iloc[:, 0].duplicated().any():
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.DUPLICATE_CELL_IDS,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"Duplicate cell identifiers in CELL_METADATA table {cell_meta_path.name}",
                            userActionRequired=True,
                        )
                    ]
            except DecompressionLimitError as exc:
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.RESOURCE_LIMIT,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Decompression limit exceeded in CELL_METADATA: {exc}",
                        userActionRequired=True,
                    )
                ]
            except CountValidationError as exc:
                return [
                    ScientificIssue(
                        code=exc.issue_code,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"CELL_METADATA validation failed: {exc}",
                        userActionRequired=True,
                    )
                ]

        # 3. Validate FEATURE_METADATA if present
        feat_meta_files = source_set.get_files_by_role(ImportFileRole.FEATURE_METADATA)
        if feat_meta_files:
            try:
                feat_meta_path = validate_trusted_local_path(feat_meta_files[0].local_path)
                feat_df, _ = read_delimited_table(
                    feat_meta_path, limits=self.config.decompression_limits
                )
                if feat_df.empty or len(feat_df.columns) == 0:
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.MALFORMED_INPUT,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"FEATURE_METADATA table in {feat_meta_path.name} is empty",
                            userActionRequired=True,
                        )
                    ]
                if feat_df.iloc[:, 0].duplicated().any():
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"Duplicate feature identifiers in FEATURE_METADATA table {feat_meta_path.name}",
                            userActionRequired=True,
                        )
                    ]
            except DecompressionLimitError as exc:
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.RESOURCE_LIMIT,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Decompression limit exceeded in FEATURE_METADATA: {exc}",
                        userActionRequired=True,
                    )
                ]
            except CountValidationError as exc:
                return [
                    ScientificIssue(
                        code=exc.issue_code,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"FEATURE_METADATA validation failed: {exc}",
                        userActionRequired=True,
                    )
                ]

        return []

    def inspect(self, source_set: ImportSourceSet) -> ImportInspectionResult:
        """Extract matrix dimensions, metadata columns, and determine required mapping."""
        if not source_set.dataset_version_id or not source_set.dataset_version_id.strip():
            raise ImporterSecurityError(
                "Cannot inspect dataset without required dataset_version_id on source_set",
                issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
            )
        version_id = source_set.dataset_version_id.strip()

        issues = self.validate(source_set)
        blocking_issues = [iss for iss in issues if iss.severity == ScientificIssueSeverity.ERROR]
        if blocking_issues:
            return ImportInspectionResult(
                datasetVersionId=version_id,
                sourceFormat=SourceFormat.GENERIC_MATRIX,
                status=InspectionStatus.INVALID,
                requiresMapping=False,
                availableCellMetadataColumns=[],
                availableFeatureMetadataColumns=[],
                issues=issues,
                cellCountEstimate=None,
                geneCountEstimate=None,
                detectedOrganism=None,
                detectedGeneIdScheme=None,
                referenceBuildCandidate=None,
            )

        matrix_file = source_set.get_single_file_by_role(ImportFileRole.MATRIX)
        matrix_path = validate_trusted_local_path(matrix_file.local_path)
        matrix_kind = self._detect_matrix_kind(matrix_path)

        # Inspect available metadata columns strictly from metadata tables
        cell_meta_cols: list[str] = []
        cell_meta_files = source_set.get_files_by_role(ImportFileRole.CELL_METADATA)
        if cell_meta_files:
            try:
                cell_df, _ = read_delimited_table(
                    cell_meta_files[0].local_path,
                    limits=self.config.decompression_limits,
                )
                cell_meta_cols = list(cell_df.columns)
            except Exception:  # noqa: BLE001
                pass

        feat_meta_cols: list[str] = []
        feat_meta_files = source_set.get_files_by_role(ImportFileRole.FEATURE_METADATA)
        feat_df_cand: pd.DataFrame | None = None
        if feat_meta_files:
            try:
                feat_df, _ = read_delimited_table(
                    feat_meta_files[0].local_path,
                    limits=self.config.decompression_limits,
                )
                feat_meta_cols = list(feat_df.columns)
                feat_df_cand = feat_df
            except Exception:  # noqa: BLE001
                pass

        # In generic matrices without orientation mapping, orientation is inherently unresolved
        inspection_issues: list[ScientificIssue] = list(issues)
        inspection_issues.append(
            ScientificIssue(
                code=ScientificIssueCode.AMBIGUOUS_ORIENTATION,
                severity=ScientificIssueSeverity.REQUIRES_USER_INPUT,
                message=(
                    "GENERIC_MATRIX orientation (CELLS_BY_GENES vs GENES_BY_CELLS) and "
                    "identifier placements require explicit user confirmation"
                ),
                userActionRequired=True,
            )
        )

        detected_org: Organism | None = None
        detected_scheme: GeneIdScheme | None = None

        cand_id_sources: list[list[str]] = []
        if matrix_kind != "MTX":
            try:
                delim = "\t" if matrix_kind == "TSV" else ","
                df, _ = read_delimited_table(
                    matrix_path,
                    limits=self.config.decompression_limits,
                    expected_delim=delim,
                )
                cand_id_sources.append([str(x).strip() for x in df.iloc[:, 0].tolist()])
                cand_id_sources.append([str(c).strip() for c in df.columns[1:]])
            except Exception:  # noqa: BLE001
                pass

        if feat_df_cand is not None and len(feat_df_cand.columns) > 0:
            cand_id_sources.append([str(x).strip() for x in feat_df_cand.iloc[:, 0].tolist()])

        for cand_ids in cand_id_sources:
            if not cand_ids:
                continue
            if check_ensembl_species_conflict(cand_ids):
                inspection_issues.append(
                    ScientificIssue(
                        code=ScientificIssueCode.UNRESOLVED_SPECIES,
                        severity=ScientificIssueSeverity.ERROR,
                        message="Matrix identifier space contains conflicting Human and Mouse Ensembl IDs",
                        userActionRequired=True,
                    )
                )
                return ImportInspectionResult(
                    datasetVersionId=version_id,
                    sourceFormat=SourceFormat.GENERIC_MATRIX,
                    status=InspectionStatus.INVALID,
                    requiresMapping=False,
                    availableCellMetadataColumns=cell_meta_cols,
                    availableFeatureMetadataColumns=feat_meta_cols,
                    issues=inspection_issues,
                    cellCountEstimate=None,
                    geneCountEstimate=None,
                    detectedOrganism=None,
                    detectedGeneIdScheme=None,
                    referenceBuildCandidate=None,
                )
            org = detect_ensembl_organism_candidates(cand_ids)
            if org is not None:
                detected_org = org
                detected_scheme = GeneIdScheme.ENSEMBL
                break

        return ImportInspectionResult(
            datasetVersionId=version_id,
            sourceFormat=SourceFormat.GENERIC_MATRIX,
            status=InspectionStatus.NEEDS_MAPPING,
            requiresMapping=True,
            availableCellMetadataColumns=cell_meta_cols,
            availableFeatureMetadataColumns=feat_meta_cols,
            issues=inspection_issues,
            cellCountEstimate=None,
            geneCountEstimate=None,
            detectedOrganism=detected_org,
            detectedGeneIdScheme=detected_scheme,
            referenceBuildCandidate=None,
        )

    def canonicalize(
        self,
        source_set: ImportSourceSet,
        output_path: str,
        organism: Organism,
        gene_id_scheme: GeneIdScheme,
        reference_build: str | None = None,
        mapping: MetadataMappingSubmission | None = None,
    ) -> CanonicalizationResult:
        """Convert generic matrix to canonical AnnData conforming to Policy B (counts in X).

        Requires explicit MetadataMappingSubmission with orientation.
        """
        issues = self.validate(source_set)
        blocking_issues = [iss for iss in issues if iss.severity == ScientificIssueSeverity.ERROR]
        if blocking_issues:
            raise ImporterSecurityError(
                f"Cannot canonicalize invalid generic matrix dataset: {blocking_issues[0].message}",
                issue_code=blocking_issues[0].code,
            )

        if mapping is None or mapping.orientation is None:
            raise ImporterSecurityError(
                "GENERIC_MATRIX canonicalization requires explicit MetadataMappingSubmission with orientation",
                issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
            )

        dest_path = Path(output_path).resolve()
        for f in source_set.files:
            if validate_trusted_local_path(f.local_path).resolve() == dest_path:
                raise ImporterSecurityError(
                    "Source file immutability violation: output_path cannot match any source file path",
                    issue_code=ScientificIssueCode.CANONICALIZATION_FAILED,
                )

        matrix_file = source_set.get_single_file_by_role(ImportFileRole.MATRIX)
        matrix_path = validate_trusted_local_path(matrix_file.local_path)
        matrix_kind = self._detect_matrix_kind(matrix_path)

        orientation = mapping.orientation

        # 1. Parse expression matrix and initial identifiers
        if matrix_kind == "MTX":
            _n_rows, _n_cols, _n_entries, coo_matrix = parse_10x_matrix_market(
                matrix_path, limits=self.config.decompression_limits
            )
            cell_meta_files = source_set.get_files_by_role(ImportFileRole.CELL_METADATA)
            feat_meta_files = source_set.get_files_by_role(ImportFileRole.FEATURE_METADATA)

            if not cell_meta_files:
                raise ImporterSecurityError(
                    "Generic Matrix Market requires CELL_METADATA to provide cell identifiers",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if not feat_meta_files:
                raise ImporterSecurityError(
                    "Generic Matrix Market requires FEATURE_METADATA to provide feature identifiers",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )

            cell_df, _ = read_delimited_table(
                cell_meta_files[0].local_path, limits=self.config.decompression_limits
            )
            feat_df, _ = read_delimited_table(
                feat_meta_files[0].local_path, limits=self.config.decompression_limits
            )

            if orientation == MatrixOrientation.CELLS_BY_GENES:
                if len(cell_df) != _n_rows or len(feat_df) != _n_cols:
                    raise CountValidationError(
                        f"Matrix Market dimensions ({_n_rows}, {_n_cols}) do not match "
                        f"CELL_METADATA ({len(cell_df)}) and FEATURE_METADATA ({len(feat_df)})",
                        issue_code=ScientificIssueCode.DIMENSION_MISMATCH,
                    )
                cell_id_col = (
                    mapping.cellIdColumn
                    if mapping.cellIdColumn and mapping.cellIdColumn in cell_df.columns
                    else cell_df.columns[0]
                )
                gene_id_col = (
                    mapping.geneIdColumn
                    if mapping.geneIdColumn and mapping.geneIdColumn in feat_df.columns
                    else feat_df.columns[0]
                )

                raw_cell_ids = cell_df[cell_id_col].astype(str).tolist()
                raw_gene_ids = feat_df[gene_id_col].astype(str).tolist()
                canonical_csr = ensure_csr_matrix(coo_matrix.tocsr())
            else:
                # GENES_BY_CELLS: rows are genes, cols are cells
                if len(feat_df) != _n_rows or len(cell_df) != _n_cols:
                    raise CountValidationError(
                        f"Matrix Market dimensions ({_n_rows}, {_n_cols}) do not match "
                        f"FEATURE_METADATA ({len(feat_df)}) and CELL_METADATA ({len(cell_df)})",
                        issue_code=ScientificIssueCode.DIMENSION_MISMATCH,
                    )
                cell_id_col = (
                    mapping.cellIdColumn
                    if mapping.cellIdColumn and mapping.cellIdColumn in cell_df.columns
                    else cell_df.columns[0]
                )
                gene_id_col = (
                    mapping.geneIdColumn
                    if mapping.geneIdColumn and mapping.geneIdColumn in feat_df.columns
                    else feat_df.columns[0]
                )

                raw_cell_ids = cell_df[cell_id_col].astype(str).tolist()
                raw_gene_ids = feat_df[gene_id_col].astype(str).tolist()
                # Explicit transpose from genes x cells -> cells x genes
                canonical_csr = ensure_csr_matrix(coo_matrix.T.tocsr())

        else:
            # CSV / TSV parsing
            delim = "\t" if matrix_kind == "TSV" else ","
            table_df, _ = read_delimited_table(
                matrix_path,
                limits=self.config.decompression_limits,
                expected_delim=delim,
            )

            if orientation == MatrixOrientation.CELLS_BY_GENES:
                # Rows are cells, cols are genes
                if mapping.cellIdColumn and mapping.cellIdColumn in table_df.columns:
                    raw_cell_ids = table_df[mapping.cellIdColumn].astype(str).tolist()
                    expr_df = table_df.drop(columns=[mapping.cellIdColumn])
                else:
                    # Column 0 is cell IDs
                    raw_cell_ids = table_df.iloc[:, 0].astype(str).tolist()
                    expr_df = table_df.iloc[:, 1:]

                raw_gene_ids = [str(c).strip() for c in expr_df.columns]
                expr_mat = expr_df.to_numpy(dtype=np.float64)
                validate_count_matrix(expr_mat)

                max_val = float(np.max(expr_mat)) if expr_mat.size > 0 else 0.0
                val_dtype = np.int64 if max_val > float(np.iinfo(np.int32).max) else np.int32
                canonical_csr = ensure_csr_matrix(sp.csr_matrix(expr_mat.astype(val_dtype)))

            else:
                # GENES_BY_CELLS: rows are genes, cols are cells
                if mapping.geneIdColumn and mapping.geneIdColumn in table_df.columns:
                    raw_gene_ids = table_df[mapping.geneIdColumn].astype(str).tolist()
                    expr_df = table_df.drop(columns=[mapping.geneIdColumn])
                else:
                    # Column 0 is gene IDs
                    raw_gene_ids = table_df.iloc[:, 0].astype(str).tolist()
                    expr_df = table_df.iloc[:, 1:]

                raw_cell_ids = [str(c).strip() for c in expr_df.columns]
                expr_mat = expr_df.to_numpy(dtype=np.float64)
                validate_count_matrix(expr_mat)

                max_val = float(np.max(expr_mat)) if expr_mat.size > 0 else 0.0
                val_dtype = np.int64 if max_val > float(np.iinfo(np.int32).max) else np.int32
                source_csr = sp.csr_matrix(expr_mat.astype(val_dtype))
                # Explicit transpose from genes x cells -> cells x genes
                canonical_csr = ensure_csr_matrix(source_csr.T)

        # 2. Identifier uniqueness and sanity validation
        clean_cell_ids = validate_identifiers(raw_cell_ids, context="cell IDs")
        clean_gene_ids = validate_identifiers(raw_gene_ids, context="gene IDs")

        if len(clean_gene_ids) != len(set(clean_gene_ids)):
            raise CountValidationError(
                "Duplicate gene identifiers found in expression matrix",
                issue_code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
            )

        assert (
            canonical_csr.shape
            == (
                len(clean_cell_ids),
                len(clean_gene_ids),
            )
        ), f"Matrix shape {canonical_csr.shape} must equal ({len(clean_cell_ids)}, {len(clean_gene_ids)})"

        # 3. Species conflict check against confirmed organism
        if check_ensembl_species_conflict(clean_gene_ids):
            raise ImporterSecurityError(
                "Cannot canonicalize dataset containing conflicting Human and Mouse Ensembl IDs",
                issue_code=ScientificIssueCode.UNRESOLVED_SPECIES,
            )
        if has_opposing_ensembl_species(clean_gene_ids, organism):
            raise ImporterSecurityError(
                f"Cannot canonicalize dataset: specified organism {organism.value} conflicts with opposing Ensembl IDs",
                issue_code=ScientificIssueCode.UNRESOLVED_SPECIES,
            )

        # 4. Construct canonical obs and apply CELL_METADATA joins by value
        canonical_obs = pd.DataFrame(index=clean_cell_ids)
        cell_meta_files = source_set.get_files_by_role(ImportFileRole.CELL_METADATA)

        if cell_meta_files:
            cell_df, _ = read_delimited_table(
                cell_meta_files[0].local_path, limits=self.config.decompression_limits
            )
            join_col = (
                mapping.cellIdColumn
                if mapping.cellIdColumn and mapping.cellIdColumn in cell_df.columns
                else cell_df.columns[0]
            )

            # Check uniqueness of metadata join column
            if cell_df[join_col].duplicated().any():
                raise CountValidationError(
                    f"Duplicate cell identifiers in CELL_METADATA column '{join_col}'",
                    issue_code=ScientificIssueCode.DUPLICATE_CELL_IDS,
                )

            # Check completeness: all matrix cells must be in metadata table
            meta_id_set = set(cell_df[join_col].astype(str))
            missing_cells = [cid for cid in clean_cell_ids if cid not in meta_id_set]
            if missing_cells:
                raise ImporterSecurityError(
                    f"Missing {len(missing_cells)} cell identifiers in CELL_METADATA table",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )

            # Deterministic value join: realigns to exact clean_cell_ids order; ignores unreferenced rows
            realigned_cell_meta = cell_df.set_index(join_col).loc[clean_cell_ids]

            # Sample mapping (Strict CellWeave Authority — Fail-Closed)
            if mapping.sampleColumn:
                if mapping.sampleColumn not in realigned_cell_meta.columns:
                    raise ImporterSecurityError(
                        f"Requested sampleColumn '{mapping.sampleColumn}' not found in CELL_METADATA",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                if not mapping.sampleMappings:
                    raise ImporterSecurityError(
                        f"sampleColumn '{mapping.sampleColumn}' provided without approved sampleMappings; "
                        "source values cannot pass through automatically as sample keys",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                map_dict = {m.sourceValue: m.sampleKey for m in mapping.sampleMappings}
                source_sample_vals = realigned_cell_meta[mapping.sampleColumn].astype(str).tolist()
                mapped_sample_keys: list[str] = []
                for val in source_sample_vals:
                    target_key = map_dict.get(val)
                    if not target_key or not target_key.strip():
                        raise ImporterSecurityError(
                            f"Incomplete sample mapping: source value '{val}' in column '{mapping.sampleColumn}' "
                            "has no approved SampleMappingEntry",
                            issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                        )
                    mapped_sample_keys.append(target_key.strip())
                canonical_obs["sample_key"] = mapped_sample_keys
            elif mapping.sampleMappings:
                raise ImporterSecurityError(
                    "sampleMappings provided without a corresponding sampleColumn",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )

            # Condition mapping
            if mapping.conditionColumn:
                if mapping.conditionColumn not in realigned_cell_meta.columns:
                    raise ImporterSecurityError(
                        f"Requested conditionColumn '{mapping.conditionColumn}' not found in CELL_METADATA",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                canonical_obs["condition"] = (
                    realigned_cell_meta[mapping.conditionColumn].astype(str).tolist()
                )

            # Batch mapping
            if mapping.batchColumn:
                if mapping.batchColumn not in realigned_cell_meta.columns:
                    raise ImporterSecurityError(
                        f"Requested batchColumn '{mapping.batchColumn}' not found in CELL_METADATA",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                canonical_obs["batch"] = (
                    realigned_cell_meta[mapping.batchColumn].astype(str).tolist()
                )

            # Biological replicate
            if mapping.biologicalReplicateColumn:
                if mapping.biologicalReplicateColumn not in realigned_cell_meta.columns:
                    raise ImporterSecurityError(
                        f"Requested biologicalReplicateColumn '{mapping.biologicalReplicateColumn}' not found in CELL_METADATA",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                canonical_obs["biological_replicate"] = (
                    realigned_cell_meta[mapping.biologicalReplicateColumn].astype(str).tolist()
                )

            # Technical replicate
            if mapping.technicalReplicateColumn:
                if mapping.technicalReplicateColumn not in realigned_cell_meta.columns:
                    raise ImporterSecurityError(
                        f"Requested technicalReplicateColumn '{mapping.technicalReplicateColumn}' not found in CELL_METADATA",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                canonical_obs["technical_replicate"] = (
                    realigned_cell_meta[mapping.technicalReplicateColumn].astype(str).tolist()
                )
        else:
            # No CELL_METADATA file
            if (
                mapping.sampleColumn
                or mapping.sampleMappings
                or mapping.conditionColumn
                or mapping.batchColumn
                or mapping.biologicalReplicateColumn
                or mapping.technicalReplicateColumn
            ):
                raise ImporterSecurityError(
                    "Cell metadata mapping was requested, but no CELL_METADATA file was provided in source set",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )

        # 5. Construct canonical var and apply FEATURE_METADATA joins by value
        final_gene_ids = clean_gene_ids
        canonical_var = pd.DataFrame(index=clean_gene_ids)
        canonical_var["source_feature_id"] = clean_gene_ids

        feat_meta_files = source_set.get_files_by_role(ImportFileRole.FEATURE_METADATA)
        if feat_meta_files:
            feat_df, _ = read_delimited_table(
                feat_meta_files[0].local_path, limits=self.config.decompression_limits
            )

            # Find matching join column in feat_df (strict value match, no positional join)
            gene_set = set(clean_gene_ids)
            join_col = None
            for col in feat_df.columns:
                if gene_set.issubset(set(feat_df[col].astype(str))):
                    join_col = col
                    break

            if join_col is None:
                raise ImporterSecurityError(
                    "Cannot join FEATURE_METADATA to matrix: matrix features missing from metadata table",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )

            if feat_df[join_col].duplicated().any():
                raise CountValidationError(
                    f"Duplicate feature identifiers in FEATURE_METADATA column '{join_col}'",
                    issue_code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                )

            # Deterministic value join: realigns to exact clean_gene_ids order; ignores unreferenced rows
            realigned_feat_meta = feat_df.set_index(join_col).loc[clean_gene_ids]

            # If mapping.geneIdColumn points to a stable ID column distinct from join_col
            if (
                mapping.geneIdColumn
                and mapping.geneIdColumn in realigned_feat_meta.columns
                and mapping.geneIdColumn != join_col
            ):
                stable_ids = realigned_feat_meta[mapping.geneIdColumn].astype(str).tolist()
                clean_stable_ids = validate_identifiers(
                    stable_ids, context="FEATURE_METADATA stable gene IDs"
                )
                if len(clean_stable_ids) != len(set(clean_stable_ids)):
                    raise CountValidationError(
                        "Duplicate stable gene identifiers in FEATURE_METADATA",
                        issue_code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                    )
                final_gene_ids = clean_stable_ids
                canonical_var = pd.DataFrame(index=clean_stable_ids)
                canonical_var["source_feature_id"] = clean_gene_ids

            # If mapping.geneSymbolColumn provides descriptive symbols
            if mapping.geneSymbolColumn:
                if mapping.geneSymbolColumn not in realigned_feat_meta.columns:
                    raise ImporterSecurityError(
                        f"Requested geneSymbolColumn '{mapping.geneSymbolColumn}' not found in FEATURE_METADATA",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                canonical_var["gene_symbol"] = (
                    realigned_feat_meta[mapping.geneSymbolColumn].astype(str).tolist()
                )
        else:
            if mapping.geneSymbolColumn:
                raise ImporterSecurityError(
                    "geneSymbolColumn mapping was requested, but no FEATURE_METADATA file was provided in source set",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )

        # 6. Build and validate canonical AnnData
        ref_build = (
            mapping.referenceBuild if (mapping and mapping.referenceBuild) else reference_build
        )

        canonical_adata = build_canonical_anndata(
            x=canonical_csr,
            obs=canonical_obs,
            var=canonical_var,
            obs_names=clean_cell_ids,
            var_names=final_gene_ids,
            organism=organism,
            gene_id_scheme=gene_id_scheme,
            reference_build=ref_build,
        )

        validate_canonical_anndata(
            canonical_adata,
            expected_organism=organism,
            expected_gene_id_scheme=gene_id_scheme,
        )

        # 7. Serialize to disk and compute canonical checksum
        os.makedirs(dest_path.parent, exist_ok=True)
        canonical_adata.write_h5ad(dest_path)
        canonical_checksum = compute_file_sha256(dest_path)

        return CanonicalizationResult(
            sourceFormat=SourceFormat.GENERIC_MATRIX,
            canonicalFormat="H5AD",
            canonicalChecksum=canonical_checksum,
            organism=organism,
            geneIdScheme=gene_id_scheme,
            cellCount=int(canonical_adata.n_obs),
            geneCount=int(canonical_adata.n_vars),
            warnings=[],
            referenceBuild=ref_build,
        )


__all__ = [
    "GenericMatrixConfig",
    "GenericMatrixImporter",
    "read_delimited_table",
    "sniff_delimiter",
]
