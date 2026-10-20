"""Generic matrix importer (GENERIC_MATRIX) conforming to Phase 3 System Design and Policy B.

Supports:
- CSV expression matrices (plain and bounded gzip).
- TSV expression matrices (plain and bounded gzip).
- Matrix Market (.mtx / .mtx.gz) expression matrices.
- Optional CELL_METADATA (CSV / TSV) for CSV/TSV matrices; mandatory for Generic MTX.
- Optional FEATURE_METADATA (CSV / TSV) for CSV/TSV matrices; mandatory for Generic MTX.

Invariants:
1. Orientation is never guessed: requires explicit MetadataMappingSubmission.orientation
   (CELLS_BY_GENES vs GENES_BY_CELLS).
2. Explicit identity authority (Step G.1/G.2):
   - No column-0 identity fallback or named non-numeric first-column heuristic.
   - No Pandas RangeIndex fallback.
   - No metadata row-position assumption or subset-overlap column guessing.
   - No 10x biological identity semantics.
   - CSV/TSV CELLS_BY_GENES requires explicit matrixCellIdColumn.
   - CSV/TSV GENES_BY_CELLS requires explicit matrixFeatureIdColumn.
   - CELL_METADATA join requires explicit cellMetadataJoinColumn (for CSV/TSV) or
     cellAxisIndexColumn (for Generic MTX).
   - FEATURE_METADATA join requires explicit featureMetadataJoinColumn (for CSV/TSV) or
     featureAxisIndexColumn (for Generic MTX).
   - Generic MTX requires orientation, CELL_METADATA, FEATURE_METADATA,
     cellAxisIndexColumn, featureAxisIndexColumn, cellIdColumn, and geneIdColumn.
3. Raw counts only: validates integer, non-negative, finite counts. Rejects fractional,
   negative, NaN/Inf, and ambiguous near-integral values.
4. Strict sample authority: source values never pass through to sample_key; all mapped
   cells must resolve through explicit SampleMappingEntry.sampleKey. Incomplete mapping fails closed.
5. Slot hygiene: Policy B AnnData with SciPy CSR matrix in X, clean layers, empty obsm/varm/uns.
6. Content-based format and delimiter detection: file extensions (.csv/.tsv) are never used
   as authoritative format evidence.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    parse_matrix_market,
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

_STRICT_POSITIVE_INTEGER_RE = re.compile(r"^[1-9][0-9]*$")


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

    All columns are initially read as strings (na_filter=False) to preserve identifier and metadata fidelity.
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
            df = pd.read_csv(bio, sep=delim, header=0, dtype=str, index_col=False, na_filter=False)
        except Exception as exc:
            raise CountValidationError(
                f"Failed to parse delimited table from {path.name}: {exc}",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            ) from exc
    else:
        try:
            df = pd.read_csv(path, sep=delim, header=0, dtype=str, index_col=False, na_filter=False)
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


def _read_raw_header_tokens(file_path: Path, delim: str, limits: DecompressionLimits) -> list[str]:
    """Read raw header tokens from the first non-empty, non-comment line before Pandas mangling."""
    for raw_line in stream_text_lines(file_path, limits=limits):
        line = raw_line.strip("\r\n")
        if not line.strip() or line.lstrip().startswith("#") or line.lstrip().startswith("%"):
            continue
        return [tok.strip() for tok in line.split(delim)]
    return []


def _is_unnamed_placeholder_header(col_name: object) -> bool:
    """Return True if a column label is empty or a Pandas-generated 'Unnamed:' placeholder."""
    s = str(col_name).strip()
    return not s or s.startswith("Unnamed:")


def _validate_and_align_mtx_axis_metadata(
    meta_df: pd.DataFrame,
    axis_col: str,
    expected_n: int,
    axis_label: str,
) -> pd.DataFrame:
    """Validate explicit 1-based integer MTX axis coordinate column and align metadata rows to 1..expected_n.

    Enforces:
    - axis_col exists in meta_df
    - integer values only (1-based, no nulls/blanks, no zero, no negatives, no fractions, no non-numeric)
    - unique coordinates
    - complete coverage and exact range: set of coordinates must equal {1, 2, ..., expected_n}
      with len(meta_df) == expected_n (no gaps, duplicates, out-of-range values, or extra coordinates).
    """
    if not axis_col or not axis_col.strip():
        raise ImporterSecurityError(
            f"Generic MTX requires explicit {axis_label} axis index column mapping",
            issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
        )
    if axis_col not in meta_df.columns:
        raise ImporterSecurityError(
            f"Requested {axis_label} axis index column '{axis_col}' not found in metadata table",
            issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
        )

    raw_vals = meta_df[axis_col].tolist()
    parsed_indices: list[int] = []

    for idx_pos, raw_val in enumerate(raw_vals):
        if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
            raise CountValidationError(
                f"Blank or null coordinate in {axis_label} axis index column '{axis_col}' at row {idx_pos + 1}",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            )
        s = str(raw_val).strip()
        if not s:
            raise CountValidationError(
                f"Blank coordinate in {axis_label} axis index column '{axis_col}' at row {idx_pos + 1}",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            )
        if not _STRICT_POSITIVE_INTEGER_RE.match(s):
            raise CountValidationError(
                f"Invalid {axis_label} axis coordinate '{s}' in column '{axis_col}': "
                "must be a strict 1-based positive integer (no zero, negatives, fractions, or non-numeric strings)",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            )
        val = int(s)
        parsed_indices.append(val)

    if len(parsed_indices) != len(set(parsed_indices)):
        raise CountValidationError(
            f"Duplicate coordinates found in {axis_label} axis index column '{axis_col}'",
            issue_code=ScientificIssueCode.DIMENSION_MISMATCH,
        )

    expected_set = set(range(1, expected_n + 1))
    actual_set = set(parsed_indices)
    if len(parsed_indices) != expected_n or actual_set != expected_set:
        raise CountValidationError(
            f"{axis_label.capitalize()} axis coordinates in '{axis_col}' must form an exact 1..{expected_n} "
            f"permutation without gaps, out-of-range values, or extra coordinates (got {len(parsed_indices)} rows)",
            issue_code=ScientificIssueCode.DIMENSION_MISMATCH,
        )

    aligned = meta_df.copy()
    aligned["_ cw_axis_coord_"] = parsed_indices
    aligned = aligned.sort_values("_ cw_axis_coord_", kind="mergesort").drop(
        columns=["_ cw_axis_coord_"]
    )
    aligned.index = pd.Index(range(1, expected_n + 1))
    return aligned


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
        """Validate cardinality, structural parsing, table consistency, headers, and raw counts.

        Note: validate() never assumes column 0 of CSV/TSV or metadata tables is an authoritative
        identifier column without explicit mapping. It validates structural integrity, header hygiene,
        and numeric count semantics on numeric expression columns.
        """
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
                _n_rows, _n_cols, _n_entries, _coo = parse_matrix_market(
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
            # CSV / TSV structural & count validation
            try:
                delim = "\t" if matrix_kind == "TSV" else ","
                df, _ = read_delimited_table(
                    matrix_path,
                    limits=self.config.decompression_limits,
                    expected_delim=delim,
                )
                raw_header_tokens = _read_raw_header_tokens(
                    matrix_path, delim=delim, limits=self.config.decompression_limits
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

            if len(df.columns) < 2 or len(raw_header_tokens) < 2:
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.MALFORMED_INPUT,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Expression matrix table must have at least 2 columns (found {len(df.columns)})",
                        userActionRequired=True,
                    )
                ]

            # Validate ALL raw header tokens: reject blank or Pandas 'Unnamed:' placeholders
            if any(_is_unnamed_placeholder_header(tok) for tok in raw_header_tokens) or any(
                _is_unnamed_placeholder_header(col) for col in df.columns
            ):
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.MALFORMED_INPUT,
                        severity=ScientificIssueSeverity.ERROR,
                        message=(
                            f"Column header of {matrix_path.name} contains blank or 'Unnamed:' "
                            "placeholder identifiers"
                        ),
                        userActionRequired=True,
                    )
                ]

            # Check duplicate column identifiers in raw header (before pandas mangles them)
            if len(raw_header_tokens) != len(set(raw_header_tokens)):
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"Duplicate identifiers found in column header of {matrix_path.name}",
                        userActionRequired=True,
                    )
                ]

            # Without knowing which column is the explicit row-ID column (requires mapping),
            # at most 1 column may be non-numeric (the row-ID column), and at least 1 column must be numeric.
            numeric_cols: list[np.ndarray[Any, np.dtype[np.float64]]] = []
            non_numeric_col_names: list[str] = []
            for col_name in df.columns:
                col_series = df[col_name].astype(str).str.strip()
                if col_series.eq("").any():
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.MALFORMED_INPUT,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"Matrix column '{col_name}' in {matrix_path.name} contains empty cells",
                            userActionRequired=True,
                        )
                    ]
                try:
                    col_arr = col_series.to_numpy(dtype=np.float64)
                    numeric_cols.append(col_arr)
                except (ValueError, TypeError):
                    non_numeric_col_names.append(str(col_name))

            if len(non_numeric_col_names) > 1 or len(numeric_cols) == 0:
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.MALFORMED_INPUT,
                        severity=ScientificIssueSeverity.ERROR,
                        message=(
                            f"Expression matrix body in {matrix_path.name} contains non-numeric expression values"
                        ),
                        userActionRequired=True,
                    )
                ]

            arr = np.column_stack(numeric_cols)
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

        # 2. Validate CELL_METADATA structural parseability if present
        cell_meta_files = source_set.get_files_by_role(ImportFileRole.CELL_METADATA)
        if cell_meta_files:
            try:
                cell_meta_path = validate_trusted_local_path(cell_meta_files[0].local_path)
                cell_df, cell_delim = read_delimited_table(
                    cell_meta_path, limits=self.config.decompression_limits
                )
                cell_raw_headers = _read_raw_header_tokens(
                    cell_meta_path, delim=cell_delim, limits=self.config.decompression_limits
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
                if any(_is_unnamed_placeholder_header(h) for h in cell_raw_headers) or len(
                    cell_raw_headers
                ) != len(set(cell_raw_headers)):
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.MALFORMED_INPUT,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"CELL_METADATA table in {cell_meta_path.name} has blank or duplicate column headers",
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

        # 3. Validate FEATURE_METADATA structural parseability if present
        feat_meta_files = source_set.get_files_by_role(ImportFileRole.FEATURE_METADATA)
        if feat_meta_files:
            try:
                feat_meta_path = validate_trusted_local_path(feat_meta_files[0].local_path)
                feat_df, feat_delim = read_delimited_table(
                    feat_meta_path, limits=self.config.decompression_limits
                )
                feat_raw_headers = _read_raw_header_tokens(
                    feat_meta_path, delim=feat_delim, limits=self.config.decompression_limits
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
                if any(_is_unnamed_placeholder_header(h) for h in feat_raw_headers) or len(
                    feat_raw_headers
                ) != len(set(feat_raw_headers)):
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.MALFORMED_INPUT,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"FEATURE_METADATA table in {feat_meta_path.name} has blank or duplicate column headers",
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

    def inspect(
        self,
        source_set: ImportSourceSet,
        mapping: MetadataMappingSubmission | None = None,
    ) -> ImportInspectionResult:
        """Extract metadata columns, evaluate mapping readiness, and report issues."""
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

        # Inspect available metadata columns strictly from CELL_METADATA and FEATURE_METADATA tables.
        # Note: ImportInspectionResult exposes availableCellMetadataColumns and availableFeatureMetadataColumns
        # for supplied metadata files; it does not have a separate field for MATRIX-internal columns.
        cell_meta_cols: list[str] = []
        cell_df_cand: pd.DataFrame | None = None
        cell_meta_files = source_set.get_files_by_role(ImportFileRole.CELL_METADATA)
        if cell_meta_files:
            try:
                cell_df, _ = read_delimited_table(
                    cell_meta_files[0].local_path,
                    limits=self.config.decompression_limits,
                )
                cell_meta_cols = list(cell_df.columns)
                cell_df_cand = cell_df
            except Exception:  # noqa: BLE001
                pass

        feat_meta_cols: list[str] = []
        feat_df_cand: pd.DataFrame | None = None
        feat_meta_files = source_set.get_files_by_role(ImportFileRole.FEATURE_METADATA)
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

        inspection_issues: list[ScientificIssue] = list(issues)

        # Evaluate candidate Ensembl organism / scheme across feature candidates
        detected_org: Organism | None = None
        detected_scheme: GeneIdScheme | None = None
        cand_id_sources: list[list[str]] = []
        table_df_cand: pd.DataFrame | None = None
        if matrix_kind != "MTX":
            try:
                delim = "\t" if matrix_kind == "TSV" else ","
                table_df_cand, _ = read_delimited_table(
                    matrix_path,
                    limits=self.config.decompression_limits,
                    expected_delim=delim,
                )
                # Candidate header tokens (excluding first column if non-numeric, or all columns)
                cand_id_sources.append([str(c).strip() for c in table_df_cand.columns[1:]])
                cand_id_sources.append([str(x).strip() for x in table_df_cand.iloc[:, 0].tolist()])
            except Exception:  # noqa: BLE001
                pass

        if feat_df_cand is not None and len(feat_df_cand.columns) > 0:
            for col in feat_df_cand.columns:
                cand_id_sources.append([str(x).strip() for x in feat_df_cand[col].tolist()])

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
            if org is not None and detected_org is None:
                detected_org = org
                detected_scheme = GeneIdScheme.ENSEMBL

        # Check orientation & explicit identity mapping requirements
        if mapping is None or mapping.orientation is None:
            inspection_issues.append(
                ScientificIssue(
                    code=ScientificIssueCode.AMBIGUOUS_ORIENTATION,
                    severity=ScientificIssueSeverity.REQUIRES_USER_INPUT,
                    message=(
                        "GENERIC_MATRIX orientation (CELLS_BY_GENES vs GENES_BY_CELLS) and "
                        "explicit identity mappings require user confirmation"
                    ),
                    userActionRequired=True,
                )
            )

        if matrix_kind == "MTX":
            if cell_df_cand is None or feat_df_cand is None:
                inspection_issues.append(
                    ScientificIssue(
                        code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                        severity=ScientificIssueSeverity.REQUIRES_USER_INPUT,
                        message="Generic MTX requires both CELL_METADATA and FEATURE_METADATA tables",
                        userActionRequired=True,
                    )
                )
            if mapping is None or not (
                mapping.cellAxisIndexColumn
                and mapping.featureAxisIndexColumn
                and mapping.cellIdColumn
                and mapping.geneIdColumn
            ):
                inspection_issues.append(
                    ScientificIssue(
                        code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                        severity=ScientificIssueSeverity.REQUIRES_USER_INPUT,
                        message=(
                            "Generic MTX requires explicit cellAxisIndexColumn, featureAxisIndexColumn, "
                            "cellIdColumn, and geneIdColumn mappings"
                        ),
                        userActionRequired=True,
                    )
                )
            elif (
                mapping.orientation is not None
                and cell_df_cand is not None
                and feat_df_cand is not None
            ):
                # Validate supplied MTX axis coordinates and column existence; malformed/duplicate -> INVALID
                try:
                    n_rows, n_cols, _n_entries, _coo = parse_matrix_market(
                        matrix_path, limits=self.config.decompression_limits
                    )
                    exp_cells = (
                        n_rows
                        if mapping.orientation == MatrixOrientation.CELLS_BY_GENES
                        else n_cols
                    )
                    exp_feats = (
                        n_cols
                        if mapping.orientation == MatrixOrientation.CELLS_BY_GENES
                        else n_rows
                    )
                    _validate_and_align_mtx_axis_metadata(
                        cell_df_cand, mapping.cellAxisIndexColumn, exp_cells, "cell"
                    )
                    _validate_and_align_mtx_axis_metadata(
                        feat_df_cand, mapping.featureAxisIndexColumn, exp_feats, "feature"
                    )
                    if mapping.cellIdColumn not in cell_df_cand.columns:
                        raise ImporterSecurityError(
                            f"cellIdColumn '{mapping.cellIdColumn}' not found in CELL_METADATA",
                            issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                        )
                    if mapping.geneIdColumn not in feat_df_cand.columns:
                        raise ImporterSecurityError(
                            f"geneIdColumn '{mapping.geneIdColumn}' not found in FEATURE_METADATA",
                            issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                        )
                except (CountValidationError, ImporterSecurityError) as exc:
                    inspection_issues.append(
                        ScientificIssue(
                            code=exc.issue_code,
                            severity=ScientificIssueSeverity.ERROR,
                            message=str(exc),
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
                        detectedOrganism=detected_org,
                        detectedGeneIdScheme=detected_scheme,
                        referenceBuildCandidate=None,
                    )
        else:
            # CSV / TSV mapping checks when mapping is supplied or missing
            if (
                mapping is not None
                and mapping.orientation is not None
                and table_df_cand is not None
            ):
                if mapping.orientation == MatrixOrientation.CELLS_BY_GENES:
                    if not mapping.matrixCellIdColumn:
                        inspection_issues.append(
                            ScientificIssue(
                                code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                                severity=ScientificIssueSeverity.REQUIRES_USER_INPUT,
                                message="CSV/TSV CELLS_BY_GENES requires explicit matrixCellIdColumn",
                                userActionRequired=True,
                            )
                        )
                    elif mapping.matrixCellIdColumn not in table_df_cand.columns:
                        inspection_issues.append(
                            ScientificIssue(
                                code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                                severity=ScientificIssueSeverity.ERROR,
                                message=f"matrixCellIdColumn '{mapping.matrixCellIdColumn}' not found in MATRIX",
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
                            detectedOrganism=detected_org,
                            detectedGeneIdScheme=detected_scheme,
                            referenceBuildCandidate=None,
                        )
                else:
                    if not mapping.matrixFeatureIdColumn:
                        inspection_issues.append(
                            ScientificIssue(
                                code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                                severity=ScientificIssueSeverity.REQUIRES_USER_INPUT,
                                message="CSV/TSV GENES_BY_CELLS requires explicit matrixFeatureIdColumn",
                                userActionRequired=True,
                            )
                        )
                    elif mapping.matrixFeatureIdColumn not in table_df_cand.columns:
                        inspection_issues.append(
                            ScientificIssue(
                                code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                                severity=ScientificIssueSeverity.ERROR,
                                message=f"matrixFeatureIdColumn '{mapping.matrixFeatureIdColumn}' not found in MATRIX",
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
                            detectedOrganism=detected_org,
                            detectedGeneIdScheme=detected_scheme,
                            referenceBuildCandidate=None,
                        )

                if cell_df_cand is not None and not mapping.cellMetadataJoinColumn:
                    inspection_issues.append(
                        ScientificIssue(
                            code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                            severity=ScientificIssueSeverity.REQUIRES_USER_INPUT,
                            message="CELL_METADATA requires explicit cellMetadataJoinColumn",
                            userActionRequired=True,
                        )
                    )
                if feat_df_cand is not None and not mapping.featureMetadataJoinColumn:
                    inspection_issues.append(
                        ScientificIssue(
                            code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                            severity=ScientificIssueSeverity.REQUIRES_USER_INPUT,
                            message="FEATURE_METADATA requires explicit featureMetadataJoinColumn",
                            userActionRequired=True,
                        )
                    )

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

        Requires explicit MetadataMappingSubmission with orientation and explicit identity mappings.
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

        cell_meta_files = source_set.get_files_by_role(ImportFileRole.CELL_METADATA)
        feat_meta_files = source_set.get_files_by_role(ImportFileRole.FEATURE_METADATA)

        realigned_cell_meta: pd.DataFrame | None = None
        realigned_feat_meta: pd.DataFrame | None = None
        canonical_cell_ids: list[str]
        canonical_gene_ids: list[str]
        source_feature_ids: list[str]
        canonical_csr: sp.csr_matrix

        # =====================================================================
        # PATH A: GENERIC MATRIX MARKET (.mtx / .mtx.gz)
        # =====================================================================
        if matrix_kind == "MTX":
            n_rows, n_cols, _n_entries, coo_matrix = parse_matrix_market(
                matrix_path, limits=self.config.decompression_limits
            )

            # Section 9: Generic MTX requires orientation, CELL_METADATA, FEATURE_METADATA,
            # cellAxisIndexColumn, featureAxisIndexColumn, cellIdColumn, and geneIdColumn. No exception.
            if not cell_meta_files:
                raise ImporterSecurityError(
                    "Generic Matrix Market requires CELL_METADATA",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if not feat_meta_files:
                raise ImporterSecurityError(
                    "Generic Matrix Market requires FEATURE_METADATA",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if not mapping.cellAxisIndexColumn or not mapping.cellAxisIndexColumn.strip():
                raise ImporterSecurityError(
                    "Generic Matrix Market requires explicit cellAxisIndexColumn mapping",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if not mapping.featureAxisIndexColumn or not mapping.featureAxisIndexColumn.strip():
                raise ImporterSecurityError(
                    "Generic Matrix Market requires explicit featureAxisIndexColumn mapping",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if not mapping.cellIdColumn or not mapping.cellIdColumn.strip():
                raise ImporterSecurityError(
                    "Generic Matrix Market requires explicit cellIdColumn mapping",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if not mapping.geneIdColumn or not mapping.geneIdColumn.strip():
                raise ImporterSecurityError(
                    "Generic Matrix Market requires explicit geneIdColumn mapping",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )

            # Reject using the numeric axis index column itself as biological cell or feature ID
            if mapping.cellIdColumn == mapping.cellAxisIndexColumn:
                raise ImporterSecurityError(
                    f"cellIdColumn ('{mapping.cellIdColumn}') cannot be the numeric cellAxisIndexColumn",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if mapping.geneIdColumn == mapping.featureAxisIndexColumn:
                raise ImporterSecurityError(
                    f"geneIdColumn ('{mapping.geneIdColumn}') cannot be the numeric featureAxisIndexColumn",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if (
                mapping.featureMetadataJoinColumn
                and mapping.featureMetadataJoinColumn == mapping.featureAxisIndexColumn
            ):
                raise ImporterSecurityError(
                    f"featureMetadataJoinColumn ('{mapping.featureMetadataJoinColumn}') cannot be the numeric featureAxisIndexColumn",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if (
                mapping.cellMetadataJoinColumn
                and mapping.cellMetadataJoinColumn == mapping.cellAxisIndexColumn
            ):
                raise ImporterSecurityError(
                    f"cellMetadataJoinColumn ('{mapping.cellMetadataJoinColumn}') cannot be the numeric cellAxisIndexColumn",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )

            cell_df, _ = read_delimited_table(
                cell_meta_files[0].local_path, limits=self.config.decompression_limits
            )
            feat_df, _ = read_delimited_table(
                feat_meta_files[0].local_path, limits=self.config.decompression_limits
            )

            # Section 11: MTX Orientation Mapping
            if orientation == MatrixOrientation.CELLS_BY_GENES:
                n_cells = n_rows
                n_features = n_cols
                canonical_csr = ensure_csr_matrix(coo_matrix.tocsr())
            else:
                n_features = n_rows
                n_cells = n_cols
                canonical_csr = ensure_csr_matrix(coo_matrix.T.tocsr())

            # Section 10: Validate 1..n_cells and 1..n_features axis coordinates and align strictly by coordinate
            realigned_cell_meta = _validate_and_align_mtx_axis_metadata(
                cell_df,
                mapping.cellAxisIndexColumn,
                expected_n=n_cells,
                axis_label="cell",
            )
            realigned_feat_meta = _validate_and_align_mtx_axis_metadata(
                feat_df,
                mapping.featureAxisIndexColumn,
                expected_n=n_features,
                axis_label="feature",
            )

            # Section 12 & 15: MTX Cell Identity & optional cellMetadataJoinColumn
            if mapping.cellIdColumn not in realigned_cell_meta.columns:
                raise ImporterSecurityError(
                    f"Requested cellIdColumn '{mapping.cellIdColumn}' not found in CELL_METADATA",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if mapping.cellMetadataJoinColumn:
                if mapping.cellMetadataJoinColumn not in realigned_cell_meta.columns:
                    raise ImporterSecurityError(
                        f"Requested cellMetadataJoinColumn '{mapping.cellMetadataJoinColumn}' not found in CELL_METADATA",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                _ = validate_identifiers(
                    realigned_cell_meta[mapping.cellMetadataJoinColumn].astype(str).tolist(),
                    context="MTX source cell tokens (cellMetadataJoinColumn)",
                )

            raw_cell_ids = realigned_cell_meta[mapping.cellIdColumn].astype(str).tolist()
            canonical_cell_ids = validate_identifiers(raw_cell_ids, context="cell IDs")

            # Section 13 & 14: MTX Feature Identity & optional featureMetadataJoinColumn
            if mapping.geneIdColumn not in realigned_feat_meta.columns:
                raise ImporterSecurityError(
                    f"Requested geneIdColumn '{mapping.geneIdColumn}' not found in FEATURE_METADATA",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            raw_gene_ids = realigned_feat_meta[mapping.geneIdColumn].astype(str).tolist()
            canonical_gene_ids = validate_identifiers(raw_gene_ids, context="gene IDs")
            if len(canonical_gene_ids) != len(set(canonical_gene_ids)):
                raise CountValidationError(
                    "Duplicate canonical gene identifiers in FEATURE_METADATA",
                    issue_code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                )

            if mapping.featureMetadataJoinColumn:
                if mapping.featureMetadataJoinColumn not in realigned_feat_meta.columns:
                    raise ImporterSecurityError(
                        f"Requested featureMetadataJoinColumn '{mapping.featureMetadataJoinColumn}' not found in FEATURE_METADATA",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                raw_source_feat_ids = (
                    realigned_feat_meta[mapping.featureMetadataJoinColumn].astype(str).tolist()
                )
                source_feature_ids = validate_identifiers(
                    raw_source_feat_ids, context="source feature IDs"
                )
                if len(source_feature_ids) != len(set(source_feature_ids)):
                    raise CountValidationError(
                        f"Duplicate source feature identifiers in FEATURE_METADATA column '{mapping.featureMetadataJoinColumn}'",
                        issue_code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                    )
            else:
                source_feature_ids = list(canonical_gene_ids)

        # =====================================================================
        # PATH B: DELIMITED MATRIX (CSV / TSV)
        # =====================================================================
        else:
            delim = "\t" if matrix_kind == "TSV" else ","
            table_df, _ = read_delimited_table(
                matrix_path,
                limits=self.config.decompression_limits,
                expected_delim=delim,
            )

            if orientation == MatrixOrientation.CELLS_BY_GENES:
                # Section 4: CSV/TSV CELLS_BY_GENES requires explicit mapping.matrixCellIdColumn
                if not mapping.matrixCellIdColumn or not mapping.matrixCellIdColumn.strip():
                    raise ImporterSecurityError(
                        "CSV/TSV CELLS_BY_GENES canonicalization requires explicit mapping.matrixCellIdColumn",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                matrix_cell_col = mapping.matrixCellIdColumn.strip()
                if matrix_cell_col not in table_df.columns:
                    raise ImporterSecurityError(
                        f"Requested matrixCellIdColumn '{matrix_cell_col}' not found in MATRIX columns",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )

                raw_source_cell_ids = table_df[matrix_cell_col].astype(str).tolist()
                expr_df = table_df.drop(columns=[matrix_cell_col])
                if expr_df.empty or len(expr_df.columns) == 0:
                    raise CountValidationError(
                        f"No expression columns remain in {matrix_path.name} after excluding matrixCellIdColumn '{matrix_cell_col}'",
                        issue_code=ScientificIssueCode.MALFORMED_INPUT,
                    )
                raw_source_feature_ids = [str(c).strip() for c in expr_df.columns]

                try:
                    expr_mat = expr_df.to_numpy(dtype=np.float64)
                except (ValueError, TypeError) as exc:
                    raise CountValidationError(
                        f"Expression matrix body in {matrix_path.name} contains non-numeric values",
                        issue_code=ScientificIssueCode.MALFORMED_INPUT,
                    ) from exc

                validate_count_matrix(expr_mat)
                max_val = float(np.max(expr_mat)) if expr_mat.size > 0 else 0.0
                val_dtype = np.int64 if max_val > float(np.iinfo(np.int32).max) else np.int32
                canonical_csr = ensure_csr_matrix(sp.csr_matrix(expr_mat.astype(val_dtype)))

            else:
                # Section 5: CSV/TSV GENES_BY_CELLS requires explicit mapping.matrixFeatureIdColumn
                if not mapping.matrixFeatureIdColumn or not mapping.matrixFeatureIdColumn.strip():
                    raise ImporterSecurityError(
                        "CSV/TSV GENES_BY_CELLS canonicalization requires explicit mapping.matrixFeatureIdColumn",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                matrix_feat_col = mapping.matrixFeatureIdColumn.strip()
                if matrix_feat_col not in table_df.columns:
                    raise ImporterSecurityError(
                        f"Requested matrixFeatureIdColumn '{matrix_feat_col}' not found in MATRIX columns",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )

                raw_source_feature_ids = table_df[matrix_feat_col].astype(str).tolist()
                expr_df = table_df.drop(columns=[matrix_feat_col])
                if expr_df.empty or len(expr_df.columns) == 0:
                    raise CountValidationError(
                        f"No expression columns remain in {matrix_path.name} after excluding matrixFeatureIdColumn '{matrix_feat_col}'",
                        issue_code=ScientificIssueCode.MALFORMED_INPUT,
                    )
                raw_source_cell_ids = [str(c).strip() for c in expr_df.columns]

                try:
                    expr_mat = expr_df.to_numpy(dtype=np.float64)
                except (ValueError, TypeError) as exc:
                    raise CountValidationError(
                        f"Expression matrix body in {matrix_path.name} contains non-numeric values",
                        issue_code=ScientificIssueCode.MALFORMED_INPUT,
                    ) from exc

                validate_count_matrix(expr_mat)
                max_val = float(np.max(expr_mat)) if expr_mat.size > 0 else 0.0
                val_dtype = np.int64 if max_val > float(np.iinfo(np.int32).max) else np.int32
                source_csr = sp.csr_matrix(expr_mat.astype(val_dtype))
                canonical_csr = ensure_csr_matrix(source_csr.T)

            # Validate source matrix cell and feature tokens
            source_cell_ids = validate_identifiers(raw_source_cell_ids, context="source cell IDs")
            source_feature_ids = validate_identifiers(
                raw_source_feature_ids, context="source feature IDs"
            )
            if len(source_feature_ids) != len(set(source_feature_ids)):
                raise CountValidationError(
                    "Duplicate source feature identifiers found in expression matrix",
                    issue_code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                )

            # Section 7: CSV/TSV CELL_METADATA join
            if cell_meta_files:
                cell_df, _ = read_delimited_table(
                    cell_meta_files[0].local_path, limits=self.config.decompression_limits
                )
                if not mapping.cellMetadataJoinColumn or not mapping.cellMetadataJoinColumn.strip():
                    raise ImporterSecurityError(
                        "CELL_METADATA requires explicit mapping.cellMetadataJoinColumn to join against matrix cell tokens",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                cell_join_col = mapping.cellMetadataJoinColumn.strip()
                if cell_join_col not in cell_df.columns:
                    raise ImporterSecurityError(
                        f"Requested cellMetadataJoinColumn '{cell_join_col}' not found in CELL_METADATA",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )

                join_vals = cell_df[cell_join_col].astype(str).str.strip()
                if join_vals.eq("").any():
                    raise CountValidationError(
                        f"Blank cell join identifier in CELL_METADATA column '{cell_join_col}'",
                        issue_code=ScientificIssueCode.MALFORMED_INPUT,
                    )
                if join_vals.duplicated().any():
                    raise CountValidationError(
                        f"Duplicate cell identifiers in CELL_METADATA join column '{cell_join_col}'",
                        issue_code=ScientificIssueCode.DUPLICATE_CELL_IDS,
                    )

                meta_cell_set = set(join_vals)
                missing_cells = [cid for cid in source_cell_ids if cid not in meta_cell_set]
                if missing_cells:
                    raise ImporterSecurityError(
                        f"Missing {len(missing_cells)} matrix cell identifiers in CELL_METADATA column '{cell_join_col}'",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )

                # Align strictly by value; extra unreferenced metadata rows are ignored
                cell_df_indexed = cell_df.copy()
                cell_df_indexed[cell_join_col] = join_vals
                realigned_cell_meta = cell_df_indexed.set_index(cell_join_col, drop=False).loc[
                    source_cell_ids
                ]

                if mapping.cellIdColumn:
                    if mapping.cellIdColumn not in realigned_cell_meta.columns:
                        raise ImporterSecurityError(
                            f"Requested cellIdColumn '{mapping.cellIdColumn}' not found in CELL_METADATA",
                            issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                        )
                    translated_cell_ids = (
                        realigned_cell_meta[mapping.cellIdColumn].astype(str).tolist()
                    )
                    canonical_cell_ids = validate_identifiers(
                        translated_cell_ids, context="canonical cell IDs"
                    )
                else:
                    canonical_cell_ids = list(source_cell_ids)
            else:
                if mapping.cellIdColumn or mapping.cellMetadataJoinColumn:
                    raise ImporterSecurityError(
                        "Cell metadata column mapping was requested, but no CELL_METADATA file was provided",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                canonical_cell_ids = list(source_cell_ids)

            # Section 6 & 8: CSV/TSV FEATURE_METADATA join
            if feat_meta_files:
                feat_df, _ = read_delimited_table(
                    feat_meta_files[0].local_path, limits=self.config.decompression_limits
                )
                if (
                    not mapping.featureMetadataJoinColumn
                    or not mapping.featureMetadataJoinColumn.strip()
                ):
                    raise ImporterSecurityError(
                        "FEATURE_METADATA requires explicit mapping.featureMetadataJoinColumn to join against matrix feature tokens",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                feat_join_col = mapping.featureMetadataJoinColumn.strip()
                if feat_join_col not in feat_df.columns:
                    raise ImporterSecurityError(
                        f"Requested featureMetadataJoinColumn '{feat_join_col}' not found in FEATURE_METADATA",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )

                feat_join_vals = feat_df[feat_join_col].astype(str).str.strip()
                if feat_join_vals.eq("").any():
                    raise CountValidationError(
                        f"Blank feature join identifier in FEATURE_METADATA column '{feat_join_col}'",
                        issue_code=ScientificIssueCode.MALFORMED_INPUT,
                    )
                if feat_join_vals.duplicated().any():
                    raise CountValidationError(
                        f"Duplicate feature identifiers in FEATURE_METADATA join column '{feat_join_col}'",
                        issue_code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                    )

                meta_feat_set = set(feat_join_vals)
                missing_feats = [fid for fid in source_feature_ids if fid not in meta_feat_set]
                if missing_feats:
                    raise ImporterSecurityError(
                        f"Missing {len(missing_feats)} matrix feature identifiers in FEATURE_METADATA column '{feat_join_col}'",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )

                feat_df_indexed = feat_df.copy()
                feat_df_indexed[feat_join_col] = feat_join_vals
                realigned_feat_meta = feat_df_indexed.set_index(feat_join_col, drop=False).loc[
                    source_feature_ids
                ]

                if mapping.geneIdColumn:
                    if mapping.geneIdColumn not in realigned_feat_meta.columns:
                        raise ImporterSecurityError(
                            f"Requested geneIdColumn '{mapping.geneIdColumn}' not found in FEATURE_METADATA",
                            issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                        )
                    stable_ids = realigned_feat_meta[mapping.geneIdColumn].astype(str).tolist()
                    canonical_gene_ids = validate_identifiers(
                        stable_ids, context="FEATURE_METADATA stable gene IDs"
                    )
                    if len(canonical_gene_ids) != len(set(canonical_gene_ids)):
                        raise CountValidationError(
                            "Duplicate stable gene identifiers in FEATURE_METADATA",
                            issue_code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                        )
                else:
                    canonical_gene_ids = list(source_feature_ids)
            else:
                if (
                    mapping.geneIdColumn
                    or mapping.geneSymbolColumn
                    or mapping.featureMetadataJoinColumn
                ):
                    raise ImporterSecurityError(
                        "Feature metadata column mapping was requested, but no FEATURE_METADATA file was provided",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                canonical_gene_ids = list(source_feature_ids)

        assert canonical_csr.shape == (
            len(canonical_cell_ids),
            len(canonical_gene_ids),
        ), (
            f"Matrix shape {canonical_csr.shape} must equal "
            f"({len(canonical_cell_ids)}, {len(canonical_gene_ids)})"
        )

        # =====================================================================
        # CONSTRUCT CANONICAL VAR (Section 25)
        # =====================================================================
        canonical_var = pd.DataFrame(index=canonical_gene_ids)
        canonical_var["source_feature_id"] = source_feature_ids

        if mapping.geneSymbolColumn:
            if realigned_feat_meta is None:
                raise ImporterSecurityError(
                    "geneSymbolColumn mapping was requested, but no FEATURE_METADATA file was provided in source set",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if mapping.geneSymbolColumn not in realigned_feat_meta.columns:
                raise ImporterSecurityError(
                    f"Requested geneSymbolColumn '{mapping.geneSymbolColumn}' not found in FEATURE_METADATA",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            symbol_vals = (
                realigned_feat_meta[mapping.geneSymbolColumn].astype(str).str.strip().tolist()
            )
            if any(not s for s in symbol_vals):
                raise CountValidationError(
                    f"Blank gene symbol in FEATURE_METADATA column '{mapping.geneSymbolColumn}'",
                    issue_code=ScientificIssueCode.MALFORMED_INPUT,
                )
            canonical_var["gene_symbol"] = symbol_vals

        # =====================================================================
        # ORGANISM & GENE SCHEME VALIDATION (Section 23)
        # =====================================================================
        for id_list_to_check in (source_feature_ids, canonical_gene_ids):
            if check_ensembl_species_conflict(id_list_to_check):
                raise ImporterSecurityError(
                    "Cannot canonicalize dataset containing conflicting Human and Mouse Ensembl IDs",
                    issue_code=ScientificIssueCode.UNRESOLVED_SPECIES,
                )
            if has_opposing_ensembl_species(id_list_to_check, organism):
                raise ImporterSecurityError(
                    f"Cannot canonicalize dataset: specified organism {organism.value} conflicts with opposing Ensembl IDs",
                    issue_code=ScientificIssueCode.UNRESOLVED_SPECIES,
                )

        if gene_id_scheme == GeneIdScheme.ENSEMBL:
            detected_org = detect_ensembl_organism_candidates(canonical_gene_ids)
            if detected_org is None and organism in (Organism.HUMAN, Organism.MOUSE):
                raise ImporterSecurityError(
                    f"gene_id_scheme=ENSEMBL requires canonical var_names to be valid Ensembl IDs for {organism.value}",
                    issue_code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                )

        # =====================================================================
        # CONSTRUCT CANONICAL OBS & SAMPLE / BIOLOGICAL MAPPINGS (Sections 22 & 24)
        # =====================================================================
        canonical_obs = pd.DataFrame(index=canonical_cell_ids)

        if realigned_cell_meta is not None:
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

        # =====================================================================
        # BUILD & VALIDATE CANONICAL ANNDATA (Policy B)
        # =====================================================================
        ref_build = (
            mapping.referenceBuild if (mapping and mapping.referenceBuild) else reference_build
        )

        canonical_adata = build_canonical_anndata(
            x=canonical_csr,
            obs=canonical_obs,
            var=canonical_var,
            obs_names=canonical_cell_ids,
            var_names=canonical_gene_ids,
            organism=organism,
            gene_id_scheme=gene_id_scheme,
            reference_build=ref_build,
        )

        validate_canonical_anndata(
            canonical_adata,
            expected_organism=organism,
            expected_gene_id_scheme=gene_id_scheme,
        )

        # Serialize to disk and compute canonical checksum
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
