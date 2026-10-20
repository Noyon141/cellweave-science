"""Shared parsing utilities, data structures, and configuration for 10x Genomics importers.

Supports:
- Fail-closed bounded decompression via Step D primitives.
- Streaming text line decoding for plain text and gzip.
- 10x barcode parsing and uniqueness validation.
- Modern 10x features.tsv (3-column) and legacy genes.tsv (2-column) parsing.
- Strict Gene Expression modality validation (V1 Policy).
- Streaming Matrix Market coordinate parser with sparse preservation.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from cellweave_science.contracts import (
    ScientificIssue,
    ScientificIssueCode,
    ScientificIssueSeverity,
)
from cellweave_science.importers.errors import CountValidationError
from cellweave_science.importers.security import (
    CompressionFormat,
    DecompressionLimits,
    inspect_physical_file,
    open_bounded_gzip_stream,
    validate_trusted_local_path,
)

GENE_EXPRESSION_MODALITY = "Gene Expression"


@dataclass(frozen=True)
class TenxImporterConfig:
    """Configuration for 10x importer technical safety and resource limits."""

    decompression_limits: DecompressionLimits = field(
        default_factory=lambda: DecompressionLimits(
            max_uncompressed_bytes=10 * 1024 * 1024 * 1024,  # 10 GiB technical ceiling
            max_compression_ratio=200.0,
        )
    )


@dataclass(frozen=True)
class TenxFeature:
    """A single feature entry parsed from a 10x feature table."""

    feature_id: str
    gene_symbol: str
    feature_type: str = GENE_EXPRESSION_MODALITY


def stream_text_lines(
    file_path: Path | str,
    limits: DecompressionLimits,
) -> Generator[str, None, None]:
    """Yield stripped text lines from plain text or bounded gzip file.

    Guards against decompression bombs using Step D open_bounded_gzip_stream.
    """
    path = validate_trusted_local_path(file_path)
    inspection = inspect_physical_file(path)

    if inspection.compression == CompressionFormat.GZIP:
        buffer = ""
        for chunk in open_bounded_gzip_stream(path, limits=limits):
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                yield line.rstrip("\r")
        if buffer:
            yield buffer.rstrip("\r")
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\r\n")


def parse_10x_barcodes(
    file_path: Path | str,
    limits: DecompressionLimits,
) -> list[str]:
    """Parse 10x barcodes file (barcodes.tsv or barcodes.tsv.gz).

    Invariants:
    1. Exactly one barcode per row.
    2. Non-empty, non-whitespace string.
    3. Barcodes must be unique (no duplicate barcodes allowed).
    4. Exact source string is preserved without stripping suffixes (e.g. -1).

    Raises:
        CountValidationError: If barcodes file is empty, contains blank lines, or has duplicates.
    """
    barcodes: list[str] = []
    seen: set[str] = set()

    for line_idx, raw_line in enumerate(stream_text_lines(file_path, limits=limits), start=1):
        line = raw_line.strip()
        if not line:
            raise CountValidationError(
                f"Empty or blank barcode encountered at line {line_idx}",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            )
        if line in seen:
            raise CountValidationError(
                f"Duplicate barcode detected: '{line}' at line {line_idx}",
                issue_code=ScientificIssueCode.DUPLICATE_CELL_IDS,
            )
        seen.add(line)
        barcodes.append(line)

    if not barcodes:
        raise CountValidationError(
            "Barcodes file is empty",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        )

    return barcodes


def parse_10x_features_tsv(
    file_path: Path | str,
    limits: DecompressionLimits,
) -> list[TenxFeature]:
    """Parse modern 10x features.tsv(.gz) or legacy genes.tsv(.gz).

    Supported formats:
    - Modern 3+ column: feature_id, feature_name, feature_type, [optional metadata]
    - Legacy 2 column: gene_id, gene_name (inferred as Gene Expression)

    Raises:
        CountValidationError: If features file is empty, malformed, or has empty identifiers.
    """
    features: list[TenxFeature] = []

    for line_idx, raw_line in enumerate(stream_text_lines(file_path, limits=limits), start=1):
        line = raw_line.strip()
        if not line:
            raise CountValidationError(
                f"Empty or blank line in features file at line {line_idx}",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            )

        parts = raw_line.split("\t")
        if len(parts) < 2:
            raise CountValidationError(
                f"Features line has fewer than 2 columns at line {line_idx}: '{raw_line}'",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            )

        feature_id = parts[0].strip()
        gene_symbol = parts[1].strip()
        if not feature_id:
            raise CountValidationError(
                f"Empty stable feature identifier at line {line_idx}",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            )

        feature_type = parts[2].strip() if len(parts) >= 3 else GENE_EXPRESSION_MODALITY

        features.append(
            TenxFeature(
                feature_id=feature_id,
                gene_symbol=gene_symbol if gene_symbol else feature_id,
                feature_type=feature_type,
            )
        )

    if not features:
        raise CountValidationError(
            "Features file is empty",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        )

    # Check for duplicate stable feature IDs
    seen_ids: set[str] = set()
    for feat in features:
        if feat.feature_id in seen_ids:
            raise CountValidationError(
                f"Duplicate stable feature identifier in features file: '{feat.feature_id}'",
                issue_code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
            )
        seen_ids.add(feat.feature_id)

    return features


def validate_10x_features(features: list[TenxFeature]) -> list[ScientificIssue]:
    """Validate parsed 10x feature list for CellWeave V1 scientific invariants.

    Invariants:
    1. V1 accepts Gene Expression only.
    2. Mixed modality or pure non-gene modality rejected as UNSUPPORTED_MODALITY.
    """
    issues: list[ScientificIssue] = []

    if not features:
        issues.append(
            ScientificIssue(
                code=ScientificIssueCode.MALFORMED_INPUT,
                severity=ScientificIssueSeverity.ERROR,
                message="Feature table is empty",
                userActionRequired=True,
            )
        )
        return issues

    unsupported_modalities = {
        feat.feature_type for feat in features if feat.feature_type != GENE_EXPRESSION_MODALITY
    }
    if unsupported_modalities:
        has_gene_expr = any(feat.feature_type == GENE_EXPRESSION_MODALITY for feat in features)
        if has_gene_expr:
            msg = (
                f"Mixed modality detected in 10x input: Gene Expression combined with "
                f"{sorted(unsupported_modalities)}. CellWeave V1 supports pure Gene Expression only. "
                "Silent feature subsetting is prohibited."
            )
        else:
            msg = (
                f"Unsupported modality in 10x input: {sorted(unsupported_modalities)}. "
                "CellWeave V1 supports Gene Expression only."
            )
        issues.append(
            ScientificIssue(
                code=ScientificIssueCode.UNSUPPORTED_MODALITY,
                severity=ScientificIssueSeverity.ERROR,
                message=msg,
                userActionRequired=True,
            )
        )

    return issues


def parse_10x_matrix_market(
    file_path: Path | str,
    limits: DecompressionLimits,
) -> tuple[int, int, int, sp.coo_matrix]:
    """Parse a Matrix Market coordinate file into a scipy.sparse.coo_matrix.

    Invariants:
    1. Header must specify 'matrix coordinate real general' or 'integer general'.
    2. Comments (%) are safely skipped.
    3. Dimension line specifies: n_rows n_cols n_entries.
    4. Source coordinates are 1-based (features x barcodes).
    5. Coordinates must be within bounds: 1 <= row <= n_rows, 1 <= col <= n_cols.
    6. All counts must be non-negative integers.
    7. Floating, fractional, negative, or ambiguous near-integral values fail closed.
    8. Truncated files (fewer lines than declared n_entries) fail closed.

    Returns:
        tuple of (n_rows, n_cols, n_entries, coo_matrix)

    Raises:
        CountValidationError: If the Matrix Market file violates any structural or numerical invariant.
    """
    line_iter = stream_text_lines(file_path, limits=limits)

    # 1. Header parsing
    try:
        header_line = next(line_iter).strip()
    except StopIteration:
        raise CountValidationError(
            "Matrix Market file is empty",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        ) from None

    if not header_line.startswith("%%MatrixMarket"):
        raise CountValidationError(
            f"Invalid Matrix Market header: '{header_line}'",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        )

    header_parts = header_line.lower().split()
    if len(header_parts) < 5 or header_parts[1] != "matrix" or header_parts[2] != "coordinate":
        raise CountValidationError(
            f"Unsupported Matrix Market format: '{header_line}'. Expected 'matrix coordinate ...'",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        )

    # 2. Skip comments and read dimension line
    dim_line = ""
    for line in line_iter:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        dim_line = s
        break

    if not dim_line:
        raise CountValidationError(
            "Matrix Market file missing dimension header",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        )

    dim_parts = dim_line.split()
    if len(dim_parts) != 3:
        raise CountValidationError(
            f"Malformed Matrix Market dimension line: '{dim_line}'. Expected 3 integers (rows, cols, entries)",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        )

    try:
        n_rows = int(dim_parts[0])
        n_cols = int(dim_parts[1])
        n_entries = int(dim_parts[2])
    except ValueError:
        raise CountValidationError(
            f"Non-integer values in Matrix Market dimension line: '{dim_line}'",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        ) from None

    if n_rows <= 0 or n_cols <= 0:
        raise CountValidationError(
            f"Matrix Market dimensions must be positive, found ({n_rows}, {n_cols})",
            issue_code=ScientificIssueCode.DIMENSION_MISMATCH,
        )

    if n_entries < 0:
        raise CountValidationError(
            f"Matrix Market entries count cannot be negative: {n_entries}",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        )

    # 3. Read data coordinate lines
    row_indices = np.empty(n_entries, dtype=np.int32)
    col_indices = np.empty(n_entries, dtype=np.int32)
    data_values = np.empty(n_entries, dtype=np.float64)

    parsed_count = 0
    for line in line_iter:
        s = line.strip()
        if not s or s.startswith("%"):
            continue

        if parsed_count >= n_entries:
            raise CountValidationError(
                f"Matrix Market file contains more data lines than declared entries ({n_entries})",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            )

        tokens = s.split()
        if len(tokens) != 3:
            raise CountValidationError(
                f"Malformed coordinate line {parsed_count + 1}: expected 3 tokens, found {len(tokens)} ('{s}')",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            )

        try:
            r = int(tokens[0])
            c = int(tokens[1])
            val_float = float(tokens[2])
        except ValueError:
            raise CountValidationError(
                f"Non-numeric values in coordinate line {parsed_count + 1}: '{s}'",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            ) from None

        # 1-based bounds check
        if r < 1 or r > n_rows or c < 1 or c > n_cols:
            raise CountValidationError(
                f"Coordinate out of bounds at line {parsed_count + 1}: ({r}, {c}) for shape ({n_rows}, {n_cols})",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            )

        # Value safety validation
        if not np.isfinite(val_float) or val_float < 0:
            raise CountValidationError(
                f"Matrix Market contains invalid non-count value at line {parsed_count + 1}: {val_float}",
                issue_code=ScientificIssueCode.NON_COUNT_INPUT,
            )

        diff = abs(val_float - round(val_float))
        if diff != 0.0:
            if diff <= 1e-12:
                raise CountValidationError(
                    f"Near-integral ambiguous value encountered at line {parsed_count + 1}: {val_float}",
                    issue_code=ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS,
                )
            raise CountValidationError(
                f"Fractional/continuous value encountered in raw count matrix at line {parsed_count + 1}: {val_float}",
                issue_code=ScientificIssueCode.NON_COUNT_INPUT,
            )

        row_indices[parsed_count] = r - 1  # Convert to 0-based index
        col_indices[parsed_count] = c - 1  # Convert to 0-based index
        data_values[parsed_count] = val_float
        parsed_count += 1

    if parsed_count != n_entries:
        raise CountValidationError(
            f"Truncated Matrix Market file: parsed {parsed_count} entries, expected {n_entries}",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        )

    # Determine optimal integer dtype
    max_val = float(np.max(data_values)) if n_entries > 0 else 0.0
    val_dtype = np.int64 if max_val > float(np.iinfo(np.int32).max) else np.int32
    clean_data = data_values.astype(val_dtype)

    coo = sp.coo_matrix(
        (clean_data, (row_indices, col_indices)),
        shape=(n_rows, n_cols),
        dtype=val_dtype,
    )

    return n_rows, n_cols, n_entries, coo


__all__ = [
    "GENE_EXPRESSION_MODALITY",
    "TenxImporterConfig",
    "TenxFeature",
    "stream_text_lines",
    "parse_10x_barcodes",
    "parse_10x_features_tsv",
    "validate_10x_features",
    "parse_10x_matrix_market",
]
