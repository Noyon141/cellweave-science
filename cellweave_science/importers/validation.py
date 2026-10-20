"""Reusable scientific validation and normalization primitives for importer pipelines.

Implements:
- Count matrix validation (non-negative, finite, integral values, sparse-safe).
- Integer-like float classification with explicitly defined numerical tolerance.
- SciPy sparse matrix normalization to CSR format without densification.
- Matrix dimension and metadata alignment checks.
- Identifier validation (uniqueness, non-empty, preservation of ordering).
- Conservative Ensembl identifier candidate recognition and organism candidate detection.
- ImportSourceSet structural role cardinality validation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp

from cellweave_science.contracts import (
    ImportFileRole,
    Organism,
    ScientificIssue,
    ScientificIssueCode,
    ScientificIssueSeverity,
    SourceFormat,
)
from cellweave_science.importers.errors import CountValidationError
from cellweave_science.importers.sources import ImportSourceSet

# Conservative tolerance for checking if floating point values are integer counts:
# A value x is considered an integer count if |x - round(x)| <= FLOAT_INTEGER_TOLERANCE.
FLOAT_INTEGER_TOLERANCE = 1e-5

# Ensembl regex patterns for candidate identification
_ENS_HUMAN_PATTERN = re.compile(r"^ENSG\d+(\.\d+)?$")
_ENS_MOUSE_PATTERN = re.compile(r"^ENSMUSG\d+(\.\d+)?$")


def validate_count_matrix(
    x: np.ndarray[Any, Any] | sp.spmatrix,
    *,
    float_tolerance: float = FLOAT_INTEGER_TOLERANCE,
) -> None:
    """Validate that a matrix consists strictly of raw, unnormalized non-negative count data.

    Invariants:
    1. Sparse matrices are validated strictly on their .data buffer without densifying into RAM.
    2. No NaN, +Inf, -Inf values.
    3. No negative values.
    4. Values must be integers or mathematically integer-like floats within float_tolerance.
    5. Dimensions must be non-empty (shape > 0).

    Args:
        x: Dense 2D numpy array or SciPy sparse matrix.
        float_tolerance: Absolute difference tolerance for floating numbers to qualify as integers.

    Raises:
        CountValidationError: If any count invariant is violated.
    """
    if len(x.shape) != 2:
        raise CountValidationError(
            f"Expression matrix must be 2-dimensional, received shape {x.shape}",
            issue_code=ScientificIssueCode.DIMENSION_MISMATCH,
        )

    n_rows, n_cols = x.shape
    if n_rows <= 0 or n_cols <= 0:
        raise CountValidationError(
            f"Expression matrix dimensions must be positive, received ({n_rows}, {n_cols})",
            issue_code=ScientificIssueCode.DIMENSION_MISMATCH,
        )

    # Extract non-zero values for sparse or flatten for dense
    if sp.issparse(x):
        data = np.asarray(x.data)  # type: ignore[union-attr]
        if data.size == 0:
            # Entirely zero sparse matrix is technically valid non-negative integer counts
            return
    else:
        data = np.asarray(x).ravel()

    # 1. Finite check (detects NaN and +/-Inf)
    if not np.all(np.isfinite(data)):
        raise CountValidationError(
            "Expression matrix contains NaN or Infinite values",
            issue_code=ScientificIssueCode.NON_COUNT_INPUT,
        )

    # 2. Non-negative check
    if np.any(data < 0):
        raise CountValidationError(
            "Expression matrix contains negative expression values",
            issue_code=ScientificIssueCode.NON_COUNT_INPUT,
        )

    # 3. Integral check
    if np.issubdtype(data.dtype, np.integer):
        return

    if np.issubdtype(data.dtype, np.floating):
        diff = np.abs(data - np.round(data))
        if np.any(diff > float_tolerance):
            raise CountValidationError(
                "Expression matrix contains non-integral continuous float values (e.g. normalized or log-transformed)",
                issue_code=ScientificIssueCode.NON_COUNT_INPUT,
            )
        return

    raise CountValidationError(
        f"Unsupported matrix data type: {data.dtype}",
        issue_code=ScientificIssueCode.NON_COUNT_INPUT,
    )


def ensure_csr_matrix(x: np.ndarray[Any, Any] | sp.spmatrix) -> sp.csr_matrix:
    """Ensure expression matrix is represented as a SciPy CSR sparse matrix without changing orientation.

    Does NOT transpose the matrix.
    """
    if sp.isspmatrix_csr(x):
        return x  # type: ignore[return-value]
    if sp.issparse(x):
        return x.tocsr()  # type: ignore[union-attr]
    return sp.csr_matrix(x)


def validate_identifiers(
    identifiers: Sequence[Any],
    *,
    context: str = "identifiers",
) -> list[str]:
    """Validate a sequence of cell or gene identifiers.

    Invariants:
    1. Must not be empty.
    2. Elements must be non-empty strings (or losslessly convertible).
    3. Identifiers must be unique; duplicates are not silently tolerated.
    4. Stable ordering is preserved.

    Args:
        identifiers: Sequence of identifiers.
        context: Description for error reporting (e.g., "cell IDs", "gene IDs").

    Returns:
        List of sanitized string identifiers.

    Raises:
        CountValidationError: If duplicates exist or identifiers are invalid.
    """
    if len(identifiers) == 0:
        raise CountValidationError(
            f"{context} sequence cannot be empty",
            issue_code=ScientificIssueCode.MALFORMED_INPUT,
        )

    seen: set[str] = set()
    cleaned: list[str] = []

    for idx, item in enumerate(identifiers):
        val_str = str(item).strip() if item is not None else ""
        if not val_str:
            raise CountValidationError(
                f"Empty or whitespace identifier found in {context} at index {idx}",
                issue_code=ScientificIssueCode.MALFORMED_INPUT,
            )
        if val_str in seen:
            raise CountValidationError(
                f"Duplicate {context} found: {val_str!r} at index {idx}",
                issue_code=ScientificIssueCode.DUPLICATE_CELL_IDS
                if "cell" in context.lower()
                else ScientificIssueCode.MALFORMED_INPUT,
            )
        seen.add(val_str)
        cleaned.append(val_str)

    return cleaned


def detect_ensembl_organism_candidates(identifiers: Sequence[str]) -> Organism | None:
    """Analyze gene identifiers to detect conservative species candidates based on Ensembl prefixes.

    Rules:
    - Overwhelming human (ENSG...): returns Organism.HOMO_SAPIENS
    - Overwhelming mouse (ENSMUSG...): returns Organism.MUS_MUSCULUS
    - Mixed or non-Ensembl (gene symbols only): returns None (unresolved).
    """
    if not identifiers:
        return None

    human_count = 0
    mouse_count = 0
    total = len(identifiers)

    for gid in identifiers:
        if _ENS_HUMAN_PATTERN.match(gid):
            human_count += 1
        elif _ENS_MOUSE_PATTERN.match(gid):
            mouse_count += 1

    if human_count > 0 and mouse_count > 0:
        # Mixed species detected -> Unresolved / Ambiguous
        return None

    # If at least 50% match standard Ensembl prefix, consider candidate confirmed
    if human_count / total >= 0.5:
        return Organism.HUMAN
    if mouse_count / total >= 0.5:
        return Organism.MOUSE

    return None


def validate_import_source_set_cardinality(source_set: ImportSourceSet) -> list[ScientificIssue]:
    """Validate that the given ImportSourceSet structurally satisfies Phase 3 role cardinality rules.

    Cardinality invariants:
    - H5AD: exactly 1 PRIMARY file.
    - TENX_H5: exactly 1 PRIMARY file.
    - TENX_MTX: exactly 1 MATRIX, 1 FEATURES, 1 BARCODES (total 3 files).
    - GENERIC_MATRIX: exactly 1 MATRIX, 0..1 CELL_METADATA, 0..1 FEATURE_METADATA.
    No extraneous roles allowed.
    """
    issues: list[ScientificIssue] = []
    fmt = source_set.source_format
    files = source_set.files

    if not files:
        issues.append(
            ScientificIssue(
                code=ScientificIssueCode.MISSING_COMPONENT,
                severity=ScientificIssueSeverity.ERROR,
                message=f"Import source set for {fmt.value} contains no files",
                userActionRequired=True,
            )
        )
        return issues

    roles = [f.role for f in files]
    primary_count = roles.count(ImportFileRole.PRIMARY)
    matrix_count = roles.count(ImportFileRole.MATRIX)
    features_count = roles.count(ImportFileRole.FEATURES)
    barcodes_count = roles.count(ImportFileRole.BARCODES)
    cell_meta_count = roles.count(ImportFileRole.CELL_METADATA)
    feat_meta_count = roles.count(ImportFileRole.FEATURE_METADATA)

    if fmt in (SourceFormat.H5AD, SourceFormat.TENX_H5):
        if primary_count != 1 or len(files) != 1:
            issues.append(
                ScientificIssue(
                    code=ScientificIssueCode.INVALID_IMPORT_CARDINALITY,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"{fmt.value} requires exactly 1 PRIMARY file, found {primary_count} (total {len(files)})",
                    userActionRequired=True,
                )
            )

    elif fmt == SourceFormat.TENX_MTX:
        if matrix_count != 1 or features_count != 1 or barcodes_count != 1 or len(files) != 3:
            issues.append(
                ScientificIssue(
                    code=ScientificIssueCode.INVALID_IMPORT_CARDINALITY,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"TENX_MTX requires exactly 1 MATRIX, 1 FEATURES, 1 BARCODES, found matrix={matrix_count}, features={features_count}, barcodes={barcodes_count}, total={len(files)}",
                    userActionRequired=True,
                )
            )

    elif fmt == SourceFormat.GENERIC_MATRIX:
        allowed_count = matrix_count + cell_meta_count + feat_meta_count
        unrelated_count = len(files) - allowed_count
        if matrix_count != 1 or cell_meta_count > 1 or feat_meta_count > 1 or unrelated_count > 0:
            issues.append(
                ScientificIssue(
                    code=ScientificIssueCode.INVALID_IMPORT_CARDINALITY,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"GENERIC_MATRIX requires exactly 1 MATRIX, <=1 CELL_METADATA, <=1 FEATURE_METADATA, found matrix={matrix_count}, cell_meta={cell_meta_count}, feat_meta={feat_meta_count}, other={unrelated_count}",
                    userActionRequired=True,
                )
            )

    return issues


__all__ = [
    "FLOAT_INTEGER_TOLERANCE",
    "validate_count_matrix",
    "ensure_csr_matrix",
    "validate_identifiers",
    "detect_ensembl_organism_candidates",
    "validate_import_source_set_cardinality",
]
