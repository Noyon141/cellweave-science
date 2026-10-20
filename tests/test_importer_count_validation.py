"""Unit tests for Phase 3 Step D count matrix and dimension validation primitives."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from cellweave_science.contracts import ScientificIssueCode
from cellweave_science.importers.errors import CountValidationError
from cellweave_science.importers.validation import (
    CountClassification,
    classify_count_matrix,
    ensure_csr_matrix,
    validate_count_matrix,
)


def test_integer_dense_matrix_valid() -> None:
    """Standard non-negative integer numpy matrix passes validation as VALID_COUNTS."""
    mat = np.array([[0, 1, 5], [10, 0, 2]], dtype=np.int32)
    assert classify_count_matrix(mat) == CountClassification.VALID_COUNTS
    validate_count_matrix(mat)


def test_integer_csr_matrix_valid() -> None:
    """Standard non-negative integer CSR sparse matrix passes validation as VALID_COUNTS."""
    mat = sp.csr_matrix([[0, 1, 0], [100, 0, 2]], dtype=np.int64)
    assert classify_count_matrix(mat) == CountClassification.VALID_COUNTS
    validate_count_matrix(mat)


def test_float_exact_integral_matrix_valid() -> None:
    """Floating-point matrix with exact integral representation (0.0, 1.0) is VALID_COUNTS."""
    mat = np.array([[0.0, 1.0, 5.0], [10.0, 0.0, 2.0]], dtype=np.float32)
    assert classify_count_matrix(mat) == CountClassification.VALID_COUNTS
    validate_count_matrix(mat)


def test_float_near_integral_ambiguous() -> None:
    """Floating-point values with tiny difference within machine precision are AMBIGUOUS, not silently VALID_COUNTS."""
    eps = np.finfo(np.float64).eps * 2
    mat = np.array([[0.0, 1.0 + eps], [2.0, 0.0]], dtype=np.float64)
    assert classify_count_matrix(mat) == CountClassification.AMBIGUOUS

    with pytest.raises(CountValidationError) as exc_info:
        validate_count_matrix(mat)
    assert exc_info.value.issue_code == ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS


def test_float_fractional_normalized_not_counts() -> None:
    """Floating-point matrix with continuous normalized values (e.g. log1p) is NOT_COUNTS."""
    mat = np.array([[0.0, 1.45, 2.1], [0.5, 0.0, 3.8]], dtype=np.float64)
    assert classify_count_matrix(mat) == CountClassification.NOT_COUNTS

    with pytest.raises(CountValidationError) as exc_info:
        validate_count_matrix(mat)
    assert exc_info.value.issue_code == ScientificIssueCode.NON_COUNT_INPUT


def test_negative_values_not_counts() -> None:
    """Negative values are classified as NOT_COUNTS and rejected."""
    mat = np.array([[0, -1, 5], [10, 0, 2]], dtype=np.int32)
    assert classify_count_matrix(mat) == CountClassification.NOT_COUNTS
    with pytest.raises(CountValidationError, match="negative expression values") as exc_info:
        validate_count_matrix(mat)
    assert exc_info.value.issue_code == ScientificIssueCode.NON_COUNT_INPUT


def test_nan_values_not_counts() -> None:
    """NaN values are classified as NOT_COUNTS and rejected."""
    mat = np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float32)
    assert classify_count_matrix(mat) == CountClassification.NOT_COUNTS
    with pytest.raises(CountValidationError, match="NaN or Infinite") as exc_info:
        validate_count_matrix(mat)
    assert exc_info.value.issue_code == ScientificIssueCode.NON_COUNT_INPUT


def test_infinite_values_not_counts() -> None:
    """Infinite values (+/-Inf) are classified as NOT_COUNTS and rejected."""
    mat = np.array([[0.0, np.inf], [1.0, 2.0]], dtype=np.float32)
    assert classify_count_matrix(mat) == CountClassification.NOT_COUNTS
    with pytest.raises(CountValidationError, match="NaN or Infinite") as exc_info:
        validate_count_matrix(mat)
    assert exc_info.value.issue_code == ScientificIssueCode.NON_COUNT_INPUT


def test_empty_dimensions_rejected() -> None:
    """Matrices with zero rows or columns are rejected with DIMENSION_MISMATCH."""
    empty_mat = np.zeros((0, 10), dtype=np.int32)
    assert classify_count_matrix(empty_mat) == CountClassification.NOT_COUNTS
    with pytest.raises(CountValidationError, match="dimensions must be positive") as exc_info:
        validate_count_matrix(empty_mat)
    assert exc_info.value.issue_code == ScientificIssueCode.DIMENSION_MISMATCH


def test_sparse_preservation_without_densification() -> None:
    """Sparse matrix validation operates strictly on .data without densifying into RAM."""
    # Logical sparse matrix: 10,000 x 50,000 cells with 2 non-zero counts
    sparse_mat = sp.csr_matrix(
        ([10, 5], ([0, 9999], [0, 49999])), shape=(10000, 50000), dtype=np.int32
    )
    assert sp.issparse(sparse_mat)
    assert classify_count_matrix(sparse_mat) == CountClassification.VALID_COUNTS
    validate_count_matrix(sparse_mat)


def test_ensure_csr_matrix_representation() -> None:
    """Dense and CSC matrices are converted to CSR without transposing."""
    dense = np.array([[1, 0], [0, 2]], dtype=np.int32)
    csr1 = ensure_csr_matrix(dense)
    assert sp.isspmatrix_csr(csr1)
    assert csr1.shape == (2, 2)
    assert csr1[0, 0] == 1 and csr1[1, 1] == 2

    csc = sp.csc_matrix(dense)
    csr2 = ensure_csr_matrix(csc)
    assert sp.isspmatrix_csr(csr2)
    assert (csr2.toarray() == dense).all()
