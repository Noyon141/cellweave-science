"""Unit tests for Phase 3 Step D streaming SHA-256 and multi-file source checksum."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from cellweave_science.contracts import ImportFileRole, SourceFormat
from cellweave_science.importers.checksum import (
    SourceChecksumFile,
    compute_file_sha256,
    compute_source_checksum,
)
from cellweave_science.importers.errors import ImporterSecurityError


def test_streaming_sha256_empty_file(tmp_path: Path) -> None:
    """Empty file returns standard SHA-256 hash."""
    empty_file = tmp_path / "empty.bin"
    empty_file.write_bytes(b"")

    sha = compute_file_sha256(empty_file)
    assert sha == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_streaming_sha256_binary_chunks(tmp_path: Path) -> None:
    """Test chunked reading boundary."""
    data = b"Hello, CellWeave scientific engine!" * 100
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(data)

    # Hash with standard chunk size
    sha_std = compute_file_sha256(test_file)

    # Hash with tiny chunk size (forcing multiple reads across boundaries)
    sha_tiny = compute_file_sha256(test_file, chunk_size=17)

    # Hash via binary stream directly
    stream = io.BytesIO(data)
    sha_stream = compute_file_sha256(stream, chunk_size=32)

    assert sha_std == sha_tiny
    assert sha_std == sha_stream
    assert len(sha_std) == 64


def test_streaming_sha256_missing_file(tmp_path: Path) -> None:
    """Missing file raises ImporterSecurityError."""
    missing = tmp_path / "does_not_exist.h5ad"
    with pytest.raises(ImporterSecurityError, match="does not exist or is not a regular file"):
        compute_file_sha256(missing)


def test_single_file_source_checksum_h5ad() -> None:
    """Single file H5AD returns exact per-file SHA-256."""
    file_sha = "a" * 64
    f = SourceChecksumFile(
        dataset_file_id="dsf_1",
        file_name="dataset.h5ad",
        byte_size=1024,
        sha256=file_sha,
        role=ImportFileRole.PRIMARY,
    )

    result = compute_source_checksum(SourceFormat.H5AD, [f])
    assert result == file_sha


def test_single_file_source_checksum_tenx_h5() -> None:
    """Single file TENX_H5 returns exact per-file SHA-256."""
    file_sha = "b" * 64
    f = SourceChecksumFile(
        dataset_file_id="dsf_1",
        file_name="filtered_feature_bc_matrix.h5",
        byte_size=2048,
        sha256=file_sha,
        role=ImportFileRole.PRIMARY,
    )

    result = compute_source_checksum(SourceFormat.TENX_H5, [f])
    assert result == file_sha


def test_single_file_wrong_role_rejected() -> None:
    """Single file formats require PRIMARY role."""
    f = SourceChecksumFile(
        dataset_file_id="dsf_1",
        file_name="dataset.h5ad",
        byte_size=1024,
        sha256="c" * 64,
        role=ImportFileRole.MATRIX,
    )
    with pytest.raises(ImporterSecurityError, match="requires role PRIMARY"):
        compute_source_checksum(SourceFormat.H5AD, [f])


def test_tenx_mtx_deterministic_aggregate() -> None:
    """3-file TENX_MTX produces deterministic aggregate regardless of input array order."""
    f_barcodes = SourceChecksumFile(
        dataset_file_id="dsf_barcodes",
        file_name="barcodes.tsv.gz",
        byte_size=256,
        sha256="1" * 64,
        role=ImportFileRole.BARCODES,
    )
    f_features = SourceChecksumFile(
        dataset_file_id="dsf_features",
        file_name="features.tsv.gz",
        byte_size=512,
        sha256="2" * 64,
        role=ImportFileRole.FEATURES,
    )
    f_matrix = SourceChecksumFile(
        dataset_file_id="dsf_matrix",
        file_name="matrix.mtx.gz",
        byte_size=1024,
        sha256="3" * 64,
        role=ImportFileRole.MATRIX,
    )

    # Order 1: barcodes, features, matrix
    hash1 = compute_source_checksum(SourceFormat.TENX_MTX, [f_barcodes, f_features, f_matrix])

    # Order 2: matrix, barcodes, features
    hash2 = compute_source_checksum(SourceFormat.TENX_MTX, [f_matrix, f_barcodes, f_features])

    # Order 3: features, matrix, barcodes
    hash3 = compute_source_checksum(SourceFormat.TENX_MTX, [f_features, f_matrix, f_barcodes])

    assert hash1 == hash2 == hash3
    assert len(hash1) == 64


def test_multi_file_checksum_sensitivity() -> None:
    """Any difference in filename, size, or hash alters the aggregate checksum."""
    base_m = SourceChecksumFile(
        dataset_file_id="dsf_m",
        file_name="matrix.mtx.gz",
        byte_size=1000,
        sha256="a" * 64,
        role=ImportFileRole.MATRIX,
    )
    base_b = SourceChecksumFile(
        dataset_file_id="dsf_b",
        file_name="barcodes.tsv.gz",
        byte_size=200,
        sha256="b" * 64,
        role=ImportFileRole.BARCODES,
    )
    base_f = SourceChecksumFile(
        dataset_file_id="dsf_f",
        file_name="features.tsv.gz",
        byte_size=300,
        sha256="c" * 64,
        role=ImportFileRole.FEATURES,
    )

    base_agg = compute_source_checksum(SourceFormat.TENX_MTX, [base_m, base_b, base_f])

    # 1. Different filename (genes.tsv.gz instead of features.tsv.gz)
    diff_name_f = SourceChecksumFile(
        dataset_file_id="dsf_f",
        file_name="genes.tsv.gz",
        byte_size=300,
        sha256="c" * 64,
        role=ImportFileRole.FEATURES,
    )
    diff_name_agg = compute_source_checksum(SourceFormat.TENX_MTX, [base_m, base_b, diff_name_f])
    assert diff_name_agg != base_agg

    # 2. Case sensitivity (Matrix.mtx.gz vs matrix.mtx.gz)
    diff_case_m = SourceChecksumFile(
        dataset_file_id="dsf_m",
        file_name="Matrix.mtx.gz",
        byte_size=1000,
        sha256="a" * 64,
        role=ImportFileRole.MATRIX,
    )
    diff_case_agg = compute_source_checksum(SourceFormat.TENX_MTX, [diff_case_m, base_b, base_f])
    assert diff_case_agg != base_agg

    # 3. Different byte size
    diff_size_m = SourceChecksumFile(
        dataset_file_id="dsf_m",
        file_name="matrix.mtx.gz",
        byte_size=1001,
        sha256="a" * 64,
        role=ImportFileRole.MATRIX,
    )
    diff_size_agg = compute_source_checksum(SourceFormat.TENX_MTX, [diff_size_m, base_b, base_f])
    assert diff_size_agg != base_agg

    # 4. Different sha
    diff_sha_m = SourceChecksumFile(
        dataset_file_id="dsf_m",
        file_name="matrix.mtx.gz",
        byte_size=1000,
        sha256="f" * 64,
        role=ImportFileRole.MATRIX,
    )
    diff_sha_agg = compute_source_checksum(SourceFormat.TENX_MTX, [diff_sha_m, base_b, base_f])
    assert diff_sha_agg != base_agg


def test_duplicate_filename_rejection() -> None:
    """Duplicate persisted file names in one import set are strictly rejected."""
    f1 = SourceChecksumFile(
        dataset_file_id="dsf_1",
        file_name="matrix.csv",
        byte_size=100,
        sha256="a" * 64,
        role=ImportFileRole.MATRIX,
    )
    f2 = SourceChecksumFile(
        dataset_file_id="dsf_2",
        file_name="matrix.csv",
        byte_size=100,
        sha256="a" * 64,
        role=ImportFileRole.CELL_METADATA,
    )
    with pytest.raises(ImporterSecurityError, match="Duplicate persisted filename"):
        compute_source_checksum(SourceFormat.GENERIC_MATRIX, [f1, f2])


def test_invalid_sha_syntax_in_descriptor() -> None:
    """Invalid SHA syntax raises ImporterSecurityError upon descriptor instantiation."""
    with pytest.raises(ImporterSecurityError, match="must be a 64-character lowercase hex string"):
        SourceChecksumFile(
            dataset_file_id="dsf_bad",
            file_name="file.csv",
            byte_size=10,
            sha256="NOT_A_VALID_HEX",
            role=ImportFileRole.PRIMARY,
        )
