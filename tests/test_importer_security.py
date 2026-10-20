"""Unit tests for Phase 3 Step D physical inspection, format detection, and compression security."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from cellweave_science.importers.errors import (
    DecompressionLimitError,
    ImporterSecurityError,
    MalformedCompressedStreamError,
)
from cellweave_science.importers.security import (
    GZIP_MAGIC,
    HDF5_MAGIC,
    MATRIX_MARKET_MAGIC,
    CompressionFormat,
    DecompressionLimits,
    PhysicalContainerFormat,
    detect_compression,
    detect_physical_format,
    inspect_physical_file,
    open_bounded_gzip_stream,
    validate_trusted_local_path,
)


def test_hdf5_magic_byte_recognition() -> None:
    """Header starting with HDF5 magic bytes is identified as HDF5."""
    assert detect_physical_format(HDF5_MAGIC + b"extra_content") == PhysicalContainerFormat.HDF5


def test_fake_h5_extension_with_non_hdf5_bytes(tmp_path: Path) -> None:
    """A file with .h5 or .h5ad extension but plaintext bytes is detected as UNKNOWN/DELIMITED, not HDF5."""
    fake_h5 = tmp_path / "fake.h5ad"
    fake_h5.write_text("not really an hdf5 file", encoding="utf-8")

    result = inspect_physical_file(fake_h5)
    assert result.detected_format != PhysicalContainerFormat.HDF5
    assert result.compression == CompressionFormat.NONE


def test_matrix_market_magic_recognition() -> None:
    """Header starting with %%MatrixMarket is recognized as MATRIX_MARKET."""
    assert (
        detect_physical_format(MATRIX_MARKET_MAGIC + b" matrix coordinate real general")
        == PhysicalContainerFormat.MATRIX_MARKET
    )


def test_gzip_compression_detection() -> None:
    """Magic bytes 1F 8B are identified as GZIP compression."""
    assert detect_compression(GZIP_MAGIC + b"\x08") == CompressionFormat.GZIP
    assert detect_compression(b"plain text") == CompressionFormat.NONE


def test_fake_gz_extension_plain_bytes(tmp_path: Path) -> None:
    """A file named something.gz with uncompressed bytes has compression NONE."""
    fake_gz = tmp_path / "fake.tsv.gz"
    fake_gz.write_bytes(b"col1\tcol2\nval1\tval2\n")

    result = inspect_physical_file(fake_gz)
    assert result.compression == CompressionFormat.NONE
    assert result.detected_format == PhysicalContainerFormat.DELIMITED_TEXT


def test_corrupt_gzip_file_fails(tmp_path: Path) -> None:
    """File starting with gzip magic bytes but corrupt payload raises MalformedCompressedStreamError."""
    corrupt_gz = tmp_path / "corrupt.gz"
    corrupt_gz.write_bytes(GZIP_MAGIC + b"\x00\x00\x00corrupted_payload")

    with pytest.raises(MalformedCompressedStreamError, match="Corrupted or invalid gzip file"):
        inspect_physical_file(corrupt_gz)


def test_genuine_gzipped_matrix_market(tmp_path: Path) -> None:
    """Gzip-wrapped MatrixMarket file is recognized with compression GZIP and inner format MATRIX_MARKET."""
    mtx_gz = tmp_path / "matrix.mtx.gz"
    content = MATRIX_MARKET_MAGIC + b" matrix coordinate integer general\n3 3 1\n1 1 5\n"
    with gzip.open(mtx_gz, "wb") as gz:
        gz.write(content)

    result = inspect_physical_file(mtx_gz)
    assert result.compression == CompressionFormat.GZIP
    assert result.detected_format == PhysicalContainerFormat.MATRIX_MARKET
    assert result.byte_size > 0
    assert len(result.sha256) == 64


def test_bounded_gzip_stream_safe_read(tmp_path: Path) -> None:
    """Bounded gzip stream yields expected decompressed chunks."""
    target = tmp_path / "data.gz"
    data = b"CellWeave safe gzip testing chunk! " * 50
    with gzip.open(target, "wb") as gz:
        gz.write(data)

    chunks = list(open_bounded_gzip_stream(target, chunk_size=64))
    reconstructed = b"".join(chunks)
    assert reconstructed == data


def test_decompression_max_bytes_limit(tmp_path: Path) -> None:
    """Exceeding small explicit max_uncompressed_bytes raises DecompressionLimitError without allocating huge files."""
    target = tmp_path / "test_bytes.gz"
    data = b"ABCDEFGHIJ" * 100  # 1000 bytes
    with gzip.open(target, "wb") as gz:
        gz.write(data)

    limits = DecompressionLimits(max_uncompressed_bytes=200)
    with pytest.raises(
        DecompressionLimitError, match="Uncompressed size exceeded limit of 200 bytes"
    ):
        for _ in open_bounded_gzip_stream(target, limits=limits, chunk_size=50):
            pass


def test_decompression_ratio_limit(tmp_path: Path) -> None:
    """Exceeding explicit compression ratio limit triggers DecompressionLimitError."""
    target = tmp_path / "bomb_ratio.gz"
    # Repetitive zeros compress to tiny gzip
    zeros = b"\x00" * 20000
    with gzip.open(target, "wb") as gz:
        gz.write(zeros)

    limits = DecompressionLimits(max_compression_ratio=5.0)
    with pytest.raises(DecompressionLimitError, match="Suspicious compression ratio"):
        for _ in open_bounded_gzip_stream(target, limits=limits, chunk_size=512):
            pass


def test_validate_trusted_local_path_rejections(tmp_path: Path) -> None:
    """Test path security: missing path, directory, symlink, null bytes."""
    # 1. Missing path
    missing = tmp_path / "non_existent.txt"
    with pytest.raises(ImporterSecurityError, match="Target path does not exist"):
        validate_trusted_local_path(missing)

    # 2. Directory instead of file
    directory = tmp_path / "some_dir"
    directory.mkdir()
    with pytest.raises(ImporterSecurityError, match="Target path is unexpectedly a directory"):
        validate_trusted_local_path(directory)

    # 3. Null byte injection
    with pytest.raises(ImporterSecurityError, match="Null character in path string"):
        validate_trusted_local_path("some/path\x00/file.txt")

    # 4. Symlink rejection
    target_file = tmp_path / "target.txt"
    target_file.write_text("safe target", encoding="utf-8")
    symlink_file = tmp_path / "link.txt"
    try:
        symlink_file.symlink_to(target_file)
        with pytest.raises(ImporterSecurityError, match="Symbolic links are strictly prohibited"):
            validate_trusted_local_path(symlink_file)
    except OSError:
        pass  # Windows unprivileged symlink fallback
