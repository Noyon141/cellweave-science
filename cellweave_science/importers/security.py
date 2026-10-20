"""Physical inspection, format detection, and compression handling for importer security.

Implements:
- Physical file inspection (SHA-256, byteSize, compression, container/magic-byte detection).
- Magic-byte detection (HDF5, GZIP, MatrixMarket).
- Bounded streaming gzip decompression with strict fail-closed safety limits against zip bombs.
- Trusted local path validation rejecting directory traversal, non-regular files, directories, and symlinks.
"""

from __future__ import annotations

import gzip
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from cellweave_science.importers.checksum import compute_file_sha256
from cellweave_science.importers.errors import (
    DecompressionLimitError,
    ImporterSecurityError,
    MalformedCompressedStreamError,
)

# Magic bytes constants
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
GZIP_MAGIC = b"\x1f\x8b"
MATRIX_MARKET_MAGIC = b"%%MatrixMarket"

STREAM_CHUNK_SIZE = 64 * 1024  # 64 KiB decompression chunks


class CompressionFormat(str, Enum):
    """Supported compression formats for Phase 3 importer storage."""

    NONE = "NONE"
    GZIP = "GZIP"


class PhysicalContainerFormat(str, Enum):
    """Detected low-level physical container or file format."""

    HDF5 = "HDF5"
    MATRIX_MARKET = "MATRIX_MARKET"
    DELIMITED_TEXT = "DELIMITED_TEXT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DecompressionLimits:
    """Configurable safety limits for stream decompression (Fail-Closed).

    Invariants:
    1. Both max_uncompressed_bytes and max_compression_ratio are mandatory and must be strictly > 0.
    2. No permissive None defaults: omissional unlimited decompression is strictly prohibited.
    3. Technical stream-decompression safety boundaries only; Phase 2 remains authoritative
       for commercial upload size policies.
    """

    max_uncompressed_bytes: int
    max_compression_ratio: float

    def __post_init__(self) -> None:
        if self.max_uncompressed_bytes is None or self.max_uncompressed_bytes <= 0:
            raise ImporterSecurityError(
                f"max_uncompressed_bytes must be an integer strictly greater than 0, got {self.max_uncompressed_bytes!r}"
            )
        if self.max_compression_ratio is None or self.max_compression_ratio <= 0.0:
            raise ImporterSecurityError(
                f"max_compression_ratio must be a float strictly greater than 0.0, got {self.max_compression_ratio!r}"
            )


@dataclass(frozen=True)
class PhysicalInspectionResult:
    """Internal science-domain result of physical file inspection.

    Crucial: Contains no DB, organization, or cloud credentials.
    """

    sha256: str
    byte_size: int
    compression: CompressionFormat
    detected_format: PhysicalContainerFormat


def validate_trusted_local_path(
    path: Path | str,
    *,
    allow_symlinks: bool = False,
) -> Path:
    """Validate that a local file path exists, is a regular file, and respects security policy.

    Args:
        path: File path to inspect.
        allow_symlinks: Whether to permit symbolic links (default False per security rule).

    Returns:
        Resolved Path instance.

    Raises:
        ImporterSecurityError: If path does not exist, is a directory, not a regular file, or violates policy.
    """
    try:
        p = Path(path)
    except (TypeError, ValueError) as exc:
        raise ImporterSecurityError(f"Invalid path representation: {exc}") from exc

    # Check path string for null characters
    if "\x00" in str(p):
        raise ImporterSecurityError("Null character in path string")

    if not p.exists():
        raise ImporterSecurityError(f"Target path does not exist: {p}")

    if not allow_symlinks and p.is_symlink():
        raise ImporterSecurityError(f"Symbolic links are strictly prohibited: {p}")

    if p.is_dir():
        raise ImporterSecurityError(f"Target path is unexpectedly a directory: {p}")

    if not p.is_file():
        raise ImporterSecurityError(f"Target path is not a regular file: {p}")

    return p.resolve()


def detect_compression(header_bytes: bytes) -> CompressionFormat:
    """Detect compression format from binary signature (magic bytes)."""
    if len(header_bytes) >= 2 and header_bytes[:2] == GZIP_MAGIC:
        return CompressionFormat.GZIP
    return CompressionFormat.NONE


def detect_physical_format(header_bytes: bytes) -> PhysicalContainerFormat:
    """Detect low-level container type from header bytes."""
    if len(header_bytes) >= len(HDF5_MAGIC) and header_bytes[: len(HDF5_MAGIC)] == HDF5_MAGIC:
        return PhysicalContainerFormat.HDF5

    if header_bytes.startswith(MATRIX_MARKET_MAGIC):
        return PhysicalContainerFormat.MATRIX_MARKET

    # Check if textual delimited (contains printable ASCII / UTF-8 lines with comma or tab)
    try:
        sample_text = header_bytes[:512].decode("utf-8")
        if "\n" in sample_text and ("," in sample_text or "\t" in sample_text):
            return PhysicalContainerFormat.DELIMITED_TEXT
    except UnicodeDecodeError:
        pass

    return PhysicalContainerFormat.UNKNOWN


def inspect_physical_file(
    file_path: Path | str,
    *,
    max_header_bytes: int = 1024,
) -> PhysicalInspectionResult:
    """Inspect physical properties of a local source file.

    Calculates:
    - Streaming SHA-256
    - Physical byte size
    - Compression format (NONE vs GZIP)
    - Physical container format (HDF5, MATRIX_MARKET, etc.)

    If file is GZIP compressed, also inspects uncompressed header to identify underlying format.
    """
    path = validate_trusted_local_path(file_path)
    byte_size = path.stat().st_size
    sha256 = compute_file_sha256(path)

    with open(path, "rb") as f:
        header = f.read(max_header_bytes)

    compression = detect_compression(header)

    if compression == CompressionFormat.GZIP:
        # Inspect inner decompressed bytes up to max_header_bytes
        try:
            with gzip.open(path, "rb") as gz:
                inner_header = gz.read(max_header_bytes)
            detected_format = detect_physical_format(inner_header)
        except (gzip.BadGzipFile, OSError) as exc:
            raise MalformedCompressedStreamError(
                f"Corrupted or invalid gzip file at {path}: {exc}"
            ) from exc
    else:
        detected_format = detect_physical_format(header)

    return PhysicalInspectionResult(
        sha256=sha256,
        byte_size=byte_size,
        compression=compression,
        detected_format=detected_format,
    )


def open_bounded_gzip_stream(
    file_path: Path | str,
    *,
    limits: DecompressionLimits,
    chunk_size: int = STREAM_CHUNK_SIZE,
) -> Generator[bytes, None, None]:
    """Yield decompressed byte chunks from a gzip file with mandatory, fail-closed safety bounds.

    Guards against:
    - Decompression bombs (excessive compression ratio).
    - Unlimited memory consumption.
    - Pathological decompression size.

    Args:
        file_path: Path to target gzip file.
        limits: Mandatory DecompressionLimits configuration instance.
        chunk_size: Stream read chunk size in bytes (default 64 KiB).

    Raises:
        ImporterSecurityError: If limits parameter is missing or invalid.
        DecompressionLimitError: If limits are exceeded.
        MalformedCompressedStreamError: If the gzip stream is corrupted.
    """
    if limits is None or not isinstance(limits, DecompressionLimits):
        raise ImporterSecurityError("DecompressionLimits must be explicitly provided (fail-closed)")

    path = validate_trusted_local_path(file_path)
    compressed_size = max(path.stat().st_size, 1)

    total_uncompressed = 0
    max_uncompressed_bytes = limits.max_uncompressed_bytes
    max_compression_ratio = limits.max_compression_ratio

    try:
        with gzip.open(path, "rb") as gz:
            while chunk := gz.read(chunk_size):
                total_uncompressed += len(chunk)
                if total_uncompressed > max_uncompressed_bytes:
                    raise DecompressionLimitError(
                        f"Uncompressed size exceeded limit of {max_uncompressed_bytes} bytes"
                    )
                ratio = total_uncompressed / compressed_size
                if total_uncompressed > 1024 and ratio > max_compression_ratio:
                    raise DecompressionLimitError(
                        f"Suspicious compression ratio ({ratio:.1f} > {max_compression_ratio}), potential decompression bomb"
                    )
                yield chunk
    except (gzip.BadGzipFile, OSError) as exc:
        if isinstance(exc, DecompressionLimitError):
            raise
        raise MalformedCompressedStreamError(
            f"Failed to decompress gzip stream at {path}: {exc}"
        ) from exc


__all__ = [
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
]
