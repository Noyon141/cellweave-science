"""Streaming cryptographic checksum calculations for scientific dataset files.

Provides:
- Streaming per-file SHA-256 computation over bounded chunks without loading large matrices into RAM.
- Deterministic multi-file aggregate source checksum calculation adhering to Phase 3 specification.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from cellweave_science.contracts import ImportFileRole, SourceFormat
from cellweave_science.importers.errors import ImporterSecurityError

DEFAULT_CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MiB chunk size for streaming
_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def compute_file_sha256(
    file_path_or_stream: Path | str | BinaryIO,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> str:
    """Compute the SHA-256 checksum of a file or binary stream in bounded chunks.

    Args:
        file_path_or_stream: Path to a regular file or a readable binary stream.
        chunk_size: Size of byte chunks to read per iteration (default: 1 MiB).

    Returns:
        Lowercase 64-character hexadecimal SHA-256 string.

    Raises:
        ImporterSecurityError: If the path is not a regular file or cannot be read.
    """
    hasher = hashlib.sha256()

    if isinstance(file_path_or_stream, Path | str):
        path = Path(file_path_or_stream)
        if not path.is_file():
            raise ImporterSecurityError(
                f"Target path does not exist or is not a regular file: {path}"
            )
        try:
            with open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
        except OSError as exc:
            raise ImporterSecurityError(f"Failed to read file for checksum {path}: {exc}") from exc
    else:
        # Readable binary stream
        try:
            while chunk := file_path_or_stream.read(chunk_size):
                hasher.update(chunk)
        except OSError as exc:
            raise ImporterSecurityError(f"Failed to read stream for checksum: {exc}") from exc

    return hasher.hexdigest().lower()


@dataclass(frozen=True)
class SourceChecksumFile:
    """Descriptor for a verified source file participating in aggregate checksum calculation.

    Attributes:
        dataset_file_id: Internal dataset file identifier.
        file_name: Exact persisted safe file name (case-sensitive).
        byte_size: Exact physical byte size (non-negative integer).
        sha256: 64-character lowercase hexadecimal SHA-256 hash.
        role: Functional role of the file in the dataset.
        ordinal: Optional ordering ordinal for multi-file groups.
    """

    dataset_file_id: str
    file_name: str
    byte_size: int
    sha256: str
    role: ImportFileRole
    ordinal: int = 0

    def __post_init__(self) -> None:
        if not self.file_name or "\x00" in self.file_name:
            raise ImporterSecurityError(f"Invalid persisted filename: {self.file_name!r}")
        if self.byte_size < 0:
            raise ImporterSecurityError(f"File byte size cannot be negative: {self.byte_size}")
        if not _SHA256_HEX_PATTERN.match(self.sha256):
            raise ImporterSecurityError(
                f"File SHA-256 must be a 64-character lowercase hex string: {self.sha256!r}"
            )


def compute_source_checksum(
    source_format: SourceFormat,
    files: tuple[SourceChecksumFile, ...] | list[SourceChecksumFile],
) -> str:
    """Compute deterministic Phase 3 sourceChecksum.

    Algorithm specification:
    - Single-file formats (H5AD, TENX_H5):
        sourceChecksum = exact per-file SHA-256.
        Requires exactly 1 file with role PRIMARY.
    - Multi-file formats (TENX_MTX, GENERIC_MATRIX):
        1. Reject duplicate persisted file names.
        2. Format each file entry as:
            {persistedFileName}\\t{byteSize}\\t{sha256}\\n
           with UTF-8 encoding and LF line endings.
        3. Sort entries lexicographically ascending by exact persisted filename (case-sensitive).
        4. sourceChecksum = SHA-256(concatenated manifest bytes).

    Args:
        source_format: Format enum of the dataset.
        files: Collection of SourceChecksumFile records.

    Returns:
        Lowercase 64-character hexadecimal SHA-256 string.

    Raises:
        ImporterSecurityError: If files violate cardinality, unique naming, or format rules.
    """
    if not files:
        raise ImporterSecurityError("Cannot compute sourceChecksum for empty file set")

    if source_format in (SourceFormat.H5AD, SourceFormat.TENX_H5):
        if len(files) != 1:
            raise ImporterSecurityError(
                f"Single-file format {source_format.value} requires exactly 1 file, received {len(files)}"
            )
        single = files[0]
        if single.role != ImportFileRole.PRIMARY:
            raise ImporterSecurityError(
                f"Single-file format {source_format.value} requires role PRIMARY, received {single.role.value}"
            )
        return single.sha256.lower()

    if source_format in (SourceFormat.TENX_MTX, SourceFormat.GENERIC_MATRIX):
        # Multi-file deterministic manifest hash
        seen_names: set[str] = set()
        for f in files:
            if f.file_name in seen_names:
                raise ImporterSecurityError(
                    f"Duplicate persisted filename detected in import set: {f.file_name!r}"
                )
            seen_names.add(f.file_name)

        # Lexicographical sort ascending by exact persisted filename
        sorted_files = sorted(files, key=lambda x: x.file_name)

        manifest_lines: list[bytes] = []
        for f in sorted_files:
            line_str = f"{f.file_name}\t{f.byte_size}\t{f.sha256}\n"
            manifest_lines.append(line_str.encode("utf-8"))

        manifest_payload = b"".join(manifest_lines)
        return hashlib.sha256(manifest_payload).hexdigest().lower()

    raise ImporterSecurityError(f"Unsupported source format for sourceChecksum: {source_format}")


__all__ = [
    "DEFAULT_CHUNK_SIZE_BYTES",
    "SourceChecksumFile",
    "compute_file_sha256",
    "compute_source_checksum",
]
