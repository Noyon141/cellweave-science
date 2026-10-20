"""Immutable canonical artifact publication via scoped HTTP capabilities.

Implements Phase 3 Step H.2 science-side publication invariants:
- No Cloudflare R2 master credentials, bucket/account env vars, or boto3.
- No organizationId / projectId tenant-context storage authority or R2 key construction.
- Validates local canonical .h5ad artifact (regular file, no symlinks, HDF5 magic bytes,
  AnnData readability, Policy B raw counts in CSR X, slot hygiene, confirmed organism & gene_id_scheme).
- Validates capability expiry, datasetVersionId consistency, HTTPS URLs, and mandatory
  If-None-Match: "*" header before any network call.
- Streams upload body from disk in bounded chunks without buffering the full artifact into memory.
- Executes mandatory streaming GET read-after-write verification (computing SHA-256 and exact
  byte size from the remote stream; never trusting ETag or object metadata headers).
- Reconciles 412 Precondition Failed idempotently as ALREADY_PRESENT only if remote bytes
  match local bytes and SHA-256; rejects conflicting remote bytes with CanonicalArtifactIntegrityError.
- Verifies local file immutability before and after upload/reconciliation.
- Redacts presigned URL query strings from all exception messages and logs.
"""

from __future__ import annotations

import hashlib
import http.client
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable
from urllib.parse import SplitResult, urlsplit, urlunsplit

import anndata as ad

from cellweave_science.contracts import (
    CanonicalArtifactPublicationCapabilities,
    CanonicalArtifactPublicationResult,
    CanonicalArtifactPublicationStatus,
)
from cellweave_science.importers.canonical import validate_canonical_anndata
from cellweave_science.importers.checksum import DEFAULT_CHUNK_SIZE_BYTES, compute_file_sha256
from cellweave_science.importers.errors import CanonicalValidationError
from cellweave_science.importers.security import (
    HDF5_MAGIC,
    CompressionFormat,
    detect_compression,
    validate_trusted_local_path,
)

DEFAULT_HTTP_TIMEOUT_SECONDS = 120.0


class CanonicalArtifactPublicationError(RuntimeError):
    """Base exception for canonical artifact publication failures."""


class CanonicalArtifactCapabilityExpiredError(CanonicalArtifactPublicationError):
    """Raised when a canonical publication capability is expired or has an invalid timestamp."""


class CanonicalArtifactIntegrityError(CanonicalArtifactPublicationError):
    """Raised when local/remote canonical artifact bytes or checksums fail integrity reconciliation."""


class CanonicalArtifactTransportError(CanonicalArtifactPublicationError):
    """Raised when the capability HTTP transport fails, redirects, or returns an unexpected status."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PutDisposition(str, Enum):
    """Normalized outcome of conditional PUT upload."""

    CREATED = "CREATED"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"


@dataclass(frozen=True)
class RemoteStreamVerification:
    """Result of streaming GET read-after-write verification from the remote capability URL."""

    sha256: str
    byte_size: int


@dataclass(frozen=True)
class LocalFileStatSnapshot:
    """Platform-safe local file metadata snapshot to detect concurrent file mutation."""

    byte_size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def capture(cls, path: Path) -> LocalFileStatSnapshot:
        stat = path.stat()
        return cls(
            byte_size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            ctime_ns=stat.st_ctime_ns,
        )


def redact_capability_url(url: str) -> str:
    """Redact query string, fragment, and credentials from a presigned capability URL."""
    try:
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.netloc:
            return "[REDACTED_CAPABILITY_URL]"
        host = parsed.hostname or "unknown-host"
        if parsed.port:
            host = f"{host}:{parsed.port}"
        redacted_query = "[REDACTED_SIGNATURE]" if parsed.query else ""
        return urlunsplit((parsed.scheme, host, parsed.path, redacted_query, ""))
    except Exception:
        return "[REDACTED_CAPABILITY_URL]"


def _validate_https_capability_url(url: str, *, label: str) -> SplitResult:
    """Ensure a capability URL uses https://, has a host, and contains no embedded credentials."""
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise CanonicalArtifactTransportError(
            f"Capability {label} must use https:// scheme, got {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise CanonicalArtifactTransportError(f"Capability {label} is missing hostname")
    if parsed.username or parsed.password:
        raise CanonicalArtifactTransportError(
            f"Capability {label} must not contain embedded user credentials"
        )
    return parsed


def _parse_capability_expires_at(expires_at_str: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp string from capability.expiresAt."""
    normalized = expires_at_str.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise CanonicalArtifactCapabilityExpiredError(
            f"Invalid capability expiresAt timestamp format: {expires_at_str!r}"
        ) from exc
    if dt.tzinfo is None:
        raise CanonicalArtifactCapabilityExpiredError(
            f"Capability expiresAt timestamp must be timezone-aware UTC: {expires_at_str!r}"
        )
    return dt.astimezone(timezone.utc)


def _lookup_header_case_insensitive(headers: dict[str, str], target_name: str) -> str | None:
    target_lower = target_name.lower()
    for key, value in headers.items():
        if key.lower() == target_lower:
            return value
    return None


@runtime_checkable
class CanonicalPublicationTransport(Protocol):
    """Protocol for capability-based conditional PUT and streaming GET verification."""

    def put_conditional_stream(
        self,
        put_url: str,
        headers: dict[str, str],
        stream: BinaryIO,
        *,
        content_length: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES,
    ) -> PutDisposition:
        """Stream local artifact bytes via HTTP PUT with mandatory conditional headers."""
        ...

    def get_verify_stream(
        self,
        get_url: str,
        headers: dict[str, str],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES,
    ) -> RemoteStreamVerification:
        """Stream remote artifact bytes via HTTP GET and compute SHA-256 and exact byte count."""
        ...


class HttpsCapabilityTransport:
    """Production stdlib HTTPS transport for presigned capability URLs.

    Security & Integrity guarantees:
    - Strictly enforces https:// scheme and TLS certificate verification.
    - Never follows HTTP redirects (301/302/303/307/308 raise CanonicalArtifactTransportError).
    - Streams PUT request body in bounded chunks (DEFAULT_CHUNK_SIZE_BYTES) with explicit Content-Length.
    - Streams GET response body in bounded chunks to compute SHA-256 and exact byte size.
    - Never logs or includes raw presigned URLs in exception messages.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._ssl_context = ssl_context or ssl.create_default_context()

    def put_conditional_stream(
        self,
        put_url: str,
        headers: dict[str, str],
        stream: BinaryIO,
        *,
        content_length: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES,
    ) -> PutDisposition:
        parsed = _validate_https_capability_url(put_url, label="putUrl")
        redacted_url = redact_capability_url(put_url)

        if_none_match = _lookup_header_case_insensitive(headers, "If-None-Match")
        if if_none_match != "*":
            raise CanonicalArtifactPublicationError(
                "Conditional PUT requires 'If-None-Match: *' in requiredPutHeaders"
            )

        target_path = parsed.path or "/"
        if parsed.query:
            target_path = f"{target_path}?{parsed.query}"

        conn = http.client.HTTPSConnection(
            host=parsed.hostname or "",
            port=parsed.port or 443,
            timeout=self._timeout_seconds,
            context=self._ssl_context,
        )
        try:
            conn.putrequest("PUT", target_path)
            has_content_length = False
            for k, v in headers.items():
                if k.lower() == "content-length":
                    has_content_length = True
                conn.putheader(k, v)
            if not has_content_length:
                conn.putheader("Content-Length", str(content_length))
            conn.endheaders()

            bytes_sent = 0
            while chunk := stream.read(chunk_size):
                conn.send(chunk)
                bytes_sent += len(chunk)

            if bytes_sent != content_length:
                raise CanonicalArtifactIntegrityError(
                    f"Local artifact byte count changed during PUT stream: expected {content_length}, sent {bytes_sent}"
                )

            response = conn.getresponse()
            status = response.status
            # Drain small response body to keep connection clean
            response.read(4096)

            if status in (301, 302, 303, 307, 308):
                raise CanonicalArtifactTransportError(
                    f"Redirect responses ({status}) are strictly prohibited for capability PUT ({redacted_url})",
                    status_code=status,
                )
            if status in (200, 201, 204):
                return PutDisposition.CREATED
            if status == 412:
                return PutDisposition.PRECONDITION_FAILED
            raise CanonicalArtifactTransportError(
                f"Capability PUT failed with HTTP {status} for {redacted_url}",
                status_code=status,
            )
        except CanonicalArtifactPublicationError:
            raise
        except Exception as exc:
            raise CanonicalArtifactTransportError(
                f"Capability PUT transport error for {redacted_url}: {type(exc).__name__}"
            ) from exc
        finally:
            conn.close()

    def get_verify_stream(
        self,
        get_url: str,
        headers: dict[str, str],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES,
    ) -> RemoteStreamVerification:
        parsed = _validate_https_capability_url(get_url, label="getUrl")
        redacted_url = redact_capability_url(get_url)

        target_path = parsed.path or "/"
        if parsed.query:
            target_path = f"{target_path}?{parsed.query}"

        conn = http.client.HTTPSConnection(
            host=parsed.hostname or "",
            port=parsed.port or 443,
            timeout=self._timeout_seconds,
            context=self._ssl_context,
        )
        try:
            conn.putrequest("GET", target_path)
            for k, v in headers.items():
                conn.putheader(k, v)
            conn.endheaders()

            response = conn.getresponse()
            status = response.status

            if status in (301, 302, 303, 307, 308):
                response.read(4096)
                raise CanonicalArtifactTransportError(
                    f"Redirect responses ({status}) are strictly prohibited for capability GET ({redacted_url})",
                    status_code=status,
                )
            if status != 200:
                response.read(4096)
                raise CanonicalArtifactTransportError(
                    f"Capability GET verification failed with HTTP {status} for {redacted_url}",
                    status_code=status,
                )

            hasher = hashlib.sha256()
            total_bytes = 0
            while chunk := response.read(chunk_size):
                hasher.update(chunk)
                total_bytes += len(chunk)

            return RemoteStreamVerification(
                sha256=hasher.hexdigest().lower(),
                byte_size=total_bytes,
            )
        except CanonicalArtifactPublicationError:
            raise
        except Exception as exc:
            raise CanonicalArtifactTransportError(
                f"Capability GET transport error for {redacted_url}: {type(exc).__name__}"
            ) from exc
        finally:
            conn.close()


def validate_local_canonical_h5ad_file(local_artifact_path: Path | str) -> Path:
    """Validate that a local file is a genuine, Policy-B-compliant canonical .h5ad artifact.

    Checks:
    1. Path exists, is a regular file, and is not a symlink (via validate_trusted_local_path).
    2. File size > 0.
    3. File is uncompressed and starts with HDF5 magic bytes.
    4. File opens cleanly with anndata.read_h5ad.
    5. AnnData passes validate_canonical_anndata (Policy B CSR raw integer counts in X,
       no layers/obsm/varm/obsp/varp/raw, valid unique obs_names & var_names).
    6. Controlled uns metadata includes both 'organism' and 'gene_id_scheme'.
    """
    resolved_path = validate_trusted_local_path(local_artifact_path, allow_symlinks=False)
    stat = resolved_path.stat()
    if stat.st_size <= 0:
        raise CanonicalValidationError(f"Canonical artifact file is empty: {resolved_path}")

    with open(resolved_path, "rb") as f:
        header = f.read(max(len(HDF5_MAGIC), 16))

    if detect_compression(header) != CompressionFormat.NONE:
        raise CanonicalValidationError(
            "Canonical .h5ad artifact must not be externally compressed (e.g. gzip)"
        )

    if len(header) < len(HDF5_MAGIC) or header[: len(HDF5_MAGIC)] != HDF5_MAGIC:
        raise CanonicalValidationError(
            "Canonical artifact file does not have valid HDF5 magic bytes"
        )

    try:
        adata = ad.read_h5ad(resolved_path)
    except Exception as exc:
        raise CanonicalValidationError(
            f"Failed to read canonical .h5ad file with AnnData: {exc}"
        ) from exc

    validate_canonical_anndata(adata)

    if "organism" not in adata.uns or not adata.uns.get("organism"):
        raise CanonicalValidationError(
            "Canonical AnnData must record confirmed organism in uns['organism']"
        )
    if "gene_id_scheme" not in adata.uns or not adata.uns.get("gene_id_scheme"):
        raise CanonicalValidationError(
            "Canonical AnnData must record confirmed gene ID scheme in uns['gene_id_scheme']"
        )

    return resolved_path


def validate_publication_capabilities(
    capabilities: CanonicalArtifactPublicationCapabilities,
    *,
    expected_dataset_version_id: str | None = None,
    now: datetime | None = None,
) -> None:
    """Validate capability contract fields, expiration, HTTPS schemes, and conditional headers."""
    if (
        expected_dataset_version_id is not None
        and capabilities.datasetVersionId != expected_dataset_version_id
    ):
        raise CanonicalArtifactPublicationError(
            f"Capability datasetVersionId ({capabilities.datasetVersionId!r}) does not match "
            f"expected datasetVersionId ({expected_dataset_version_id!r})"
        )

    expires_at = _parse_capability_expires_at(capabilities.expiresAt)
    current_time = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    if current_time >= expires_at:
        raise CanonicalArtifactCapabilityExpiredError(
            f"Canonical publication capability for datasetVersionId={capabilities.datasetVersionId!r} "
            f"expired at {capabilities.expiresAt}"
        )

    _validate_https_capability_url(capabilities.putUrl, label="putUrl")
    _validate_https_capability_url(capabilities.getUrl, label="getUrl")

    if_none_match = _lookup_header_case_insensitive(
        capabilities.requiredPutHeaders, "If-None-Match"
    )
    if if_none_match != "*":
        raise CanonicalArtifactPublicationError(
            "Capabilities requiredPutHeaders must include 'If-None-Match': '*'"
        )


def publish_canonical_artifact(
    local_artifact_path: Path | str,
    capabilities: CanonicalArtifactPublicationCapabilities,
    *,
    expected_dataset_version_id: str | None = None,
    transport: CanonicalPublicationTransport | None = None,
    now: datetime | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> CanonicalArtifactPublicationResult:
    """Publish a validated local canonical.h5ad artifact using scoped PUT/GET capabilities.

    Flow:
    1. Validate capability expiry, datasetVersionId, HTTPS scheme, and If-None-Match: "*" header.
    2. Validate local canonical.h5ad file (regular file, no symlinks, HDF5 magic, Policy B AnnData).
    3. Snapshot local file metadata and compute streaming SHA-256 and exact byteSize.
    4. Stream conditional PUT upload using capabilities.putUrl and capabilities.requiredPutHeaders.
       - 200/201/204 -> candidate status PUBLISHED
       - 412 Precondition Failed -> candidate status ALREADY_PRESENT
    5. Stream GET verification using capabilities.getUrl and capabilities.requiredGetHeaders,
       computing remote SHA-256 and remote byteSize directly from the downloaded byte stream.
    6. Reconcile remote byteSize and SHA-256 against local byteSize and SHA-256; raise
       CanonicalArtifactIntegrityError on any mismatch (never overwrite, never retry PUT).
    7. Re-verify local file metadata snapshot and SHA-256 to guarantee the local file was not
       mutated during upload or verification.
    8. Return CanonicalArtifactPublicationResult.
    """
    validate_publication_capabilities(
        capabilities,
        expected_dataset_version_id=expected_dataset_version_id,
        now=now,
    )

    resolved_path = validate_local_canonical_h5ad_file(local_artifact_path)

    initial_snapshot = LocalFileStatSnapshot.capture(resolved_path)
    local_sha256 = compute_file_sha256(resolved_path, chunk_size=chunk_size).lower()
    local_byte_size = initial_snapshot.byte_size

    active_transport: CanonicalPublicationTransport = (
        transport if transport is not None else HttpsCapabilityTransport()
    )

    with open(resolved_path, "rb") as file_stream:
        put_disposition = active_transport.put_conditional_stream(
            capabilities.putUrl,
            dict(capabilities.requiredPutHeaders),
            file_stream,
            content_length=local_byte_size,
            chunk_size=chunk_size,
        )

    post_put_snapshot = LocalFileStatSnapshot.capture(resolved_path)
    if post_put_snapshot != initial_snapshot:
        raise CanonicalArtifactIntegrityError(
            "Local canonical artifact file metadata changed during conditional PUT upload"
        )

    remote_verification = active_transport.get_verify_stream(
        capabilities.getUrl,
        dict(capabilities.requiredGetHeaders),
        chunk_size=chunk_size,
    )

    if remote_verification.byte_size != local_byte_size:
        raise CanonicalArtifactIntegrityError(
            f"Remote canonical artifact byteSize ({remote_verification.byte_size}) does not match "
            f"local byteSize ({local_byte_size}) after {put_disposition.value}"
        )

    if remote_verification.sha256.lower() != local_sha256:
        raise CanonicalArtifactIntegrityError(
            f"Remote canonical artifact SHA-256 ({remote_verification.sha256}) does not match "
            f"local SHA-256 ({local_sha256}) after {put_disposition.value}"
        )

    final_snapshot = LocalFileStatSnapshot.capture(resolved_path)
    final_local_sha256 = compute_file_sha256(resolved_path, chunk_size=chunk_size).lower()
    if final_snapshot != initial_snapshot or final_local_sha256 != local_sha256:
        raise CanonicalArtifactIntegrityError(
            "Local canonical artifact file was mutated during publication verification"
        )

    publication_status = (
        CanonicalArtifactPublicationStatus.PUBLISHED
        if put_disposition == PutDisposition.CREATED
        else CanonicalArtifactPublicationStatus.ALREADY_PRESENT
    )

    return CanonicalArtifactPublicationResult(
        datasetVersionId=capabilities.datasetVersionId,
        canonicalChecksum=local_sha256,
        byteSize=local_byte_size,
        status=publication_status,
    )


__all__ = [
    "DEFAULT_HTTP_TIMEOUT_SECONDS",
    "CanonicalArtifactPublicationError",
    "CanonicalArtifactCapabilityExpiredError",
    "CanonicalArtifactIntegrityError",
    "CanonicalArtifactTransportError",
    "PutDisposition",
    "RemoteStreamVerification",
    "LocalFileStatSnapshot",
    "CanonicalPublicationTransport",
    "HttpsCapabilityTransport",
    "redact_capability_url",
    "validate_local_canonical_h5ad_file",
    "validate_publication_capabilities",
    "publish_canonical_artifact",
]
