"""Unit tests for Phase 3 Step H.2 science-side canonical artifact publication."""

from __future__ import annotations

import ast
import gzip
import hashlib
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellweave_science.contracts import (
    CanonicalArtifactPublicationCapabilities,
    CanonicalArtifactPublicationStatus,
    GeneIdScheme,
    Organism,
)
from cellweave_science.importers import (
    CanonicalValidationError,
    CountValidationError,
    ImporterSecurityError,
    build_canonical_anndata,
    compute_file_sha256,
)
from cellweave_science.publication import (
    CanonicalArtifactCapabilityExpiredError,
    CanonicalArtifactIntegrityError,
    CanonicalArtifactPublicationError,
    CanonicalArtifactTransportError,
    HttpsCapabilityTransport,
    PutDisposition,
    RemoteStreamVerification,
    publish_canonical_artifact,
    redact_capability_url,
)


def _create_valid_canonical_h5ad(path: Path, *, seed_offset: int = 0) -> Path:
    """Create a valid Policy-B canonical .h5ad file on disk."""
    counts = sp.csr_matrix(
        np.array(
            [
                [1 + seed_offset, 0, 5],
                [0, 3 + seed_offset, 2],
            ],
            dtype=np.int64,
        )
    )
    obs = pd.DataFrame(
        {"sample_id": ["S1", "S1"]},
        index=pd.Index(["cell_1", "cell_2"], name="cell_id"),
    )
    var = pd.DataFrame(
        {"gene_symbol": ["TP53", "EGFR", "MYC"]},
        index=pd.Index(["ENSG00000141510", "ENSG00000146648", "ENSG00000136997"], name="gene_id"),
    )
    adata = build_canonical_anndata(
        counts,
        obs=obs,
        var=var,
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        reference_build="GRCh38",
    )
    adata.write_h5ad(path)
    return path


def _make_capabilities(
    *,
    dataset_version_id: str = "dsv_01h2x3y4z5a6b7c8d9e0f1g2h3",
    put_url: str = "https://cellweave-dev.account123.r2.cloudflarestorage.com/org/org_1/project/prj_1/datasets/dsv_01h2x3y4z5a6b7c8d9e0f1g2h3/canonical/canonical.h5ad?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-SignedHeaders=content-type%3Bhost%3Bif-none-match&X-Amz-Signature=supersecretputsig12345",
    get_url: str = "https://cellweave-dev.account123.r2.cloudflarestorage.com/org/org_1/project/prj_1/datasets/dsv_01h2x3y4z5a6b7c8d9e0f1g2h3/canonical/canonical.h5ad?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-SignedHeaders=host&X-Amz-Signature=supersecretgetsig67890",
    required_put_headers: dict[str, str] | None = None,
    required_get_headers: dict[str, str] | None = None,
    expires_at: str | None = None,
) -> CanonicalArtifactPublicationCapabilities:
    if expires_at is None:
        expires_at = (
            (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
        )
    if required_put_headers is None:
        required_put_headers = {
            "If-None-Match": "*",
            "Content-Type": "application/x-hdf5",
        }
    if required_get_headers is None:
        required_get_headers = {}

    return CanonicalArtifactPublicationCapabilities(
        datasetVersionId=dataset_version_id,
        putUrl=put_url,
        getUrl=get_url,
        requiredPutHeaders=required_put_headers,
        requiredGetHeaders=required_get_headers,
        expiresAt=expires_at,
    )


@dataclass
class InMemoryCapabilityStorageTransport:
    """Thread-safe simulated R2 capability transport supporting conditional PUT and streaming GET."""

    stored_bytes: bytes | None = None
    corrupt_after_put_bytes: bytes | None = None
    misleading_etag: str = '"00000000000000000000000000000000-99"'
    put_calls: int = 0
    get_calls: int = 0
    last_put_headers: dict[str, str] = field(default_factory=dict)
    last_get_headers: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def put_conditional_stream(
        self,
        put_url: str,
        headers: dict[str, str],
        stream: BinaryIO,
        *,
        content_length: int,
        chunk_size: int = 65536,
    ) -> PutDisposition:
        assert put_url.startswith("https://")
        if headers.get("If-None-Match") != "*":
            raise CanonicalArtifactPublicationError("Missing If-None-Match: *")

        chunks: list[bytes] = []
        while chunk := stream.read(chunk_size):
            chunks.append(chunk)
        uploaded = b"".join(chunks)
        assert len(uploaded) == content_length

        with self._lock:
            self.put_calls += 1
            self.last_put_headers = dict(headers)
            if self.stored_bytes is not None:
                return PutDisposition.PRECONDITION_FAILED
            if self.corrupt_after_put_bytes is not None:
                self.stored_bytes = self.corrupt_after_put_bytes
            else:
                self.stored_bytes = uploaded
            return PutDisposition.CREATED

    def get_verify_stream(
        self,
        get_url: str,
        headers: dict[str, str],
        *,
        chunk_size: int = 65536,
    ) -> RemoteStreamVerification:
        assert get_url.startswith("https://")
        with self._lock:
            self.get_calls += 1
            self.last_get_headers = dict(headers)
            if self.stored_bytes is None:
                raise CanonicalArtifactTransportError(
                    "Remote object not found (404)", status_code=404
                )
            payload = self.stored_bytes

        # Stream in chunks and compute SHA-256 directly from bytes, ignoring misleading_etag
        hasher = hashlib.sha256()
        total = 0
        bio = io.BytesIO(payload)
        while chunk := bio.read(chunk_size):
            hasher.update(chunk)
            total += len(chunk)

        return RemoteStreamVerification(
            sha256=hasher.hexdigest().lower(),
            byte_size=total,
        )


class TestFirstPublicationAndIdempotency:
    """Tests 1-5: First publication, idempotent republish, conflict, corruption, misleading ETag."""

    def test_first_publication_succeeds_and_verifies_stream(self, tmp_path: Path) -> None:
        h5ad_path = _create_valid_canonical_h5ad(tmp_path / "canonical.h5ad")
        expected_sha = compute_file_sha256(h5ad_path)
        expected_size = h5ad_path.stat().st_size

        transport = InMemoryCapabilityStorageTransport()
        caps = _make_capabilities()

        result = publish_canonical_artifact(
            h5ad_path,
            caps,
            expected_dataset_version_id=caps.datasetVersionId,
            transport=transport,
        )

        assert result.status == CanonicalArtifactPublicationStatus.PUBLISHED
        assert result.datasetVersionId == caps.datasetVersionId
        assert result.canonicalChecksum == expected_sha
        assert result.byteSize == expected_size
        assert transport.put_calls == 1
        assert transport.get_calls == 1
        assert transport.last_put_headers["If-None-Match"] == "*"
        assert transport.last_put_headers["Content-Type"] == "application/x-hdf5"

    def test_idempotent_republish_same_bytes_returns_already_present(self, tmp_path: Path) -> None:
        h5ad_path = _create_valid_canonical_h5ad(tmp_path / "canonical.h5ad")
        expected_sha = compute_file_sha256(h5ad_path)
        expected_size = h5ad_path.stat().st_size

        # Pre-populate remote storage with identical bytes
        transport = InMemoryCapabilityStorageTransport(stored_bytes=h5ad_path.read_bytes())
        caps = _make_capabilities()

        result = publish_canonical_artifact(
            h5ad_path,
            caps,
            expected_dataset_version_id=caps.datasetVersionId,
            transport=transport,
        )

        assert result.status == CanonicalArtifactPublicationStatus.ALREADY_PRESENT
        assert result.datasetVersionId == caps.datasetVersionId
        assert result.canonicalChecksum == expected_sha
        assert result.byteSize == expected_size
        assert transport.put_calls == 1
        assert transport.get_calls == 1

    def test_republish_conflict_with_different_remote_bytes_raises_integrity_error(
        self, tmp_path: Path
    ) -> None:
        local_path = _create_valid_canonical_h5ad(tmp_path / "local.h5ad", seed_offset=0)
        other_path = _create_valid_canonical_h5ad(tmp_path / "other.h5ad", seed_offset=10)
        original_remote_bytes = other_path.read_bytes()

        transport = InMemoryCapabilityStorageTransport(stored_bytes=original_remote_bytes)
        caps = _make_capabilities()

        with pytest.raises(CanonicalArtifactIntegrityError, match="does not match"):
            publish_canonical_artifact(
                local_path,
                caps,
                expected_dataset_version_id=caps.datasetVersionId,
                transport=transport,
            )

        # Remote object must remain untouched (no overwrite, no second PUT)
        assert transport.put_calls == 1
        assert transport.get_calls == 1
        assert transport.stored_bytes == original_remote_bytes

    def test_post_upload_remote_corruption_raises_integrity_error(self, tmp_path: Path) -> None:
        local_path = _create_valid_canonical_h5ad(tmp_path / "canonical.h5ad")
        raw_bytes = bytearray(local_path.read_bytes())
        raw_bytes[-1] ^= 0xFF  # Flip final byte while keeping identical byteSize

        transport = InMemoryCapabilityStorageTransport(corrupt_after_put_bytes=bytes(raw_bytes))
        caps = _make_capabilities()

        with pytest.raises(CanonicalArtifactIntegrityError, match="SHA-256"):
            publish_canonical_artifact(
                local_path,
                caps,
                expected_dataset_version_id=caps.datasetVersionId,
                transport=transport,
            )

        assert transport.put_calls == 1
        assert transport.get_calls == 1

    def test_misleading_etag_regression_never_trusted_over_stream_sha256(
        self, tmp_path: Path
    ) -> None:
        local_path = _create_valid_canonical_h5ad(tmp_path / "canonical.h5ad")
        expected_sha = compute_file_sha256(local_path)

        # Even when ETag is multipart/unrelated, matching stream bytes succeed
        transport_ok = InMemoryCapabilityStorageTransport(
            misleading_etag='"multipart-unrelated-etag-99999-12"'
        )
        res = publish_canonical_artifact(local_path, _make_capabilities(), transport=transport_ok)
        assert res.canonicalChecksum == expected_sha

        # And if ETag pretends to match local_sha256 while actual bytes differ, verification MUST fail
        corrupted = bytearray(local_path.read_bytes())
        corrupted[-2] ^= 0x55
        transport_bad = InMemoryCapabilityStorageTransport(
            corrupt_after_put_bytes=bytes(corrupted),
            misleading_etag=f'"{expected_sha}"',
        )
        with pytest.raises(CanonicalArtifactIntegrityError, match="SHA-256"):
            publish_canonical_artifact(local_path, _make_capabilities(), transport=transport_bad)


class TestLocalArtifactValidation:
    """Test 6 (Section 36): Invalid local artifact tests before any upload occurs."""

    def test_rejects_missing_file_before_upload(self, tmp_path: Path) -> None:
        transport = InMemoryCapabilityStorageTransport()
        with pytest.raises(ImporterSecurityError, match="does not exist"):
            publish_canonical_artifact(
                tmp_path / "nonexistent.h5ad", _make_capabilities(), transport=transport
            )
        assert transport.put_calls == 0

    def test_rejects_empty_file_before_upload(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.h5ad"
        empty_file.write_bytes(b"")
        transport = InMemoryCapabilityStorageTransport()
        with pytest.raises(CanonicalValidationError, match="empty"):
            publish_canonical_artifact(empty_file, _make_capabilities(), transport=transport)
        assert transport.put_calls == 0

    def test_rejects_plain_text_file_before_upload(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "fake.h5ad"
        txt_file.write_text("cell_id,ENSG00001\ncell_1,5\n", encoding="utf-8")
        transport = InMemoryCapabilityStorageTransport()
        with pytest.raises(CanonicalValidationError, match="HDF5 magic bytes"):
            publish_canonical_artifact(txt_file, _make_capabilities(), transport=transport)
        assert transport.put_calls == 0

    def test_rejects_gzip_compressed_file_before_upload(self, tmp_path: Path) -> None:
        valid_h5ad = _create_valid_canonical_h5ad(tmp_path / "valid.h5ad")
        gz_path = tmp_path / "compressed.h5ad"
        with gzip.open(gz_path, "wb") as gz:
            gz.write(valid_h5ad.read_bytes())
        transport = InMemoryCapabilityStorageTransport()
        with pytest.raises(CanonicalValidationError, match="compressed"):
            publish_canonical_artifact(gz_path, _make_capabilities(), transport=transport)
        assert transport.put_calls == 0

    def test_rejects_truncated_corrupted_hdf5_before_upload(self, tmp_path: Path) -> None:
        bad_h5 = tmp_path / "corrupt.h5ad"
        bad_h5.write_bytes(b"\x89HDF\r\n\x1a\n" + b"corrupted_hdf5_payload")
        transport = InMemoryCapabilityStorageTransport()
        with pytest.raises(CanonicalValidationError, match="Failed to read canonical .h5ad"):
            publish_canonical_artifact(bad_h5, _make_capabilities(), transport=transport)
        assert transport.put_calls == 0

    def test_rejects_anndata_violating_policy_b_counts_layer_or_forbidden_slots(
        self, tmp_path: Path
    ) -> None:
        valid_path = _create_valid_canonical_h5ad(tmp_path / "base.h5ad")

        # Case A: layers["counts"] present
        adata_layers = ad.read_h5ad(valid_path)
        adata_layers.layers["counts"] = adata_layers.X.copy()
        layers_path = tmp_path / "with_counts_layer.h5ad"
        adata_layers.write_h5ad(layers_path)

        transport = InMemoryCapabilityStorageTransport()
        with pytest.raises(CanonicalValidationError, match="layers"):
            publish_canonical_artifact(layers_path, _make_capabilities(), transport=transport)
        assert transport.put_calls == 0

        # Case B: obsm["X_umap"] present
        adata_obsm = ad.read_h5ad(valid_path)
        adata_obsm.obsm["X_umap"] = np.zeros((adata_obsm.n_obs, 2), dtype=np.float32)
        obsm_path = tmp_path / "with_obsm.h5ad"
        adata_obsm.write_h5ad(obsm_path)

        with pytest.raises(CanonicalValidationError, match="obsm"):
            publish_canonical_artifact(obsm_path, _make_capabilities(), transport=transport)
        assert transport.put_calls == 0

    def test_rejects_anndata_with_non_count_float_matrix(self, tmp_path: Path) -> None:
        valid_path = _create_valid_canonical_h5ad(tmp_path / "base.h5ad")
        adata = ad.read_h5ad(valid_path)
        adata.X = sp.csr_matrix(np.array([[1.25, 0.0, 3.5], [0.0, 2.75, 1.1]], dtype=np.float32))
        non_count_path = tmp_path / "non_count.h5ad"
        adata.write_h5ad(non_count_path)

        transport = InMemoryCapabilityStorageTransport()
        with pytest.raises((CanonicalValidationError, CountValidationError)):
            publish_canonical_artifact(non_count_path, _make_capabilities(), transport=transport)
        assert transport.put_calls == 0

    def test_rejects_symlink_path_before_upload(self, tmp_path: Path) -> None:
        valid_path = _create_valid_canonical_h5ad(tmp_path / "real.h5ad")
        symlink_path = tmp_path / "link.h5ad"
        try:
            os.symlink(valid_path, symlink_path)
        except OSError:
            # On Windows without developer mode privileges, simulate is_symlink=True
            with patch.object(Path, "is_symlink", return_value=True):
                transport = InMemoryCapabilityStorageTransport()
                with pytest.raises(ImporterSecurityError, match="Symbolic links"):
                    publish_canonical_artifact(
                        valid_path, _make_capabilities(), transport=transport
                    )
                assert transport.put_calls == 0
            return

        transport = InMemoryCapabilityStorageTransport()
        with pytest.raises(ImporterSecurityError, match="Symbolic links"):
            publish_canonical_artifact(symlink_path, _make_capabilities(), transport=transport)
        assert transport.put_calls == 0


class TestCapabilityValidationAndSecurity:
    """Tests 7, 8, 10, 11: Capability expiry, If-None-Match, redirects, local mutation, redaction."""

    def test_rejects_expired_capability_before_upload(self, tmp_path: Path) -> None:
        h5ad_path = _create_valid_canonical_h5ad(tmp_path / "canonical.h5ad")
        expired = (
            (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
        )
        caps = _make_capabilities(expires_at=expired)
        transport = InMemoryCapabilityStorageTransport()

        with pytest.raises(CanonicalArtifactCapabilityExpiredError, match="expired"):
            publish_canonical_artifact(h5ad_path, caps, transport=transport)
        assert transport.put_calls == 0

    def test_rejects_dataset_version_id_mismatch_before_upload(self, tmp_path: Path) -> None:
        h5ad_path = _create_valid_canonical_h5ad(tmp_path / "canonical.h5ad")
        caps = _make_capabilities(dataset_version_id="dsv_actual_111")
        transport = InMemoryCapabilityStorageTransport()

        with pytest.raises(CanonicalArtifactPublicationError, match="does not match"):
            publish_canonical_artifact(
                h5ad_path,
                caps,
                expected_dataset_version_id="dsv_expected_222",
                transport=transport,
            )
        assert transport.put_calls == 0

    def test_rejects_missing_if_none_match_header_in_transport(self) -> None:
        http_transport = HttpsCapabilityTransport()
        with pytest.raises(CanonicalArtifactPublicationError, match="If-None-Match"):
            http_transport.put_conditional_stream(
                "https://cellweave-dev.account123.r2.cloudflarestorage.com/obj",
                {"Content-Type": "application/x-hdf5"},
                io.BytesIO(b"test"),
                content_length=4,
            )

    def test_rejects_non_https_capability_urls(self, tmp_path: Path) -> None:
        h5ad_path = _create_valid_canonical_h5ad(tmp_path / "canonical.h5ad")
        caps = _make_capabilities(
            put_url="http://cellweave-dev.account123.r2.cloudflarestorage.com/obj?X-Amz-Signature=secret"
        )
        transport = InMemoryCapabilityStorageTransport()

        with pytest.raises(CanonicalArtifactTransportError, match="https://"):
            publish_canonical_artifact(h5ad_path, caps, transport=transport)
        assert transport.put_calls == 0

    def test_https_transport_rejects_redirects_and_redacts_presigned_urls(self) -> None:
        secret_put_url = (
            "https://cellweave-dev.account123.r2.cloudflarestorage.com/org/o1/project/p1/"
            "datasets/d1/canonical/canonical.h5ad?X-Amz-Signature=TOP_SECRET_PUT_SIG_999"
        )
        redacted = redact_capability_url(secret_put_url)
        assert "TOP_SECRET_PUT_SIG_999" not in redacted
        assert "[REDACTED_SIGNATURE]" in redacted

        class FakeResponse:
            status = 307

            def read(self, _amt: int | None = None) -> bytes:
                return b""

        class FakeHTTPSConnection:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def putrequest(self, *args: object, **kwargs: object) -> None:
                pass

            def putheader(self, *args: object, **kwargs: object) -> None:
                pass

            def endheaders(self) -> None:
                pass

            def send(self, _data: bytes) -> None:
                pass

            def getresponse(self) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with patch("http.client.HTTPSConnection", FakeHTTPSConnection):
            transport = HttpsCapabilityTransport()
            with pytest.raises(CanonicalArtifactTransportError) as exc_info:
                transport.put_conditional_stream(
                    secret_put_url,
                    {"If-None-Match": "*", "Content-Type": "application/x-hdf5"},
                    io.BytesIO(b"1234"),
                    content_length=4,
                )
            assert exc_info.value.status_code == 307
            assert "TOP_SECRET_PUT_SIG_999" not in str(exc_info.value)
            assert "[REDACTED_SIGNATURE]" in str(exc_info.value)

    def test_detects_local_file_mutation_during_upload(self, tmp_path: Path) -> None:
        h5ad_path = _create_valid_canonical_h5ad(tmp_path / "canonical.h5ad")
        original_bytes = h5ad_path.read_bytes()

        class MutatingTransport(InMemoryCapabilityStorageTransport):
            def put_conditional_stream(
                self,
                put_url: str,
                headers: dict[str, str],
                stream: BinaryIO,
                *,
                content_length: int,
                chunk_size: int = 65536,
            ) -> PutDisposition:
                res = super().put_conditional_stream(
                    put_url, headers, stream, content_length=content_length, chunk_size=chunk_size
                )
                # Mutate local file on disk immediately after PUT
                mutated = bytearray(original_bytes)
                mutated[-1] ^= 0xFF
                h5ad_path.write_bytes(bytes(mutated))
                return res

        transport = MutatingTransport()
        with pytest.raises(CanonicalArtifactIntegrityError, match="Local canonical artifact file"):
            publish_canonical_artifact(h5ad_path, _make_capabilities(), transport=transport)


class TestConcurrentPublicationSimulation:
    """Test 9 (Section 39): Two concurrent workers publishing to the same capability target."""

    def test_concurrent_same_bytes_one_published_one_already_present(self, tmp_path: Path) -> None:
        h5ad_path = _create_valid_canonical_h5ad(tmp_path / "canonical.h5ad")
        transport = InMemoryCapabilityStorageTransport()
        caps = _make_capabilities()

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(publish_canonical_artifact, h5ad_path, caps, transport=transport)
            f2 = pool.submit(publish_canonical_artifact, h5ad_path, caps, transport=transport)
            r1 = f1.result()
            r2 = f2.result()

        statuses = {r1.status, r2.status}
        assert statuses == {
            CanonicalArtifactPublicationStatus.PUBLISHED,
            CanonicalArtifactPublicationStatus.ALREADY_PRESENT,
        }
        assert r1.canonicalChecksum == r2.canonicalChecksum

    def test_concurrent_different_bytes_winner_published_loser_raises_integrity_error(
        self, tmp_path: Path
    ) -> None:
        path_a = _create_valid_canonical_h5ad(tmp_path / "a.h5ad", seed_offset=0)
        path_b = _create_valid_canonical_h5ad(tmp_path / "b.h5ad", seed_offset=25)
        transport = InMemoryCapabilityStorageTransport()
        caps = _make_capabilities()

        # Worker A wins first, Worker B arrives right after
        res_a = publish_canonical_artifact(path_a, caps, transport=transport)
        assert res_a.status == CanonicalArtifactPublicationStatus.PUBLISHED

        with pytest.raises(CanonicalArtifactIntegrityError, match="does not match"):
            publish_canonical_artifact(path_b, caps, transport=transport)


class TestBoundaryIsolationVerification:
    """Verify science publication module contains no R2 SDK, boto3, or object-key construction."""

    def test_publication_module_imports_no_cloud_sdk_or_tenant_key_builder(self) -> None:
        pub_file = (
            Path(__file__).resolve().parents[1]
            / "cellweave_science"
            / "importers"
            / "publication.py"
        )
        source = pub_file.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])

        forbidden_imports = {"boto3", "botocore", "s3fs", "aiobotocore", "minio"}
        assert imported_modules.isdisjoint(forbidden_imports)
        assert "organizationId" not in source.replace(
            "No organizationId / projectId tenant-context storage authority", ""
        )
        assert "R2_SECRET_ACCESS_KEY" not in source
        assert "R2_ACCESS_KEY_ID" not in source
