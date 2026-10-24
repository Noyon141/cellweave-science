from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellweave_contracts import PreparedExecution
from cellweave_science.execution import ExecutionInputError, execute_prepared
from cellweave_science.importers.canonical import validate_canonical_anndata
from cellweave_science.importers.publication import PutDisposition, RemoteStreamVerification


class Input:
    def __init__(self, content: bytes):
        self.content = content

    def chunks(self, url: str) -> Iterator[bytes]:
        assert url.startswith("https://")
        yield self.content


class Publication:
    def __init__(self) -> None:
        self.content = b""
        self.calls = 0

    def put_conditional_stream(
        self,
        put_url: str,
        headers: dict[str, str],
        stream: BinaryIO,
        *,
        content_length: int,
        chunk_size: int = 1048576,
    ) -> PutDisposition:
        assert put_url.startswith("https://") and chunk_size > 0
        assert headers["If-None-Match"] == "*"
        self.calls += 1
        self.content = stream.read()
        assert len(self.content) == content_length
        return PutDisposition.CREATED

    def get_verify_stream(
        self, get_url: str, headers: dict[str, str], *, chunk_size: int = 1048576
    ) -> RemoteStreamVerification:
        assert get_url.startswith("https://") and chunk_size > 0
        assert headers == {}
        return RemoteStreamVerification(
            sha256=hashlib.sha256(self.content).hexdigest(), byte_size=len(self.content)
        )


def prepared(tmp_path: Path) -> tuple[PreparedExecution, bytes]:
    path = tmp_path / "source.h5ad"
    source = ad.AnnData(
        sp.csr_matrix(np.array([[1, 0], [2, 3]], dtype=np.int64)),
        obs=pd.DataFrame(index=["CELL_A-1", "CELL_B-1"]),
        var=pd.DataFrame(index=["ENSG00000141510.7", "ENSG00000146648.2"]),
    )
    source.write_h5ad(path)
    content = path.read_bytes()
    checksum = hashlib.sha256(content).hexdigest()
    deadline = (
        (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    )
    request = {
        "envelope": {
            "schemaVersion": "1.0.0",
            "environment": "development",
            "analysisRunId": "run",
            "organizationId": "org",
            "projectId": "project",
            "datasetVersionId": "version",
            "importId": "import",
            "pipelineVersion": "approved-import",
            "stepId": "step",
            "submissionId": "run:canonicalize",
            "deadlineAt": deadline,
            "organism": "HUMAN",
            "geneIdScheme": "ENSEMBL",
            "referenceBuild": None,
            "ceilings": {
                "resourceClass": "CPU_SMALL",
                "maxMemoryClass": "CPU_SMALL",
                "estimatedComputeUnits": 1,
                "maxComputeUnits": 60,
                "maxRuntimeSeconds": 60,
                "maxInputBytes": len(content),
                "maxDecompressedBytes": 10000000,
                "maxCompressionRatio": 100,
            },
        },
        "sourceFormat": "H5AD",
        "sourceChecksum": checksum,
        "files": [
            {
                "datasetFileId": "file",
                "fileName": "source.h5ad",
                "byteSize": len(content),
                "sha256": checksum,
                "role": "PRIMARY",
                "ordinal": 0,
                "downloadUrl": "https://storage.example/source?ephemeral=secret",
            }
        ],
        "publication": {
            "datasetVersionId": "version",
            "putUrl": "https://storage.example/canonical?put=secret",
            "getUrl": "https://storage.example/canonical?get=secret",
            "expiresAt": deadline,
            "requiredPutHeaders": {"If-None-Match": "*", "Content-Type": "application/x-hdf5"},
            "requiredGetHeaders": {},
        },
    }
    import json

    return PreparedExecution.model_validate_json(json.dumps(request), strict=True), content


def test_real_phase3_handoff_preserves_identity_and_validates_canonical_bytes(
    tmp_path: Path,
) -> None:
    request, content = prepared(tmp_path)
    publication = Publication()
    result = execute_prepared(
        request, input_transport=Input(content), publication_transport=publication
    )
    assert result.analysisRunId == "run"
    assert result.publication.canonicalChecksum == hashlib.sha256(publication.content).hexdigest()
    assert result.canonicalization.cellCount == 2
    assert result.canonicalization.geneCount == 2
    assert result.canonicalization.localArtifactPath is None
    artifact = tmp_path / "published.h5ad"
    artifact.write_bytes(publication.content)
    canonical = ad.read_h5ad(artifact)
    validate_canonical_anndata(canonical)
    assert canonical.obs_names.tolist() == ["CELL_A-1", "CELL_B-1"]
    assert canonical.var_names.tolist() == ["ENSG00000141510.7", "ENSG00000146648.2"]


@pytest.mark.parametrize(
    "case", ["checksum", "size", "manifest", "path", "duplicate", "ceiling", "dataset"]
)
def test_invalid_handoff_fails_before_publication(tmp_path: Path, case: str) -> None:
    request, content = prepared(tmp_path)
    data = request.model_dump(mode="json")
    if case == "checksum":
        data["files"][0]["sha256"] = "a" * 64
    elif case == "size":
        data["files"][0]["byteSize"] += 1
    elif case == "manifest":
        data["sourceChecksum"] = "a" * 64
    elif case == "path":
        data["files"][0]["fileName"] = "../source.h5ad"
    elif case == "duplicate":
        data["files"].append(data["files"][0])
    elif case == "ceiling":
        data["envelope"]["ceilings"]["maxInputBytes"] = 1
    else:
        data["publication"]["datasetVersionId"] = "foreign-version"
    import json

    invalid = PreparedExecution.model_validate_json(json.dumps(data), strict=True)
    publication = Publication()
    with pytest.raises(ExecutionInputError):
        execute_prepared(invalid, input_transport=Input(content), publication_transport=publication)
    assert publication.calls == 0
