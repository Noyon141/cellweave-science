"""Bounded Phase 4 transport/staging for existing Phase 3 importers; no new science."""

from __future__ import annotations

import hashlib
import http.client
import ssl
import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from cellweave_contracts import ExecutionResult, PreparedExecution, SourceFormat
from cellweave_science.importers import DatasetImporter, ImportSourceFile, ImportSourceSet
from cellweave_science.importers.checksum import SourceChecksumFile, compute_source_checksum
from cellweave_science.importers.generic import GenericMatrixConfig, GenericMatrixImporter
from cellweave_science.importers.h5ad import H5ADImporter
from cellweave_science.importers.publication import (
    CanonicalPublicationTransport,
    publish_canonical_artifact,
)
from cellweave_science.importers.security import DecompressionLimits
from cellweave_science.importers.tenx_h5 import TenxH5Importer
from cellweave_science.importers.tenx_mtx import TenxMtxImporter
from cellweave_science.importers.tenx_shared import TenxImporterConfig

CHUNK_BYTES = 1024 * 1024


class ExecutionInputError(ValueError):
    """Controlled transport/integrity failure; capabilities never appear in errors."""


class InputTransport(Protocol):
    def chunks(self, url: str) -> Iterator[bytes]: ...


class HttpsInputTransport:
    def chunks(self, url: str) -> Iterator[bytes]:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ExecutionInputError("Invalid scoped input capability")
        connection = http.client.HTTPSConnection(
            parsed.hostname,
            port=parsed.port or 443,
            timeout=30,
            context=ssl.create_default_context(),
        )
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        try:
            connection.request("GET", target)
            response = connection.getresponse()
            if response.status != 200:
                raise ExecutionInputError("Scoped input download rejected")
            while chunk := response.read(CHUNK_BYTES):
                yield chunk
        except (OSError, http.client.HTTPException):
            raise ExecutionInputError("Scoped input transport unavailable") from None
        finally:
            connection.close()


def execute_prepared(
    request: PreparedExecution,
    *,
    input_transport: InputTransport | None = None,
    publication_transport: CanonicalPublicationTransport | None = None,
) -> ExecutionResult:
    envelope = request.envelope
    if datetime.fromisoformat(envelope.deadlineAt.replace("Z", "+00:00")) <= datetime.now(
        timezone.utc
    ):
        raise ExecutionInputError("Execution admission deadline exceeded")
    ceilings = envelope.ceilings
    if ceilings.estimatedComputeUnits > ceilings.maxComputeUnits:
        raise ExecutionInputError("Invalid execution ceilings")
    if request.publication.datasetVersionId != envelope.datasetVersionId:
        raise ExecutionInputError("Publication dataset mismatch")
    if sum(f.byteSize for f in request.files) > ceilings.maxInputBytes:
        raise ExecutionInputError("Input byte ceiling exceeded")
    names: set[str] = set()
    identities: set[str] = set()
    source_files: list[ImportSourceFile] = []
    facts: list[SourceChecksumFile] = []
    transport = input_transport or HttpsInputTransport()
    with tempfile.TemporaryDirectory(prefix="cellweave-execution-") as directory:
        scratch = Path(directory)
        for source in request.files:
            name = source.fileName
            if (
                not name
                or name in (".", "..")
                or name != name.strip()
                or any(c in name for c in ("/", "\\", "\x00", "\t", "\n", "\r"))
                or name in names
                or source.datasetFileId in identities
            ):
                raise ExecutionInputError("Invalid exact source membership")
            names.add(name)
            identities.add(source.datasetFileId)
            path = scratch / name
            digest = hashlib.sha256()
            size = 0
            with path.open("xb") as stream:
                for chunk in transport.chunks(source.downloadUrl):
                    if len(chunk) > CHUNK_BYTES:
                        raise ExecutionInputError("Input transport chunk ceiling exceeded")
                    size += len(chunk)
                    if size > source.byteSize:
                        raise ExecutionInputError("Input size mismatch")
                    digest.update(chunk)
                    stream.write(chunk)
            checksum = digest.hexdigest()
            if size != source.byteSize or checksum != source.sha256:
                raise ExecutionInputError("Input integrity mismatch")
            source_files.append(
                ImportSourceFile(source.datasetFileId, source.role, path, source.ordinal)
            )
            facts.append(
                SourceChecksumFile(
                    source.datasetFileId, name, size, checksum, source.role, source.ordinal
                )
            )
        if compute_source_checksum(request.sourceFormat, facts) != request.sourceChecksum:
            raise ExecutionInputError("Source manifest checksum mismatch")
        limits = DecompressionLimits(ceilings.maxDecompressedBytes, ceilings.maxCompressionRatio)
        tenx_config = TenxImporterConfig(limits)
        importer: DatasetImporter
        if request.sourceFormat == SourceFormat.H5AD:
            importer = H5ADImporter()
        elif request.sourceFormat == SourceFormat.TENX_H5:
            importer = TenxH5Importer(tenx_config)
        elif request.sourceFormat == SourceFormat.TENX_MTX:
            importer = TenxMtxImporter(tenx_config)
        else:
            importer = GenericMatrixImporter(GenericMatrixConfig(limits))
        source_set = ImportSourceSet(
            request.sourceFormat, tuple(source_files), envelope.datasetVersionId
        )
        output = scratch / "output" / "canonical.h5ad"
        # Existing importers enforce source and destination immutability and canonical policy.
        canonical = importer.canonicalize(
            source_set,
            str(output),
            envelope.organism,
            envelope.geneIdScheme,
            envelope.referenceBuild,
            envelope.mapping,
        )
        publication = publish_canonical_artifact(
            output,
            request.publication,
            expected_dataset_version_id=envelope.datasetVersionId,
            transport=publication_transport,
        )
        # Scratch paths are not persisted provenance or remote caller authority.
        canonical = canonical.model_copy(update={"localArtifactPath": None})
        return ExecutionResult(
            analysisRunId=envelope.analysisRunId,
            submissionId=envelope.submissionId,
            publication=publication,
            canonicalization=canonical,
        )
