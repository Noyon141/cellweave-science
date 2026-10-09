"""Test Pydantic contract parsing and schema validation."""

from __future__ import annotations

from typing import Any, Dict
import pytest
from pydantic import ValidationError

from cellweave_science.contracts.models import (
    ArtifactManifest,
    ArtifactStorageRef,
    ArtifactType,
    ComputeJobSubmitRequest,
    DomainError,
    ResourceClass,
    RunManifest,
    ScientificJobPayload,
    ScientificJobType,
)


@pytest.mark.unit
def test_scientific_job_payload_valid(sample_job_payload_dict: Dict[str, Any]) -> None:
    """Verify that a canonical job payload parses cleanly into Pydantic models."""
    payload = ScientificJobPayload.model_validate(sample_job_payload_dict)
    assert payload.jobId == "job-smoke-001"
    assert payload.jobType == ScientificJobType.RUN_QC
    assert payload.inputs.primaryDataset.format == "h5ad"
    assert payload.manifestFragment.randomSeeds["default"] == 42


@pytest.mark.unit
def test_payload_invalid_sha256_rejected(sample_job_payload_dict: Dict[str, Any]) -> None:
    """Verify that invalid sha256 hashes are strictly rejected."""
    bad_dict = dict(sample_job_payload_dict)
    bad_dict["inputs"] = {
        "primaryDataset": {
            "downloadUrl": "https://storage.cellweave.internal/test.h5ad",
            "sha256": "invalid-non-hex-hash",
        }
    }
    with pytest.raises(ValidationError):
        ScientificJobPayload.model_validate(bad_dict)


@pytest.mark.unit
def test_contract_models_are_frozen(sample_job_payload_dict: Dict[str, Any]) -> None:
    """Ensure contract models are immutable (frozen=True) to prevent accidental mutation."""
    payload = ScientificJobPayload.model_validate(sample_job_payload_dict)
    with pytest.raises(ValidationError):
        payload.jobId = "mutated-id"  # type: ignore[misc]


@pytest.mark.unit
def test_domain_error_schema() -> None:
    """Verify DomainError creation and serialization."""
    err = DomainError(
        code="INVALID_DATASET_DIMENSIONS",
        message="Dataset contains 0 cells",
        technicalDetail="obs dimension is 0",
        recommendedAction="Upload a dataset containing valid cell observations",
        correlationId="req-trace-001",
        retryable=False,
    )
    dumped = err.model_dump()
    assert dumped["code"] == "INVALID_DATASET_DIMENSIONS"
    assert dumped["retryable"] is False


@pytest.mark.unit
def test_artifact_manifest_schema() -> None:
    """Verify ArtifactManifest creation and field enforcement."""
    manifest = ArtifactManifest(
        schemaVersion="1.0.0",
        artifactId="art-umap-001",
        runId="run-001",
        type=ArtifactType.EMBEDDING,
        storage=ArtifactStorageRef(
            provider="R2",
            bucket="cellweave-artifacts",
            objectKey="org/proj/runs/run-001/steps/umap/embedding.json",
        ),
        sha256="c" * 64,
        sizeBytes=1048576,
        mimeType="application/json",
        producingStepId="step-umap-001",
        lineage=[],
        createdAt="2026-10-09T12:00:00Z",
    )
    assert manifest.type == ArtifactType.EMBEDDING
    assert manifest.storage.provider == "R2"
