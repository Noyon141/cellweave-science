"""Global pytest fixtures and test setup for cellweave-science."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest

from cellweave_science.adapters.local import LocalComputeAdapter
from cellweave_science.seeds.seed import DEFAULT_SEED, set_deterministic_seeds


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Generator[None, None, None]:
    """Ensure every test runs with predictable deterministic random seed."""
    set_deterministic_seeds(DEFAULT_SEED)
    yield
    set_deterministic_seeds(DEFAULT_SEED)


@pytest.fixture
def local_adapter() -> LocalComputeAdapter:
    """Fixture providing an instance of LocalComputeAdapter."""
    return LocalComputeAdapter()


@pytest.fixture
def sample_job_payload_dict() -> dict[str, Any]:
    """Fixture providing a valid ScientificJobPayload dictionary."""
    return {
        "schemaVersion": "1.0.0",
        "pipelineVersion": "1.0.0",
        "jobId": "job-smoke-001",
        "runId": "run-smoke-001",
        "stepId": "step-qc-001",
        "jobType": "run_qc",
        "inputs": {
            "primaryDataset": {
                "downloadUrl": "https://storage.cellweave.internal/datasets/test.h5ad",
                "sha256": "a" * 64,
                "format": "h5ad",
            },
            "auxiliaryInputs": [],
        },
        "outputs": {
            "targets": [
                {
                    "name": "qc_report",
                    "uploadUrl": "https://storage.cellweave.internal/artifacts/qc.json",
                    "expectedMimeType": "application/json",
                }
            ]
        },
        "parameters": {"min_genes": 200, "max_mito_percent": 0.20},
        "manifestFragment": {
            "containerDigest": "sha256:" + "b" * 64,
            "randomSeeds": {"default": 42},
            "softwareVersions": {"scanpy": "1.10.0", "anndata": "0.10.7"},
        },
        "resourceClass": "CPU_SMALL",
    }


@pytest.fixture
def golden_manifest_path() -> Path:
    """Path to the golden reference manifest file."""
    manifest = Path(__file__).resolve().parent.parent / "reference_data" / "golden_manifest.json"
    assert manifest.exists(), f"Golden manifest not found at {manifest}"
    return manifest


@pytest.fixture
def golden_manifest(golden_manifest_path: Path) -> dict[str, Any]:
    """Loaded golden reference manifest dict."""
    with open(golden_manifest_path, encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))
