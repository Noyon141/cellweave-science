"""Test local and Modal compute adapter boundaries and behaviors."""

from __future__ import annotations

from typing import Any, Dict
import pytest

from cellweave_science.adapters.compute import ComputeMode
from cellweave_science.adapters.local import LocalComputeAdapter
from cellweave_science.adapters.modal_adapter import ModalComputeAdapter
from cellweave_science.contracts.models import (
    ComputeJobCancelRequest,
    ComputeJobSubmitRequest,
    ResourceClass,
    ScientificJobPayload,
)


@pytest.mark.unit
def test_local_adapter_lifecycle(sample_job_payload_dict: Dict[str, Any]) -> None:
    """Verify local adapter submit, status, execute, and cancel workflow."""
    adapter = LocalComputeAdapter()
    assert adapter.mode == ComputeMode.LOCAL

    submit_req = ComputeJobSubmitRequest(
        jobId="job-test-01",
        jobType="run_qc",
        resourceClass=ResourceClass.CPU_SMALL,
        payloadUri="local://dummy",
        maxRuntimeSeconds=600,
        correlationId="corr-test-01",
    )
    prov_id = adapter.submit_job(submit_req)
    assert prov_id.startswith("local-")

    status = adapter.get_status("job-test-01", prov_id)
    assert status.status == "RUNNING"

    cancel_req = ComputeJobCancelRequest(
        jobId="job-test-01",
        providerJobId=prov_id,
        reason="User cancellation",
    )
    cancelled = adapter.cancel_job(cancel_req)
    assert cancelled is True

    status_after = adapter.get_status("job-test-01", prov_id)
    assert status_after.status == "CANCELLED"


@pytest.mark.unit
def test_local_adapter_sync_execution(sample_job_payload_dict: Dict[str, Any]) -> None:
    """Verify local adapter synchronous payload execution."""
    adapter = LocalComputeAdapter()
    payload = ScientificJobPayload.model_validate(sample_job_payload_dict)

    res = adapter.execute_payload_sync(payload)
    assert res.status == "COMPLETED"
    assert res.jobId == payload.jobId
    assert res.metrics.cpuSeconds >= 0.0


@pytest.mark.unit
def test_modal_adapter_resource_profiles() -> None:
    """Verify Modal adapter properly resolves logical resource classes to hardware limits."""
    adapter = ModalComputeAdapter()
    assert adapter.mode == ComputeMode.MODAL

    cpu_small = adapter.get_resource_config(ResourceClass.CPU_SMALL)
    assert cpu_small["cpu"] == 2.0
    assert cpu_small["memory"] == 4096

    gpu_large = adapter.get_resource_config(ResourceClass.GPU_LARGE)
    assert gpu_large["gpu"] == "A10G"
    assert gpu_large["cpu"] == 8.0
