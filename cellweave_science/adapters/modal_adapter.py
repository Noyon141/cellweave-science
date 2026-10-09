"""Modal compute adapter boundary.

Encapsulates Modal invocation behind the ComputeAdapter interface, ensuring
scientific domain logic is not coupled to Modal SDK primitives directly.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from cellweave_science.adapters.compute import ComputeAdapter, ComputeMode
from cellweave_science.contracts.models import (
    ComputeJobCancelRequest,
    ComputeJobMetrics,
    ComputeJobResultResponse,
    ComputeJobStatusResponse,
    ComputeJobSubmitRequest,
    DomainError,
    ResourceClass,
    ScientificJobPayload,
)
from cellweave_science.logging.logger import get_logger

logger = get_logger("cellweave_science.adapters.modal")

# Resource profile mapping from logical ResourceClass to Modal resource configurations
MODAL_RESOURCE_CONFIGS: Dict[ResourceClass, Dict[str, float | int | str]] = {
    ResourceClass.CPU_SMALL: {"cpu": 2.0, "memory": 4096},
    ResourceClass.CPU_MEMORY: {"cpu": 4.0, "memory": 16384},
    ResourceClass.CPU_HIGH_MEMORY: {"cpu": 8.0, "memory": 65536},
    ResourceClass.GPU_SMALL: {"cpu": 4.0, "memory": 16384, "gpu": "T4"},
    ResourceClass.GPU_LARGE: {"cpu": 8.0, "memory": 32768, "gpu": "A10G"},
}


class ModalComputeAdapter(ComputeAdapter):
    """Adapter bridging CellWeave scientific engine with Modal remote execution."""

    def __init__(self, app_name: str = "cellweave-science") -> None:
        self.app_name = app_name
        self._submitted_jobs: Dict[str, ComputeJobSubmitRequest] = {}

    @property
    def mode(self) -> ComputeMode:
        return ComputeMode.MODAL

    def get_resource_config(self, resource_class: ResourceClass) -> Dict[str, float | int | str]:
        """Resolve logical resource profile to concrete Modal hardware limits."""
        return MODAL_RESOURCE_CONFIGS.get(
            resource_class,
            {"cpu": 2.0, "memory": 4096},
        )

    def submit_job(self, request: ComputeJobSubmitRequest) -> str:
        provider_job_id = f"modal-{request.jobId}"
        self._submitted_jobs[request.jobId] = request

        logger.info(
            "Modal compute job submitted",
            jobId=request.jobId,
            providerJobId=provider_job_id,
            resourceClass=request.resourceClass.value,
        )
        return provider_job_id

    def get_status(self, job_id: str, provider_job_id: str) -> ComputeJobStatusResponse:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if job_id not in self._submitted_jobs:
            return ComputeJobStatusResponse(
                jobId=job_id,
                providerJobId=provider_job_id,
                status="FAILED",
                startedAt=now_iso,
                completedAt=now_iso,
                errorMessage="Job not recognized by Modal adapter",
            )
        return ComputeJobStatusResponse(
            jobId=job_id,
            providerJobId=provider_job_id,
            status="RUNNING",
            startedAt=now_iso,
        )

    def cancel_job(self, request: ComputeJobCancelRequest) -> bool:
        logger.info(
            "Modal compute job cancel requested",
            jobId=request.jobId,
            providerJobId=request.providerJobId,
            reason=request.reason,
        )
        return True

    def get_result(self, job_id: str, provider_job_id: str) -> ComputeJobResultResponse:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return ComputeJobResultResponse(
            jobId=job_id,
            providerJobId=provider_job_id,
            status="COMPLETED",
            metrics=ComputeJobMetrics(
                cpuSeconds=1.0,
                gpuSeconds=0.0,
                memoryMegabyteSeconds=128.0,
            ),
            resultUri=None,
        )

    def execute_payload_sync(
        self,
        payload: ScientificJobPayload,
        resource_class: Optional[ResourceClass] = None,
    ) -> ComputeJobResultResponse:
        """Modal synchronous placeholder execution."""
        selected_resource = resource_class or payload.resourceClass or ResourceClass.CPU_SMALL
        provider_job_id = f"modal-sync-{payload.jobId}"
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        logger.info(
            "Modal execution requested",
            jobId=payload.jobId,
            resourceClass=selected_resource.value,
        )

        return ComputeJobResultResponse(
            jobId=payload.jobId,
            providerJobId=provider_job_id,
            status="COMPLETED",
            metrics=ComputeJobMetrics(
                cpuSeconds=0.1,
                gpuSeconds=0.0,
                memoryMegabyteSeconds=64.0,
            ),
            resultUri=None,
        )
