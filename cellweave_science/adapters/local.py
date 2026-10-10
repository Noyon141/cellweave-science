"""Local execution compute adapter.

Allows executing scientific tasks locally for development and unit testing
without connecting to Modal or requiring cloud credentials.
"""

from __future__ import annotations

import time

from cellweave_science.adapters.compute import ComputeAdapter, ComputeMode
from cellweave_science.contracts import (
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
from cellweave_science.seeds.seed import set_deterministic_seeds

logger = get_logger("cellweave_science.adapters.local")


class LocalComputeAdapter(ComputeAdapter):
    """Local execution provider implementing the ComputeAdapter interface."""

    def __init__(self) -> None:
        self._jobs: dict[str, ComputeJobStatusResponse] = {}
        self._results: dict[str, ComputeJobResultResponse] = {}

    @property
    def mode(self) -> ComputeMode:
        return ComputeMode.LOCAL

    def submit_job(self, request: ComputeJobSubmitRequest) -> str:
        provider_job_id = f"local-{request.jobId}"
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        self._jobs[request.jobId] = ComputeJobStatusResponse(
            jobId=request.jobId,
            providerJobId=provider_job_id,
            status="RUNNING",
            startedAt=now_iso,
        )
        logger.info(
            "Local compute job submitted",
            jobId=request.jobId,
            jobType=request.jobType,
            resourceClass=request.resourceClass.value,
        )
        return provider_job_id

    def get_status(self, job_id: str, provider_job_id: str) -> ComputeJobStatusResponse:
        if job_id not in self._jobs:
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return ComputeJobStatusResponse(
                jobId=job_id,
                providerJobId=provider_job_id,
                status="FAILED",
                startedAt=now_iso,
                completedAt=now_iso,
                errorMessage="Job not found in local registry",
            )
        return self._jobs[job_id]

    def cancel_job(self, request: ComputeJobCancelRequest) -> bool:
        if request.jobId in self._jobs:
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._jobs[request.jobId] = ComputeJobStatusResponse(
                jobId=request.jobId,
                providerJobId=request.providerJobId,
                status="CANCELLED",
                startedAt=self._jobs[request.jobId].startedAt,
                completedAt=now_iso,
            )
            logger.info("Local compute job cancelled", jobId=request.jobId, reason=request.reason)
            return True
        return False

    def get_result(self, job_id: str, provider_job_id: str) -> ComputeJobResultResponse:
        if job_id in self._results:
            return self._results[job_id]

        return ComputeJobResultResponse(
            jobId=job_id,
            providerJobId=provider_job_id,
            status="FAILED",
            metrics=ComputeJobMetrics(cpuSeconds=0.0, gpuSeconds=0.0, memoryMegabyteSeconds=0.0),
            error=DomainError(
                code="JOB_RESULT_NOT_FOUND",
                message="Result for local job was not found",
                technicalDetail=f"Job {job_id} has not registered a terminal result",
                recommendedAction="Verify job submission completed before requesting result",
                correlationId=job_id,
                retryable=False,
            ),
        )

    def execute_payload_sync(
        self,
        payload: ScientificJobPayload,
        resource_class: ResourceClass | None = None,
    ) -> ComputeJobResultResponse:
        """Synchronously execute a scientific job payload in local mode."""
        start_time = time.monotonic()
        provider_job_id = f"local-sync-{payload.jobId}"
        _selected_resource = resource_class or payload.resourceClass or ResourceClass.CPU_SMALL

        # Enforce deterministic random seeds from the manifest fragment
        seed = payload.manifestFragment.randomSeeds.get("default", 42)
        set_deterministic_seeds(seed)

        logger.info(
            "Executing payload synchronously in local mode",
            jobId=payload.jobId,
            jobType=payload.jobType.value,
            stepId=payload.stepId,
        )

        elapsed = time.monotonic() - start_time
        metrics = ComputeJobMetrics(
            cpuSeconds=round(elapsed, 4),
            gpuSeconds=0.0,
            memoryMegabyteSeconds=round(elapsed * 128.0, 2),
            peakMemoryBytes=128 * 1024 * 1024,
        )

        # In Phase 0, algorithms are not yet implemented; return a clean Phase 0 validated response
        result = ComputeJobResultResponse(
            jobId=payload.jobId,
            providerJobId=provider_job_id,
            status="COMPLETED",
            metrics=metrics,
            resultUri=None,
        )
        self._results[payload.jobId] = result
        return result
