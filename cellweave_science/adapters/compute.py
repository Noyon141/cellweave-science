"""Compute provider abstraction and interface.

Defines the contract for executing scientific jobs either locally (for testing
and local development) or on Modal (for production elastic compute).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from cellweave_science.contracts import (
    ComputeJobCancelRequest,
    ComputeJobResultResponse,
    ComputeJobStatusResponse,
    ComputeJobSubmitRequest,
    ResourceClass,
    ScientificJobPayload,
)


class ComputeMode(str, Enum):
    """Execution mode selection."""

    LOCAL = "LOCAL"
    MODAL = "MODAL"


class ComputeAdapter(ABC):
    """Abstract interface for scientific compute execution."""

    @property
    @abstractmethod
    def mode(self) -> ComputeMode:
        """Return the compute mode handled by this adapter."""
        ...

    @abstractmethod
    def submit_job(self, request: ComputeJobSubmitRequest) -> str:
        """Submit a job for execution and return the provider-specific job ID."""
        ...

    @abstractmethod
    def get_status(self, job_id: str, provider_job_id: str) -> ComputeJobStatusResponse:
        """Query execution status of a submitted job."""
        ...

    @abstractmethod
    def cancel_job(self, request: ComputeJobCancelRequest) -> bool:
        """Request cancellation of an in-flight job."""
        ...

    @abstractmethod
    def get_result(self, job_id: str, provider_job_id: str) -> ComputeJobResultResponse:
        """Retrieve completed job result or terminal failure details."""
        ...

    @abstractmethod
    def execute_payload_sync(
        self,
        payload: ScientificJobPayload,
        resource_class: ResourceClass | None = None,
    ) -> ComputeJobResultResponse:
        """Direct synchronous execution entrypoint for testing and local smoke runs."""
        ...
