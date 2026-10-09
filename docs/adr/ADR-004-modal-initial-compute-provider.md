# ADR-004: Modal Initial Scientific Compute Provider

## Status
Accepted

## Context
Scientific workloads in CellWeave require elastic CPU and GPU resources that scale to zero and handle high-memory batches without requiring Kubernetes cluster operations in V1.

## Decision
Modal is selected as the initial elastic compute provider for scientific batch jobs. The scientific engine encapsulates Modal behind a logical `ComputeAdapter` interface (`LOCAL` vs `MODAL`) so domain logic does not depend directly on Modal-specific SDK primitives. For testing and development, local execution must run without Modal dependencies.

## Consequences
- Fast autoscaling and containerized GPU/CPU execution without cluster management overhead.
- Portability preserved via the `ComputeAdapter` boundary to permit future adapters (AWS Batch, RunPod).
- Science container images run containerized workloads isolated from business web servers.
