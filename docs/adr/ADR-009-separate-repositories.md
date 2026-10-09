# ADR-009: Separate Repositories Architecture

## Status
Accepted

## Context
CellWeave spans multiple technical domains: Next.js/React frontend & API (`cellweave-web`), Temporal workflow orchestrator (`cellweave-orchestrator`), Python scientific compute engine (`cellweave-science`), shared contract schemas (`cellweave-contracts`), and infrastructure code (`cellweave-infra`).

## Decision
CellWeave maintains five separate repositories with strict architectural boundaries. The `cellweave-science` repository owns only the downstream Python scientific engine, validation, QC, normalization, embeddings, clustering, annotation, DE, pathways, visualization artifact generation, export bundles, provenance generation, and Modal app deployment. It does not own SaaS authentication, billing logic, or customer-facing API routes.

## Consequences
- Clean separation of concerns; scientific engine dependencies (PyTorch, Scanpy, AnnData) do not pollute application or orchestration layers.
- Cross-repository contracts are maintained as code-generated types from `cellweave-contracts`.
- Least-privilege credential scopes for scientific execution containers.
