# CellWeave Canon — Operational Mirror

This document is a high-signal operational mirror of the authoritative CellWeave Production Master System Design v1.0. It exists so agents can quickly recover invariants. It does not supersede the PDF.

## Product identity

CellWeave is a production-grade, multi-tenant B2B scientific SaaS for small/mid-sized biotech companies and CRO/bioinformatics teams. V1 performs downstream single-cell RNA-seq analysis from processed inputs and is Research Use Only.

Hard product boundaries:
- Research Use Only.
- Non-diagnostic.
- No clinical decision support or treatment recommendations.
- PHI is prohibited.
- Processed-data-first.
- Human-reviewable.
- Scientifically reproducible.
- Multi-tenant secure.
- Compute-provider portable.
- Billing-provider isolated.

## V1 supported scientific scope

Supported:
- `.h5ad`
- 10x filtered matrices
- CSV/TSV expression matrix + metadata
- human datasets
- mouse datasets
- dataset validation
- per-sample QC
- doublet identification
- user-controlled filtering
- normalization
- highly variable gene selection
- PCA
- neighbor graph
- UMAP
- Leiden clustering
- automated cell-type annotation
- manual annotation correction
- marker-gene analysis
- differential expression
- pathway/enrichment
- experiment comparison
- interactive exploration
- shared projects
- comments
- analysis history
- immutable run provenance
- export bundles
- Polar subscription + compute usage billing
- operations console
- audit trails

Explicit V1 non-goals:
- raw FASTQ processing
- clinical diagnosis
- treatment recommendations
- medical decision support
- patient management
- PHI workflows
- native mobile apps
- arbitrary user-written Python
- arbitrary user-written R
- n8n-style visual pipeline builder
- real-time Google-Docs-style editing
- spatial transcriptomics
- proteomics
- ATAC-seq
- full multi-omics
- custom ML model training

## Five repository boundary

### cellweave-web
Owns:
- Next.js App Router
- React / TypeScript
- shadcn/ui / Tailwind / Motion
- BetterAuth
- Drizzle / Neon access
- R2 integration
- Polar integration
- Temporal client
- customer application
- operations console
- public API
- webhooks

It must not become a scientific compute server.

### cellweave-orchestrator
Owns:
- Temporal workflows
- Temporal activities
- compute-provider interfaces
- Modal adapter
- workflow state projection
- usage reporting
- cancellation
- retry policies
- reconciliation

It exposes no customer-facing REST API. It must not become a business database.

### cellweave-science
Owns:
- Python scientific engine
- importers
- validation
- QC
- normalization
- embeddings
- clustering
- annotation
- differential expression
- pathways
- visualization artifact generation
- exports
- provenance
- Modal application
- scientific tests / benchmarks / reference data

It must not become a SaaS backend.

### cellweave-contracts
Owns cross-language source-of-truth contracts:
- OpenAPI 3.1
- JSON Schema
- event schemas
- run manifest schema
- artifact manifest schema

It generates:
- TypeScript types
- Python Pydantic models

No duplicated handwritten cross-repository payload interfaces.

### cellweave-infra
Owns:
- Terraform/OpenTofu
- environment definitions
- Cloudflare configuration
- Temporal configuration
- deployment definitions
- monitoring configuration
- security policies
- CI templates
- runbooks

## The four boundaries that must never collapse

1. WEB / BUSINESS
2. ORCHESTRATION
3. SCIENTIFIC COMPUTE
4. SCIENTIFIC ARTIFACTS

Never:
- turn Next.js into the compute worker
- turn the science worker into the SaaS backend
- use Temporal as the business database
- store large scientific matrices in Postgres
- use R2 as the permissions database
- use Polar as the entitlement engine

## Authorization

Authorization is two-level:
- Organization
- Project

Every protected tenant-owned domain object must resolve to `organizationId`.

Never trust browser-supplied `organizationId` without server-side membership derivation and validation.

Every tenant-owned repository query must accept a `TenantContext`, not a free-form tenant id.

Cross-tenant authorization integration tests are mandatory for every protected resource type.

Where practical, production domain tables also use Postgres RLS as defense-in-depth. Application authorization remains mandatory.

## Roles

Organization Owner:
- subscription
- organization
- users
- security
- retention
- billing
- audit logs
- organization deletion

Organization Admin:
- members
- projects
- project access
- usage
- most org settings
- cannot transfer ownership without Owner permission

Bioinformatician:
- import
- metadata
- analysis parameters
- run/re-run
- QC
- annotation
- comparisons
- scientific exports

Scientist:
- explore
- review QC
- interact with visualizations
- adjust approved parameters
- comparisons
- annotate
- comments
- exports
- no organization administration

Viewer:
- read-only approved projects/results/reports
- cannot mutate scientific state

## Core user journey

Create Organization → Create Project → Upload Dataset → Validate → Configure Samples/Metadata → Generate QC → Scientist Reviews QC → Confirm Filtering → Start Analysis → Pipeline Executes → Review Clusters → Automatic Annotation → Human Review/Corrections → Marker Analysis → Condition Comparison → Differential Expression → Pathway Analysis → Share/Discuss → Export Reproducibility Bundle.

QC filtering must not silently destroy cells through universal thresholds. Suggested thresholds require user review and an immutable QC Decision before downstream analysis.

## Web architecture

Feature/domain organization, not business logic in route files.

Required conceptual structure:
- `src/app/(marketing)`
- `src/app/(auth)`
- `src/app/(app)/dashboard`
- `projects`
- `datasets`
- `analyses`
- `comparisons`
- `billing`
- `settings`
- `src/app/(ops)`
- `src/app/api/auth`
- `src/app/api/v1`
- `src/app/api/internal`
- `src/app/api/webhooks/polar`
- `src/features/*`
- `src/server/domain`
- `src/server/application`
- `src/server/infrastructure`
- `src/server/authorization`
- `src/server/database`
- `src/server/storage`
- `src/server/integrations`
- `src/components/ui`
- `src/components/scientific`
- `src/components/animations`
- `src/types`

Route Handlers stay thin:
authentication → validation → authorization → application service → response mapping.

Server Actions may be used for UI ergonomics but must call the same application layer.

Durable domain APIs use `/api/v1/*`.

## Authentication

V1:
- email/password
- Google OAuth
- BetterAuth organizations

Architecture remains ready for future OIDC/SAML enterprise SSO.

No auth logic belongs in scientific services.

## Upload architecture

Large scientific files never flow browser → Vercel request body → R2.

Correct pattern:
browser → authorized upload initialization in Next.js → presigned/multipart direct browser upload to private R2 → complete/verify.

Upload states:
CREATED → UPLOADING → UPLOADED → VERIFYING → VERIFIED → IMPORTED.

Failure states:
FAILED, ABORTED, EXPIRED, QUARANTINED.

After upload, ingest computes:
- SHA-256
- file size
- detected format
- compression metadata

R2 multipart ETag is not a scientific content hash.

Cross-organization deduplication must never disclose identical customer data.

Uploads begin quarantined. Validate limits, declared/actual format, file signature, archive traversal, decompression ratio, file count, expected structure, schema sanity, dimensions.

Never run uploaded code.

ZIP handling rejects:
- `../`
- absolute paths
- sandbox-escaping symlinks
- zip bombs
- unexpected executables

## Canonical scientific representation

All supported import formats convert to canonical AnnData:
`canonical.h5ad`

The downstream pipeline operates on this canonical representation.

Future storage enum may include H5AD and ZARR, but V1 canonical behavior remains as specified.

Ambiguous generic matrix inputs must never be silently guessed; users explicitly map cells, genes, samples, conditions, batches, and replicates.

## Scientific pipeline

Stage 0 validation:
dimensions, sparse/dense, cell ids, gene ids, metadata, species, sample assignment, count availability.

No analysis starts on scientifically malformed datasets.

QC:
- total counts
- detected genes
- mitochondrial percentage
- ribosomal metrics
- doublet score
- per-sample analysis, not only global

Default doublet implementation:
- Scrublet
- store score, predicted flag, algorithm version, parameters
- do not immediately delete predicted doublets

QC gate:
status becomes `QC_READY_FOR_REVIEW`; user confirms filtering; immutable QC Decision records thresholds, included/excluded cells, doublet handling, user, timestamp, reason.

Normalization:
- retain raw counts
- library-size normalization
- log1p
- never overwrite raw counts

Feature selection:
- highly variable genes
- version parameters
- account for batch/sample where appropriate

Dimensional reduction:
PCA → neighbors → UMAP.
Store both numerical embeddings and visualization artifacts.

Clustering:
- Leiden
- validated resolution options
- every clustering change produces a new artifact/version
- never silently mutate historical assignments

Annotation:
- provider abstraction
- human/mouse
- output predicted label, confidence, method, model, model version, reference dataset, timestamp
- insufficient confidence → `Unassigned`
- human accept/reject/replace/merge/split
- manual correction never erases automated provenance

Advanced ML:
- scVI/scANVI is advanced analysis, not required for every basic run
- route workloads to CPU/GPU according to need

Markers:
- validated non-parametric methods such as Wilcoxon
- store gene, cluster, score, log fold change, p-value, adjusted p-value, method, parameters

Differential expression:
- cluster-vs-cluster cell-level analysis may be used when explicitly marker discovery
- condition comparisons default toward sample-level/pseudobulk when biological replicates exist
- never treat cells as independent biological replicates
- validate samples, conditions, replicates, batches, cell counts
- weak comparison quality produces a scientific warning

Pathways:
- provider abstraction
- record database, database version, gene universe, method, parameters, multiple-testing correction
- review licensing before commercial production
- reference databases never silently change beneath an existing analysis

## Experiment hierarchy

Organization → Project → Experiment → Samples → Dataset Versions → Analysis Runs.

Comparisons are immutable analytical objects. Parameter edits create a new comparison version.

## Reproducibility and lineage

Every completed run creates immutable Run Manifest including:
- run id
- pipeline name/version
- container digest
- dataset version/input checksum
- organism
- parameters
- references
- software
- random seeds
- steps
- artifacts
- timestamps

Pipeline versions are immutable after production use.

Every derived artifact must know:
- generator
- inputs
- step
- version

Outputs form a DAG.

Avoid unnecessary recompute:
- chart color change does not rerun science
- annotation change does not rerun UMAP
- filtering change invalidates downstream artifacts and branches a new run

## Temporal

Main durable workflow:
validate entitlement → validate dataset → prepare canonical input → QC → wait for human QC signal → filter → normalize → HVG → PCA → neighbors → UMAP → clustering → automatic annotation → markers → artifacts → provenance → usage → complete.

Signals include:
- confirm QC
- cancel
- manual annotation decision
- resume after quota purchase

Temporal is authoritative for execution state/timers/retries/activity progress.

Postgres is authoritative for business entities/customer projections/permissions/billing state/scientific metadata.

Workflow code must not perform arbitrary DB/network calls; non-deterministic work belongs in Activities.

Stable workflow id:
`analysis-run:{analysisRunId}`

## Analysis states

Run states:
DRAFT, UPLOADING, VALIDATING, QC_RUNNING, QC_REVIEW_REQUIRED, QUEUED, RUNNING, PAUSED_QUOTA, CANCEL_REQUESTED, CANCELLED, SUCCEEDED, FAILED.

Terminal:
SUCCEEDED, FAILED, CANCELLED.

Workflow step states:
PENDING, QUEUED, RUNNING, SUCCEEDED, RETRYING, FAILED, SKIPPED, CANCELLED.

Each attempt records attempt number, timing, resources, error classification, cost, and logs reference.

Retry transient failures; do not infinitely retry invalid data/schema/scientific validation, OOM after ceiling, permission failure, or quota exhaustion.

Failures must be classified.

## Compute

Science engine logical jobs:
- validate_dataset
- run_qc
- filter_cells
- normalize
- select_hvg
- run_pca
- build_neighbors
- run_umap
- cluster
- annotate
- find_markers
- run_differential_expression
- run_pathway_analysis
- generate_exports

Temporal delegates through `ComputeProvider`.

Contract:
- submit
- status
- cancel
- result

Initial implementation:
- `ModalComputeProvider`

No domain logic directly calls Modal.

Resource classes:
CPU_SMALL, CPU_MEMORY, CPU_HIGH_MEMORY, GPU_SMALL, GPU_LARGE.

Resource estimation uses cells, genes, sparsity, dataset size, method, batch count.

Every expensive run receives estimatedComputeUnits, maxComputeUnits, maxRuntime, maxMemoryClass. Enforced ceilings terminate compute.

## Storage

R2 is private; no public dataset bucket.

Path convention:
`org/{orgId}/project/{projectId}/datasets/{datasetVersionId}/...`
and
`org/{orgId}/project/{projectId}/runs/{runId}/...`

Downloads:
browser → CellWeave API permission check → short-lived signed URL → browser downloads from R2.

Never expose permanent public object URLs.

Postgres stores structured metadata. R2 stores heavy scientific artifacts.

## Data model invariants

Use PostgreSQL/Neon + Drizzle.

Core domain tables include:
organizations, organization_settings, projects, project_members, experiments, samples, datasets, dataset_versions, dataset_files, upload_sessions, imports, pipeline_definitions, pipeline_versions, analysis_runs, analysis_run_parameters, workflow_steps, artifacts, artifact_lineage, cluster_annotations, annotation_revisions, comparisons, comparison_versions, comments, reports, audit_events, notifications, usage_ledger, usage_aggregates, billing_accounts, billing_subscriptions, entitlements, polar_events, polar_sync_outbox, idempotency_keys, service_events, data_deletion_jobs.

BetterAuth owns its auth/session/organization tables where appropriate; do not duplicate without domain reason.

External identifiers use sortable server-generated identifiers such as UUIDv7, never sequential integer ids.

Dataset versions that have become analysis inputs are immutable.

Annotation changes retain history.

Audit events are append-only at application level. Never place raw scientific datasets in audit metadata.

## Service and worker security

Science compute receives only:
- specific input signed URL
- specific output upload capability
- run manifest fragment
- required parameters

Science compute does not receive:
- BetterAuth credentials
- Polar credentials
- R2 master credentials
- DB admin credentials

Service-to-service communication uses short-lived signed service credentials with issuer, audience, scope, expiration. Static shared master keys are not the permanent architecture.

## Billing

Commercial billing provider: Polar.

Model:
subscription + included credits + metered usage.

CellWeave internal immutable usage ledger is source of truth for usage/entitlement reconciliation. Polar is commercial billing system.

Internal measurements:
cpu_seconds, gpu_seconds, memory_class, bytes_processed, storage_byte_hours, egress_bytes.

Customer-facing:
Compute Credits, Storage Allowance.

Conversion is configuration-driven.

Usage flow:
scientific step → usage event → usage_ledger INSERT → billing outbox INSERT → Polar event → acknowledgment.

Idempotency key:
`runId:stepId:attempt`

Entitlements are local:
maxUsers, maxProjects, storageBytes, includedComputeCredits, concurrentRuns, retentionDays, advancedML, SSO, auditExport, priorityCompute.

Polar webhook updates subscription projection; application derives entitlements.

Plans use `PlanDefinition` + `Entitlement`; do not hardcode plan names throughout domain code.

Polar webhook requires signature verification, unique external event id, persistence before business processing, idempotent/retry-safe handlers, dead-letter visibility.

Billing is organization-scoped.

## Visualization/UI

Browser never downloads a multi-gigabyte H5AD merely to render UMAP.

Science generates optimized visualization artifacts.

Small datasets: Canvas.
Larger datasets: WebGL.
Use a rendering abstraction such as `ScientificScatterRenderer`.

Result page sections:
Overview, QC, UMAP, Clusters, Cell Types, Markers, Differential Expression, Pathways, Comparisons, Artifacts, Provenance, Activity.

Every chart must answer:
- what am I seeing?
- which run produced it?
- which parameters?
- can I export underlying values?

Dashboard is operational:
active projects, recent/failed analyses, QC review queue, compute/storage usage, team activity.

Analysis wizard:
dataset → metadata → pipeline → validated parameters → estimated compute → start.

Scientific errors use domain codes/messages/actions/correlation ids, not generic 500 as primary UX.

Notifications V1:
in-app + Resend email for analysis completion/failure, QC review, comparison completion, invitation, billing issue, storage threshold, compute threshold.

Never email raw scientific results unless intentionally exported.

## Operations

`/(ops)` uses completely separate staff authorization.

Capabilities:
tenant lookup, subscription/usage/storage, running/failed analyses, Temporal IDs, Modal IDs, retry/cancel, billing sync, webhook failures, notifications, incidents, feature flags.

Staff cannot inspect customer scientific data by default.

Break-glass access requires privileged role, explicit reason, short expiration, audit event. No silent impersonation.

Support diagnostics use correlation ids and operational metadata without raw scientific content by default.

## Exports

Export:
processed H5AD, CSV, figures, QC, DE, pathways, report, Reproducibility Bundle.

Bundle:
run-manifest.json, parameters.json, software-versions.json, artifact-manifest.json, results/, README.

## Retention/deletion

Lifecycle:
ACTIVE → SOFT_DELETED → RECOVERY_WINDOW → PURGE_PENDING → PURGED.

Organization deletion requires Owner authentication, confirmation, optional export, grace period, scheduled purge.

Purge covers projects/datasets/artifacts/comments/scientific results/API credentials while retaining minimized financial/audit records only where legitimately required.

Deletion must never affect another tenant.

## Security baseline

Required:
TLS everywhere, encryption at rest, private storage, short-lived signed URLs, least privilege, strict tenant auth, rate limiting, secure headers, CSRF where relevant, SameSite secure cookies, session rotation, webhook verification, idempotency, dependency scanning, secret scanning, container vulnerability scanning, audit logging.

Secrets never go in Git, browser bundles, scientific artifacts, run manifests, or logs.

Separate secrets per environment and rotate independently.

## SOC 2 readiness

Architecture/process is SOC 2-friendly but certification is not an early requirement.

Controls include staff accounts, MFA, production access review, least privilege, change approval, CI/CD evidence, audit logging, incident response, backup testing, vulnerability management, vendor inventory, secret management, offboarding, policies.

## Observability

Three pillars:
- logs
- metrics
- traces

Use OpenTelemetry, Sentry, structured logs, Temporal observability, Modal metrics.

Correlate where appropriate with requestId, organizationId, analysisRunId, workflowId.

Never log scientific matrices.

Separate PLATFORM_ERROR from SCIENTIFIC_VALIDATION_ERROR.

Customer scientific warnings remain in run provenance.

## Reliability / DR

Initial:
- web/API monthly availability 99.9%
- workflow request durability >99.99%
- successful upload verification >99.5%
- billing reconciliation 100%
- cross-tenant access 0

Metadata:
Neon PITR + scheduled logical backups.
Initial Metadata RPO ≤1 hour, RTO ≤4 hours.

Quarterly restore testing is mandatory.

## Environments

Required:
- local
- development
- staging
- production
- Vercel previews

Production customer data never enters previews.

Separate Neon, R2, Temporal namespaces, Modal environments, Polar sandbox/prod, OAuth credentials, secrets.

Preview DBs use sanitized staging schema/data, never arbitrary production customer metadata.

Staging uses the same architectural shape as production.

## CI/CD

Web PR:
install → lint → format → typecheck → unit tests → authorization tests → DB migration validation → build → Playwright smoke → preview.

Web production:
migration safety gate → deploy → health check → synthetic smoke → progressive feature activation.

Science:
lint → type checks → unit tests → scientific regression → golden datasets → container build → SBOM → vulnerability scan → benchmark.

Algorithm change cannot deploy merely because unit tests pass.

## Scientific regression

Golden datasets include:
small smoke, human PBMC, mouse, multi-sample, batch-effect, known-treatment, large benchmark.

Evaluate:
filter cell counts, QC distributions, PCA properties, cluster consistency, marker recovery, annotation accuracy, DE direction, pathway expectations, runtime, memory.

Use meaningful tolerance; do not require byte-identical stochastic UMAP.

Pipeline versions are immutable. Existing runs remain pinned.

## Database migrations

Use expand → migrate/backfill → switch → contract.

Avoid unsafe locking/drop/rewrite patterns. Preserve backward compatibility through deployment windows.

## API/contract compatibility

Breaking public/API changes require `/v2`.

Internal contracts have explicit schema versions.

Scientific job payload always contains `schemaVersion` and `pipelineVersion`.

## Idempotency / consistency

Idempotency required for:
upload completion, run creation, workflow startup, Polar webhooks, billing events, notifications, artifact registration.

Write requests may accept `Idempotency-Key`.

Run creation uses START_PENDING + deterministic Temporal start + reconciliation retry so paid jobs are not lost.

Background reconciliation checks START_PENDING, workflow projections, stale uploads, orphaned multipart uploads, unreported usage, unsynced Polar events, stuck deletion jobs, expired upload sessions.

## Performance

Server Components for data-heavy initial views.

Client components only where interactivity requires them: plots/forms/filters/animations/live status.

TanStack Query for dynamic client data.

Do not make the dashboard a client-rendered SPA.

Animations are standardized under `components/animations` and respect reduced motion.

Desktop is primary, but laptop/tablet/mobile review remains functional. Full high-density scientific parity is not required on mobile.

Target WCAG 2.2 AA.

No UI request waits synchronously on a scientific pipeline.

Large uploads are resumable. Result visualization progressively loads.

## Threat/cost protection

Mitigate:
cross-tenant IDOR, public scientific files, malicious archives, worker compromise, duplicate workflows, duplicate billing, webhook replay, privilege escalation, compute abuse, secret leaks, dependency compromise, silent scientific regression.

Controls include tenant authorization/tests/RLS, private R2/signed URLs, sandbox extraction, scoped worker credentials, deterministic Temporal ids, immutable usage ids, webhook signatures/uniqueness, RBAC, quotas/resource ceilings, isolated secrets, lockfiles/SBOM/scanning, benchmark/golden dataset gates.

## Feature flags

Scope by environment, organization, user.

Scientific rollouts begin with internal/test tenants.

## Interfaces that isolate providers

- `DatasetImporter`
- `ObjectStorage`
- `BillingProvider`
- `NotificationProvider`
- `ComputeProvider`
- `PathwayProvider`
- `AnnotationProvider`

Business/domain logic must not depend directly on provider payloads.

## Scalability

Stage 1:
Vercel, Neon, R2, Temporal Cloud, 1–2 orchestrator replicas, Modal autoscaling. No Kubernetes.

Stage 2:
more workers, task queues by resource type, DB read scaling, result caching, higher Modal concurrency, per-org workflow limits.

Stage 3:
regional compute pools, dedicated enterprise queues, optional AWS Batch/ECS, dedicated storage/DB, cross-region DR.

Do not introduce Kubernetes prematurely.

## Product analytics/privacy

Analytics may track operational events such as project/dataset/run/comparison/export/upgrade events.

Never send gene names, patient identifiers, matrices, or biological conclusions to ordinary analytics.

Third parties receive minimum operational metadata. Maintain subprocessor inventory.

## Search

V1 search is Postgres-based for projects, datasets, experiments, analysis runs, users.

Do not introduce Elasticsearch/OpenSearch prematurely.

## Reports

Scientific reports are asynchronous artifacts and never a separate source of truth.

## Payment failure

Payment recovery does not immediately delete scientific data.

Account states may include ACTIVE, PAST_DUE, SUSPENDED, CLOSING, DELETED.

Restriction may prevent new expensive runs while allowing viewing/export. Deletion follows retention policy.

## Documentation

Maintain:
Architecture, data model, security model, scientific methods, API reference, runbooks, deployment guide, incident response, billing model, data retention, pipeline validation.

Documentation is part of feature Definition of Done.

## ADRs

Maintain `/docs/adr`.

Baseline decisions include:
AnnData canonical representation, R2, Temporal, Modal, Polar, processed-data-only, pseudobulk condition DE, no PHI, separate repositories.

Changing one requires explicit architectural review.
