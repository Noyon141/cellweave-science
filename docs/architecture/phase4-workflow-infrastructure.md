# Phase 4 durable execution infrastructure

Phase 4 orchestrates the approved Phase 3 import/canonicalization/publication path. It adds no QC, normalization, clustering, annotation, billing, or results UI.

## Authority and persistence

Web authenticates the customer and resolves TenantContext, project access, persisted import membership, registered sample mappings, and server-configured pipeline/resource ceilings. A required Idempotency-Key is bound to organization, actor, and an immutable request digest. The same request returns the same run; a conflicting request fails.

Migration 0005 introduces the master-design analysis_runs and workflow_steps operational records. Creation commits START_PENDING before Temporal start. A lost start acknowledgment is recovered using deterministic workflow ID analysis-run:{analysisRunId}; workflow type and run/envelope identity must match. Temporal owns execution history and timers; PostgreSQL owns business authorization and customer projections. Workflow code performs no arbitrary network/database work. Activities make authenticated internal API calls.

Run/step transitions are generated from contract schemas. Sequence numbers reject stale projections. Logical submission IDs and persisted native provider IDs prevent duplicate compute submission. Unknown provider outcomes remain pending reconciliation; no fabricated provider IDs or metrics are generated. Bounded retries classify permanent input/security errors separately from network/rate-limit uncertainty.

Cancellation is durable and idempotent. A persistent provider cancel tombstone covers cancellation before submission. Finalization intent serializes against cancellation: cancellation before that boundary prevents scientific commit; once commit has started, completion/replay wins. Only the existing isolated finalizer-role SECURITY DEFINER function mutates scientific truth. It remains atomic and idempotent after acknowledgment loss.

## Service authentication

Approved JWT convention: jose 6.2.12, ES256 only, P-256; issuer owns PKCS#8 PEM private key; verifier holds configured SPKI public keys indexed by exact kid. Default TTL 300 seconds, hard maximum 900 seconds, clock tolerance 30 seconds. Required claims: iss, sub, aud, iat, exp, jti, scope. Issuer is cellweave-orchestrator:{environment}; subject cellweave-orchestrator; audience cellweave-web-internal:{environment}. Environment is local/development/staging/production. Keys are separate per environment. Unknown kid/algorithm/claim/scope fails closed. Rotation adds public trust first, switches active issuer key, waits longer than max TTL, then removes retired trust. No shared master key, public JWKS endpoint, or runtime key generation.

Scopes are execution.prepare, execution.project, execution.finalize, execution.cancel, capability.issue; each operation verifies its required scopes before resolving the assigned executor and authoritative resource relationships. JWT validity never grants caller-supplied tenant authority. Customer sessions and Temporal native authentication remain separate.

Issuer configuration: SERVICE_AUTH_ENVIRONMENT, SERVICE_AUTH_ACTIVE_KID, SERVICE_AUTH_PRIVATE_KEY_PEM, optional SERVICE_AUTH_TTL_SECONDS/SERVICE_AUTH_ALGORITHM. Web verifier configuration: CELLWEAVE_ENVIRONMENT, SERVICE_AUTH_ENVIRONMENT and SERVICE_AUTH_TRUST_BUNDLE (JSON array of kid/publicKeyPem). Private material never goes to web verifier or science. Tokens, keys, signed URLs and credentials must never be logged.

## Compute and science

The production adapter calls a configured HTTPS private Modal endpoint using native Modal proxy credentials. Science receives only a validated execution fragment and scoped input/output capabilities, never DB, BetterAuth, R2 master, service JWT, or billing credentials. It streams bounded downloads and verifies exact persisted source facts before using the existing Phase 3 importer and publisher. Source checksum semantics remain format-specific. Conditional publication and exact-byte verification remain unchanged.

Modal SDK is pinned to 0.77.0. Deployment requires CELLWEAVE_ENVIRONMENT and CELLWEAVE_MODAL_PROFILE_JSON containing resourceClass, cpu, memoryMiB and maxRuntimeSeconds. Operator provisions the environment-isolated Modal Dict before deployment; code does not silently provision it. Atomic logical claims map to real native FunctionCall IDs; unacknowledged claims require reconciliation rather than blind resubmission. Expiring capability URLs are excluded from submission identity. The admission deadline is shorter than retained native idempotency records.

CPU-only resource profiles are enforced by native allocation, child-process CPU limit, wall deadline and memory sampling/native memory ceiling. maxComputeUnits means a positive integer process CPU-second ceiling, not billable credits. Usage records are measured CPU seconds, sampled MiB-seconds and peak bytes. maxRuntimeSeconds is bounded at 3300 because the existing publication capability TTL is capped at 3600 seconds; 300 seconds is reserved for handoff. No source bytes pass through Next.js.

## Runtime and qualification

Worker configuration requires explicit nonlocal Temporal address/namespace/task queue and native TLS/API-key settings, plus matching service identity and Modal configuration. The canonical queue is cellweave-analysis-queue. Containers run non-root, install CA certificates, handle SIGTERM and expose timestamped liveness/readiness checks. Periodic reconciliation recovers pending starts, cancellation intent and incomplete projections. No infra apply/deployment is part of local validation.

The local test:recovery harness uses real Temporal workflow history, separate killable worker processes, all SQL migrations and disposable PostgreSQL with the real finalizer role/function. It kills workers before submit, after provider acceptance before acknowledgment, while polling, after provider completion, after SQL finalization before acknowledgment, and during cancellation. The external compute provider is a durable test fixture in that harness. These tests prove worker-loss/idempotency/SQL behavior; they do not claim a live Modal, Temporal Cloud, R2 or deployed privilege qualification. Live DEV qualification remains separate and must report NOT RUN when credentials are unavailable.

CI recovery qualification checks out the pinned web application revision beside orchestrator. Private repositories require a read-only repository checkout credential in CELLWEAVE_REPOSITORY_READ_TOKEN; this CI credential is unrelated to production service JWTs. No deployment is triggered.
