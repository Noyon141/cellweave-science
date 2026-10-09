# CellWeave System Design Analysis for Agentic Implementation

## 1. What the architecture is optimizing for

CellWeave is not designed as a thin SaaS wrapper around Scanpy. Its architecture separates five concerns that fail differently and scale differently:

- user/business application
- durable orchestration
- scientific computation
- scientific artifact storage
- commercial/operational infrastructure

The design's central operating principle is controlled scientific computation with reproducibility, human review, tenant isolation, cost control, and recoverable long-running workflows.

## 2. Highest-risk architectural failures the agent environment must prevent

### Boundary collapse
The most damaging drift would be:
- running scientific compute from Next.js
- putting SaaS auth/business logic into Python workers
- using Temporal as customer/business persistence
- putting large scientific data in Postgres
- treating R2 as authorization
- letting Polar directly determine product entitlements

These are blocked by always-on repository boundary rules.

### Tenant isolation failure
Every protected domain object ultimately belongs to an organization and may also be constrained by project access. The agent environment therefore makes server-derived TenantContext and cross-tenant tests mandatory.

### Scientific silent mutation
The design treats scientific history as immutable. Dataset versions, pipeline versions, comparisons, annotations/revisions, artifacts, and provenance must preserve history. Agent rules forbid "update in place" shortcuts that destroy reproducibility.

### Scientific overconfidence
QC, annotation, DE, and pathways require human/scientific safeguards. The environment forbids silent universal QC thresholds, forced low-confidence annotation, and cell-as-replicate condition DE.

### Long-running request fragility
Scientific analysis cannot depend on Vercel/HTTP lifetime. Temporal is the durable owner of execution. The environment forces side effects into Activities, deterministic workflow ids, bounded retries, cancellation, signals, and reconciliation.

### Cost runaway
Every expensive run requires estimation and ceilings. The environment prevents an agent from introducing unbounded compute or inventing commercial limits before benchmarking.

### Billing duplication
Usage is immutable and idempotent. The environment binds usage identity to run/step/attempt and preserves local ledger + outbox + Polar projection.

### Upload security
Large scientific uploads bypass Vercel, land private/quarantined, are verified by real SHA-256, and are inspected safely. The environment blocks upload proxying, public buckets, executable uploads, traversal, zip bombs, and cross-tenant dedup leakage.

### Operational blindness
Production readiness includes structured telemetry, correlation ids, provider health, runbooks, DR tests, and support diagnostics without scientific raw-content access.

## 3. Why rules and skills are separated

Always-on rules encode invariants that must affect every edit:
- repository boundaries
- RUO/PHI
- tenancy
- immutability
- source-of-truth
- production readiness
- no-guess behavior

Task-specific rules activate when the task involves:
- uploads
- billing
- APIs
- database
- scientific methods
- Temporal
- CI/CD
- DR
- results UI

Skills encode multi-step execution procedures:
- implement feature
- add API endpoint
- change database
- change workflow
- add compute step
- change scientific algorithm
- change contracts
- qualify production

This keeps agent context focused while preserving non-negotiable constraints.

## 4. Design choices that are deliberately unresolved

The architecture intentionally leaves some implementation decisions open. The agent environment treats those as "do not guess" inputs rather than invitations to choose a favorite tool.

Examples:
- package managers
- initial always-on worker host
- exact annotation model
- exact DE/enrichment tool
- exact limits/prices/retention durations
- rate-limit numbers
- feature flag/log/analytics vendors
- service credential implementation

This is necessary to satisfy the user's zero-improvisation requirement.

## 5. Implementation dependency structure

The 15 phases are real dependency gates, not a backlog ordering suggestion.

Identity/tenancy must exist before customer data features.
Dataset/upload must exist before science import.
Import must exist before workflow execution.
Workflow durability must exist before QC pipeline.
QC gate must exist before core analysis.
Core analysis precedes annotation/comparison/results.
Billing and operations come after core product behavior.
Security/governance and production qualification close the production-readiness loop.

The agent environment therefore makes phase-gate awareness always-on in every repository.

## 6. Separate repository environment model

Each repository receives:
- root `AGENTS.md`
- flat `.agents/rules/*.md`
- focused `.agents/skills/<skill>/SKILL.md`
- immutable architecture PDF
- operational canon
- do-not-guess list
- implementation order
- production Definition of Done
- coverage map

This preserves the authoritative separate-repository design while giving Antigravity enough local context to act autonomously.

## 7. Expected agent behavior after installation

A good agent task should become:

requirement trace
→ repository ownership check
→ rules/skill activation
→ contract impact
→ tenancy/security impact
→ scientific/provenance impact
→ implementation
→ tests
→ failure/idempotency checks
→ observability
→ documentation
→ architecture invariant review

The environment is designed so "make it work" cannot legally bypass "make it correct, recoverable, tenant-safe, reproducible, and bounded."
