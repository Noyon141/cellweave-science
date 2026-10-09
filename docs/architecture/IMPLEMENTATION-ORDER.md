# CellWeave Mandatory Implementation Order

Do not skip ahead when a later phase depends on an unmet earlier exit criterion.

## Phase 0 — Foundations
Deliver:
- all repositories
- CI
- environment strategy
- contract generation
- coding standards
- ADR system
- infrastructure foundation

Exit: all repositories deploy placeholder services to development/staging.

## Phase 1 — Identity & Tenancy
Deliver:
- BetterAuth
- Google OAuth
- organizations
- RBAC
- project permissions
- organization settings
- audit framework

Exit: automated cross-tenant access tests pass.

## Phase 2 — Project & Dataset Foundation
Deliver:
- projects
- experiments
- samples
- datasets
- dataset versions
- R2
- multipart uploads
- upload sessions

Exit: a large interrupted upload can resume and verify.

## Phase 3 — Scientific Import System
Deliver:
- H5AD importer
- 10x importer
- matrix importer
- canonicalization
- input checksum
- metadata wizard

Exit: all golden importer datasets canonicalize to valid AnnData.

## Phase 4 — Workflow Infrastructure
Deliver:
- Temporal Cloud
- orchestrator worker
- run state machine
- Modal adapter
- cancellation
- retry
- reconciliation

Exit: kill workers during jobs and verify workflows recover.

## Phase 5 — QC Pipeline
Deliver:
- QC metrics
- doublets
- QC plots
- QC review gate
- filter confirmation

Exit: human approval signal reliably resumes Temporal.

## Phase 6 — Core Analysis
Deliver:
- normalization
- HVG
- PCA
- neighbors
- UMAP
- Leiden
- marker discovery

Exit: golden datasets pass scientific regression thresholds.

## Phase 7 — Annotation
Deliver:
- human/mouse annotation
- confidence
- manual override
- annotation history

Exit: low-confidence data is never silently forced into confident labels.

## Phase 8 — Experiment Comparison
Deliver:
- conditions
- biological replicates
- comparison builder
- pseudobulk DE
- pathway analysis
- scientific warnings

Exit: sample-level comparison safeguards pass.

## Phase 9 — Results Experience
Deliver:
- interactive UMAP
- QC explorer
- marker tables
- DE tables
- pathways
- artifact downloads
- provenance

Exit: representative large dataset can be explored without downloading entire H5AD to browser memory.

## Phase 10 — Collaboration
Deliver:
- comments
- annotations
- activity
- notifications
- shared projects

## Phase 11 — Billing
Deliver:
- Polar checkout
- subscription projection
- usage ledger
- credits
- quota enforcement
- overage
- customer portal
- webhook reconciliation

Exit: duplicate workflow/webhook events cannot create duplicate charges.

## Phase 12 — Operations
Deliver:
- ops console
- Sentry
- OpenTelemetry
- alerts
- feature flags
- admin tools
- support correlation IDs

## Phase 13 — Security & Governance
Deliver:
- retention
- deletion
- break-glass
- security headers
- rate limits
- MFA policies
- vendor inventory
- incident runbooks
- backup restore tests

## Phase 14 — Production Qualification
Execute:
- security testing
- load testing
- scientific benchmark suite
- DR exercise
- billing reconciliation
- tenant isolation review
- failure injection

Only after critical gates pass does CellWeave move to general production.
