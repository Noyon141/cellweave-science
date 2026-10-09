# System Design → Agent Environment Coverage Map

This map prevents sections of the 167-part design from disappearing during implementation.

| Design area | Primary environment owner |
|---|---|
| Product definition, promise, scope, non-goals, expansion, personas | all repository `AGENTS.md`; shared canon |
| Authorization, core journey | web rules; orchestrator signals; science QC gate |
| High-level architecture, repository architecture | all `architecture-boundaries` rules |
| Next.js architecture, API, auth | web |
| Upload, integrity, security, importers | web + science + orchestrator |
| Canonical AnnData and scientific pipeline | science |
| QC review gate | science + orchestrator + web |
| Normalization/HVG/PCA/neighbors/UMAP/Leiden | science |
| Annotation / advanced ML | science |
| Markers / DE / pathways | science + contracts |
| Experiment model / comparison | web + science + contracts |
| Collaboration | web |
| Reproducibility / lineage / selective recompute | science + orchestrator + contracts |
| Temporal / run states / retry | orchestrator |
| Compute provider / resources / cost guardrails | orchestrator + science |
| Storage / object access | web + infra |
| Database entities / IDs / immutability | web + contracts |
| Tenant isolation / RLS / worker permissions / service auth | web + orchestrator + infra |
| Polar billing / usage / entitlements / webhook | web + orchestrator + contracts |
| Visualization / results / dashboard / UX / errors | web |
| Notifications | web |
| Ops / break-glass / diagnostics | web + infra |
| Exports / retention / deletion | web + orchestrator + science + infra |
| Security / RUO / SOC2 / secrets | all |
| Observability / metrics / SLO / DR / backup | infra + all repo telemetry rules |
| Environments / previews / isolation | infra + all |
| CI/CD | infra plus repo-specific test rules |
| Scientific regression / golden datasets / releases | science |
| DB migrations / API compatibility / idempotency | web + contracts + orchestrator |
| Reconciliation | orchestrator |
| Next.js perf / animations / responsive / accessibility | web |
| Abuse / threat model / error classification / warnings | all relevant repos |
| Governance / feature flags / future import adapters | web + science + infra |
| Provider interfaces | owning repo |
| Public API future | web + contracts |
| Runbooks / severity / support / admin metrics | infra + web |
| Scalability / cost / caching / cancellation / quota | orchestrator + infra + web |
| Email / analytics / privacy / V2 AI boundary | web |
| TS / Python quality / logging / local dev / staging | repo-specific |
| Production gates / load / data size / performance | infra + web + science |
| Search / reports / billing failure / suspension | web |
| Enterprise future | architecture rules; no V1 implementation unless explicitly authorized |
| ADRs / docs / implementation order / DoD | all |
| Final tech/deployment baseline / four boundaries | all |
