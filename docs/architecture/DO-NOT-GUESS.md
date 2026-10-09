# Do Not Guess — Design Inputs Not Selected by the Authoritative Blueprint

The authoritative design deliberately fixes many architecture decisions but does not select every implementation detail. Antigravity must not silently turn an unspecified detail into architecture.

When one of the following is absent from the existing repository, stop the design decision at the boundary and report it as unresolved rather than inventing a permanent choice.

## Explicitly not fixed by the design

- JavaScript package manager.
- Python package/environment manager.
- Exact always-on hosting provider for the initial Temporal Worker.
- Exact enterprise SSO implementation/provider wiring beyond readiness for OIDC/SAML and examples.
- Exact MFA implementation/provider.
- Exact feature-flag vendor.
- Exact centralized log backend.
- Exact product-analytics vendor.
- Exact rate-limit numbers.
- Exact upload/file/cell commercial limits; the design says benchmarks determine plan limits.
- Exact plan prices and exact credit conversion formula.
- Exact retention-day values by plan.
- Exact service-to-service credential technology; only the security properties are fixed.
- Exact annotation model/provider/tool for production beyond the required abstraction, metadata, confidence behavior, and human review.
- Exact differential-expression tool implementation beyond the scientific policy and validated-tool requirement.
- Exact pathway/enrichment provider/database beyond the provider abstraction, versioning, multiple-testing metadata, and licensing review.
- Exact reference genome/build policy when not available from data.
- Exact customer-facing support tooling.
- Exact incident alerting vendor.
- Exact report-generation library.
- Exact RLS policy SQL shape before schema exists.
- Exact worker task-queue names beyond the conceptual stage-2 classes.
- Exact cost thresholds/resource ceilings before benchmarking and plan definition.
- Exact cache implementation.
- Exact API-token implementation timing because public API auth is future.
- Exact future Seurat/Parse/spatial implementation details.
- Exact future AI/LLM provider; AI interpretation is V2 and not part of V1 scientific truth.

## Agent behavior

If an implementation task requires one of these decisions:
1. inspect the repository for an already-authorized choice;
2. if present, preserve it if it does not violate the design;
3. if absent, do not introduce a new architectural dependency as if it were canonical;
4. implement only the abstract boundary or scaffolding that the design explicitly authorizes;
5. clearly report the unresolved design input.

Never use “best practice” as permission to expand CellWeave.
