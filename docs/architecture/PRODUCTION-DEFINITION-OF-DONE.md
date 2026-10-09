# Production Definition of Done

CellWeave is production-ready only when all of the following are true:

- A failed worker does not lose an analysis.
- A duplicate request does not duplicate an analysis.
- A duplicate billing event does not duplicate a charge.
- One tenant cannot access another tenant's data.
- Large datasets do not traverse Vercel request bodies unnecessarily.
- A pipeline update never changes old provenance.
- A scientist can reconstruct how a result was produced.
- A customer can export results.
- A customer can delete data according to the designed lifecycle.
- A failed provider can be detected.
- The database can be restored.
- A compute job can be cancelled.
- Scientific regressions block deployment.
- Operations staff can diagnose failures.
- Billing and usage reconcile.
- Every expensive job has a cost ceiling.

Repository-level completion claims must be consistent with this system-level definition.
