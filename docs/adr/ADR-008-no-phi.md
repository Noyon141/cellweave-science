# ADR-008: Prohibition of Protected Health Information (No PHI)

## Status
Accepted

## Context
CellWeave is a Research Use Only (RUO) scientific analysis platform for biotech and CRO researchers, not a clinical diagnostic system or medical record system.

## Decision
CellWeave strictly prohibits uploading Protected Health Information (PHI) or identifiable patient data. Only de-identified research datasets are supported. Logging raw data, patient identifiers, or expression matrices in logs or error traces is strictly forbidden.

## Consequences
- Strict compliance boundaries avoiding HIPAA covered-entity liabilities for V1.
- Mandatory sanitization in all structured logging and diagnostics components.
