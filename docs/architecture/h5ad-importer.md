# CellWeave Phase 3 — H5AD Importer Architecture & Specification

## 1. Scope & Authority
- **Source of Truth**: `CellWeave-Production-Master-System-Design-v1.0.pdf` (Section 121: Scientific Import System, Phase 3).
- **Format Target**: `SourceFormat.H5AD` (single `PRIMARY` file input cardinality).
- **Scientific Policy**: CellWeave Canonical AnnData Policy B (raw unnormalized integer counts stored directly in `X` as SciPy CSR; `layers["counts"]` is omitted; slots `obsm`, `varm`, and non-essential `uns` keys are sanitized).

---

## 2. Importer Lifecycle & Protocol Implementation

`H5ADImporter` implements the `DatasetImporter` protocol:

### A. `detect(source_set: ImportSourceSet) -> bool`
1. Verifies `source_set.source_format == SourceFormat.H5AD`.
2. Enforces single-file cardinality: exactly 1 file assigned to role `ImportFileRole.PRIMARY`.
3. Performs physical magic-byte sniffing: verifies the leading 8 bytes match the HDF5 signature (`\x89HDF\r\n\x1a\n`).

### B. `validate(source_set: ImportSourceSet) -> list[ScientificIssue]`
1. Validates trusted storage path and file existence.
2. Checks physical container format.
3. Opens AnnData safely with warning suppression to prevent internal duplicate-name warnings from escaping as unhandled errors.
4. Checks dimensions: `n_obs > 0` and `n_vars > 0`.
5. Validates cell identifiers (`obs_names`): rejects duplicates with `DUPLICATE_CELL_IDS` and whitespace/empty identifiers with `MALFORMED_INPUT`.
6. Validates raw count matrix availability:
   - Scans `X`, `layers["counts"]`, and `adata.raw.X`.
   - Rejects non-count expression data (normalized/log-transformed floats, negative values) with `NON_COUNT_INPUT`.
   - Rejects conflicting count candidates with `AMBIGUOUS_COUNT_SEMANTICS`.
7. Validates feature identifiers:
   - Checks `var_names` and candidate stable ID columns (`gene_ids`, `gene_id`, `ensembl_id`, `ensembl`).
   - If `var_names` contains duplicate gene symbols and no alternate stable unique identifier column exists, rejects with `AMBIGUOUS_GENE_IDENTIFIERS`.

### C. `inspect(source_set: ImportSourceSet) -> ImportInspectionResult`
1. If validation finds blocking errors, returns `InspectionStatus.INVALID` with the diagnostic issues.
2. Computes dataset dimensions (`cellCountEstimate`, `geneCountEstimate`).
3. Enumerates available column names in `obs` and `var`.
4. Detects gene identifier scheme:
   - Uses conservative Step D `detect_ensembl_organism_candidates` requiring 100% consistent Ensembl IDs with 0 non-Ensembl and 0 opposing species IDs (no arbitrary percentage threshold).
   - Distinguishes Human (`ENSG...`) vs Mouse (`ENSMUSG...`) Ensembl IDs.
   - Detects gene symbols when stable IDs are not Ensembl.
5. Resolves organism:
   - Inspects explicit metadata in `uns` (`organism`, `species`, `taxon_id`) using exact normalized values (no fuzzy substring matching).
     - Accepted Human: `"human"`, `"homo sapiens"`, `"homo_sapiens"`, `"9606"`.
     - Accepted Mouse: `"mouse"`, `"mus musculus"`, `"mus_musculus"`, `"10090"`.
     - Explicit unsupported values (e.g. `"danio rerio"`, `"rat"`, `"humanized mouse"`, `"mouse-like"`) reject with `UNSUPPORTED_SPECIES`.
   - Uses Step D helpers (`check_ensembl_species_conflict`, `has_opposing_ensembl_species`) to enforce consistent Ensembl ID evidence.
   - Rejects mixtures of Human and Mouse Ensembl IDs with `UNRESOLVED_SPECIES` (`ERROR`).
   - Rejects conflicts between explicit metadata and opposing Ensembl IDs with `UNRESOLVED_SPECIES` (`ERROR`).
   - If species cannot be conclusively resolved without ambiguity (e.g. symbol-only with no metadata, or 80% Ensembl + 20% symbols), sets `detectedOrganism=None` and returns `InspectionStatus.NEEDS_MAPPING` with `requiresMapping=True`.
6. Inspects reference build from `uns` (`reference_build`, `genome`, `reference_genome`, `assembly`).

### D. `canonicalize(...) -> CanonicalizationResult`
1. Preflight validation & path security checks:
   - Enforces immutability: `output_path` must not match `source_path`.
   - Validates parent directory existence.
2. Extracts and verifies raw counts into SciPy CSR (`int32` or `int64` non-negative integer representation).
3. If counts come from `adata.raw.X`, reorders and filters to match the target features.
4. Constructs clean `var` table with `gene_id` (index) and optional `gene_symbol` column.
5. Constructs clean `obs` table:
   - Preserves user metadata columns while lowercasing/sanitizing internal keys.
   - Applies explicit `MetadataMappingSubmission` column assignments for `sample_key`, `condition`, `batch`, `biological_replicate`, `technical_replicate`.
   - Applies explicit sample value mappings if provided.
   - Generates deterministic default `SMP_001` if no sample column is available or mapped.
6. Builds new canonical AnnData via `build_canonical_anndata`:
   - Enforces Policy B: `X` has raw counts, `layers` does not contain redundant raw counts.
   - Slot sanitization: omits unvetted `obsm`, `varm`, `obsp`, `varp`.
   - Canonical `uns` schema stores strictly approved scientific metadata:
     - `organism` (authoritative scientific field)
     - `gene_id_scheme` (authoritative scientific field)
     - `reference_build` (authoritative scientific field, if present)
   - Source file SHA-256 is NOT embedded in `uns["cellweave"]` (source checksum authority belongs to persistence).
   - Source `uns` analysis metadata is NOT copied wholesale.
7. Validates canonical AnnData with `validate_canonical_anndata`.
8. Writes atomic `.h5ad` artifact to destination path and returns `CanonicalizationResult` with canonical artifact SHA-256 and byte size.

---

## 3. Golden Test Fixtures (G1–G10)

The implementation is verified against the complete Phase 3 golden dataset matrix:
- **G1**: Valid Human Ensembl counts in `X` $\rightarrow$ Valid, auto-detected.
- **G2**: Valid Mouse Ensembl counts in `X` $\rightarrow$ Valid, auto-detected.
- **G3**: Valid counts in `layers["counts"]` with normalized `X` $\rightarrow$ Resolves `layers["counts"]`.
- **G4**: Valid counts in `adata.raw.X` with normalized `X` $\rightarrow$ Resolves `raw.X`.
- **G5**: Normalized expression only (no raw counts anywhere) $\rightarrow$ Rejected (`NON_COUNT_INPUT`).
- **G6**: Duplicate `obs_names` $\rightarrow$ Rejected (`DUPLICATE_CELL_IDS`).
- **G7**: Stable Ensembl `var_names` with duplicate descriptive `gene_symbol` column $\rightarrow$ Valid.
- **G8**: Duplicate symbol-only `var_names` without alternate stable ID $\rightarrow$ Rejected (`AMBIGUOUS_GENE_IDENTIFIERS`).
- **G9**: Symbol-only dataset without explicit metadata $\rightarrow$ `NEEDS_MAPPING`.
- **G10**: Conflicting organism metadata (e.g. Mouse metadata with Human Ensembl IDs) $\rightarrow$ Rejected (`UNRESOLVED_SPECIES`).
