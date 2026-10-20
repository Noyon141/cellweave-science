# CellWeave Phase 3 — Generic Matrix Importer Architecture & Specification

## 1. Scope & Authority
- **Authority**: `docs/architecture/CellWeave-Production-Master-System-Design-v1.0.pdf` (Section 121: Scientific Import System, Phase 3).
- **Supported Formats**:
  - `SourceFormat.GENERIC_MATRIX`: Delimited expression matrices (CSV / TSV, plain or bounded `.gz`) and Matrix Market coordinate files (`.mtx` / `.mtx.gz`).
- **Participating Roles & Cardinality**:
  - `ImportFileRole.MATRIX`: exactly 1 file (mandatory).
  - `ImportFileRole.CELL_METADATA`: 0..1 file (optional for CSV/TSV, required for MTX).
  - `ImportFileRole.FEATURE_METADATA`: 0..1 file (optional for CSV/TSV, required for MTX).
  - Extraneous roles (`PRIMARY`, `BARCODES`, `FEATURES`) are strictly prohibited (`INVALID_IMPORT_CARDINALITY`).
- **Canonical Representation**: CellWeave Canonical AnnData Policy B (raw unnormalized integer counts stored strictly in `X` as SciPy CSR; `layers` empty; `obsm`, `varm`, `obsp`, `varp` empty; minimal controlled provenance in `uns`).

---

## 2. Content-Based Format & Delimiter Detection
- **No Extension-Authoritative Decisions**: Parsing semantics and delimiter resolution are inferred strictly from stream contents, never from `.csv` or `.tsv` filename extensions.
- **Matrix Market**: Detected via leading magic bytes `%%MatrixMarket` in plain or bounded gzip chunks.
- **Delimiter Sniffing (`sniff_delimiter`)**:
  - Lines containing commas and 0 tabs resolve as comma-delimited (`,`).
  - Lines containing tabs and 0 commas resolve as tab-delimited (`\t`).
  - Mixed lines require a $\ge 2:1$ predominance ratio; otherwise, fails closed with `ScientificIssueCode.FORMAT_MISMATCH`.
  - Undelimited single-column files (0 commas and 0 tabs across lines) resolve deterministically to a single-column table independent of filename extension.
  - An expression matrix file must contain at least 2 columns (identifier column + expression columns); single-column matrices fail closed with `ScientificIssueCode.MALFORMED_INPUT`.
- **Bounded Decompression Safety**: All gzip reading streams through Step D `open_bounded_gzip_stream` governed by explicit `DecompressionLimits`. Bypassing decompression limits via unconstrained readers is forbidden.

---

## 3. Matrix Identity Model & Orientation Authority
Orientation is **never** guessed from dimension heuristics, filenames, or identifier patterns.

| Orientation | Cell Identity Authority | Feature Identity Authority | Transpose Behavior |
| :--- | :--- | :--- | :--- |
| **`CELLS_BY_GENES`** | CSV/TSV: Row labels (column 0 or `mapping.cellIdColumn`).<br>MTX: Rows ($1 \dots M$) mapped to `CELL_METADATA`. | CSV/TSV: Column headers ($1 \dots N$).<br>MTX: Columns ($1 \dots N$) mapped to `FEATURE_METADATA`. | None (source is already cells $\times$ genes). |
| **`GENES_BY_CELLS`** | CSV/TSV: Column headers ($1 \dots N$).<br>MTX: Columns ($1 \dots N$) mapped to `CELL_METADATA`. | CSV/TSV: Row labels (column 0 or `mapping.geneIdColumn`).<br>MTX: Rows ($1 \dots M$) mapped to `FEATURE_METADATA`. | Explicit transpose: $(genes \times cells) \to (cells \times genes)$ via SciPy CSR `.T`. |

- **Prohibition on Fabricated IDs**: Fallback synthetic identifiers (`cell_1`, `row_1`, `col_1`, or integer ordinal sequences) are strictly prohibited unless those literal strings were the authoritative input values.
- **Inspection Invariant**: Because orientation cannot be assumed, `inspect()` reports `InspectionStatus.NEEDS_MAPPING` with `requiresMapping=True` and `ScientificIssueCode.AMBIGUOUS_ORIENTATION` until explicitly confirmed by the user.

---

## 4. Contract Field Semantics: `cellIdColumn`, `geneIdColumn`, `geneSymbolColumn`

### Cell Identity (`cellIdColumn`)
- In `CELL_METADATA`: If specified, `cellIdColumn` must exist as a column in the metadata table.
- Values must be non-empty strings and unique (`DUPLICATE_CELL_IDS`).
- Every cell present in the matrix must exist in `CELL_METADATA` (`MISSING_REQUIRED_METADATA`).
- Joins are performed strictly by value, not by row position.

### Feature Identity (`geneIdColumn`)
- In `FEATURE_METADATA`: Resolves the stable feature identifier column (e.g. Ensembl gene IDs).
- When `geneIdColumn` is distinct from the source join key, it supplies the canonical `var_names`.
- All stable IDs must be unique; duplicate stable IDs fail with `AMBIGUOUS_GENE_IDENTIFIERS`.

### Feature Description (`geneSymbolColumn`)
- In `FEATURE_METADATA`: Resolves descriptive gene symbols (e.g. HGNC symbols).
- Supplies `adata.var["gene_symbol"]`.
- When stable IDs are unique, duplicate symbols across different stable IDs are valid biology and preserved without mangling or `make_unique()`.

---

## 5. Metadata Join & Extra-Row Policy
- **Deterministic Value Join**: Metadata tables are joined on explicit key values. Row ordering in `CELL_METADATA` or `FEATURE_METADATA` does not matter; tables are realigned deterministically to match matrix order.
- **Extra Metadata Rows**: Extra unique metadata rows in `CELL_METADATA` or `FEATURE_METADATA` that are not referenced by the expression matrix are safely ignored (filtered out). Every matrix cell/feature must resolve exactly once.
- **Missing Matrix Keys**: If any cell or feature present in the matrix is missing from the required metadata table, the importer fails closed with `ScientificIssueCode.MISSING_REQUIRED_METADATA`.
- **Positional Joins Prohibited**: Falling back to row/column index positions when names do not match is strictly forbidden.

---

## 6. Sample Mapping Authority (CellWeave PostgreSQL Truth)
- **Zero Pass-Through**: Source-side values in `sampleColumn` never pass through automatically as canonical `sample_key`.
- **Complete Mapping Invariant**: If `sampleColumn` is specified, every unique source sample value present in the data must be explicitly mapped to an approved `SampleMappingEntry.sampleKey`. Missing, unmapped, or blank targets fail closed with `MISSING_REQUIRED_METADATA`.
- **No Mapping Without Column**: Providing `sampleMappings` without a corresponding `sampleColumn` fails closed.
- **No Global Single-Sample Guessing**: Assigning a global sample key without an explicit source column and mapping is prohibited.

---

## 7. Biological Annotation Mappings
- **Clinical/Experimental Fields**:
  - `conditionColumn` $\to$ `adata.obs["condition"]`
  - `batchColumn` $\to$ `adata.obs["batch"]`
  - `biologicalReplicateColumn` $\to$ `adata.obs["biological_replicate"]`
  - `technicalReplicateColumn` $\to$ `adata.obs["technical_replicate"]`
- **Fail-Closed on Missing Request**: Requesting a mapping column that does not exist in `CELL_METADATA` fails closed with `MISSING_REQUIRED_METADATA`.
- **Exclusion of Unmapped Columns**: Source metadata columns that are not explicitly mapped are never copied into canonical `obs`.

---

## 8. Canonical Slot Schemas

### Canonical `obs` Schema
- **Index**: `cell_id` (sanitized unique cell identifiers).
- **Columns**: Strictly limited to mapped fields:
  ```python
  ["sample_key", "condition", "batch", "biological_replicate", "technical_replicate"]
  ```
- **Minimal Mapping**: If no metadata columns are mapped, `obs` contains 0 optional columns (`len(adata.obs.columns) == 0`).

### Canonical `var` Schema
- **Index**: `gene_id` (canonical stable identifiers).
- **Columns**:
  - `source_feature_id`: Always present (original feature token from source matrix).
  - `gene_symbol`: Present **only** when `geneSymbolColumn` is explicitly mapped and resolved.
- **Minimal Schema**: When no symbol column is mapped, `var` contains only `["source_feature_id"]`. Arbitrary feature metadata columns are not copied into `var`.

---

## 9. Organism, Gene ID Scheme, & Reference Build Policy
- **Step D Organism Policy Reused**:
  - 100% Human Ensembl IDs $\implies$ `Organism.HUMAN` candidate.
  - 100% Mouse Ensembl IDs $\implies$ `Organism.MOUSE` candidate.
  - Human + Mouse mixture $\implies$ `InspectionStatus.INVALID` with `ScientificIssueCode.UNRESOLVED_SPECIES`.
  - Ensembl mixed with symbols $\implies$ candidate unresolved (`None`).
  - Symbol-only identifiers $\implies$ candidate unresolved (`None`).
- **Gene ID Scheme**:
  - `GeneIdScheme.ENSEMBL`: Confirmed Ensembl identifier space.
  - `GeneIdScheme.SYMBOL`: Explicit confirmed unique symbol identifiers. Duplicate symbols under symbol scheme fail closed with `AMBIGUOUS_GENE_IDENTIFIERS`.
- **Reference Build**:
  - Preserved verbatim only from explicit user mapping (`mapping.referenceBuild`).
  - Never inferred or guessed from organism, Ensembl prefix, or file name. If omitted, canonical AnnData has no `reference_build` in `uns`.

---

## 10. Memory & Performance Limitations
- Delimited text expression matrices (CSV/TSV) require line-by-line streaming during inspection and bounded chunk parsing during canonicalization.
- For extremely large datasets ($>100\text{k}$ cells), CSV/TSV text serialization introduces significant parsing overhead compared to binary HDF5 (`.h5ad` / 10x H5).
- Resource limits are strictly enforced via `GenericMatrixConfig.decompression_limits`.
