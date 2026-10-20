# CellWeave Phase 3 — 10x Genomics Importers Architecture & Specification

## 1. Scope & Authority
- **Authority**: `CellWeave-Production-Master-System-Design-v1.0.pdf` (Section 121: Scientific Import System, Phase 3).
- **Supported Formats**:
  - `SourceFormat.TENX_H5`: 10x Genomics Filtered Feature Barcode Matrix in HDF5 format (single `PRIMARY` file).
  - `SourceFormat.TENX_MTX`: 10x Genomics Matrix Market triplet (`MATRIX x1`, `FEATURES x1`, `BARCODES x1`).
- **Canonical Policy**: CellWeave Canonical AnnData Policy B (raw unnormalized integer counts stored directly in `X` as SciPy CSR; `layers["counts"]` omitted; slots `obsm`, `varm`, `obsp`, `varp`, and unapproved `uns` keys sanitized).

---

## 2. 10x HDF5 Structure (`TenxH5Importer`)

### Structural Layout
The importer directly inspects the 10x Cell Ranger filtered feature matrix hierarchy via `h5py` without pulling in heavy analysis dependencies:
```text
/matrix/
  ├── data (1D non-negative integer counts)
  ├── indices (1D feature row indices)
  ├── indptr (1D column pointer, length = n_barcodes + 1)
  ├── shape ([n_features, n_barcodes])
  ├── barcodes (1D cell barcodes)
  └── features/
        ├── id (1D stable feature identifiers, e.g. ENSG...)
        ├── name (1D descriptive gene symbols, e.g. TP53)
        └── feature_type (1D modality string, e.g. "Gene Expression")
```

### Invariants & Validation
- **Missing Components**: If `/matrix` or any required dataset is absent, the importer fails with `ScientificIssueCode.MISSING_COMPONENT`.
- **Sparse Representation**: Source matrix is represented in Compressed Sparse Column (CSC) format as `(n_features, n_barcodes)`. Indptr monotonicity and index boundary ranges are strictly checked.
- **Orientation Conversion**: Source `(features x barcodes)` is mandatorily transposed to canonical `(cells x genes) = (barcodes x features)` and converted to SciPy CSR without dense array allocations.
- **Modality Policy**: CellWeave V1 strictly supports `Gene Expression` only. Any occurrence of non-gene modalities (e.g. `Antibody Capture`, `CRISPR Guide Capture`) or mixed modalities fails with `ScientificIssueCode.UNSUPPORTED_MODALITY`. Silent feature subsetting is prohibited.

---

## 3. 10x Matrix Market Triplet Structure (`TenxMtxImporter`)

### Triplet Roles & Cardinality
1. **`ImportFileRole.MATRIX`**:
   - `matrix.mtx` or `matrix.mtx.gz`.
   - Header must specify `%%MatrixMarket matrix coordinate integer general` (or `real general`).
   - Dimension line specifies `n_rows n_cols n_entries` (features x barcodes x nonzero counts).
   - Coordinate lines are 1-based: `row_idx col_idx count_value`.
2. **`ImportFileRole.FEATURES`**:
   - Modern: `features.tsv` or `features.tsv.gz` (3 columns: `feature_id`, `feature_name`, `feature_type`).
   - Legacy: `genes.tsv` or `genes.tsv.gz` (2 columns: `gene_id`, `gene_name`).
3. **`ImportFileRole.BARCODES`**:
   - `barcodes.tsv` or `barcodes.tsv.gz` (1 barcode string per line).

### Invariants & Safety
- **Fail-Closed Gzip**: Decompression of `.gz` files utilizes Step D `open_bounded_gzip_stream` with mandatory `DecompressionLimits` (guards against decompression bombs and unbounded memory).
- **Dimension Assertions**:
  $$\text{n\_source\_rows} = \text{len(features)}$$
  $$\text{n\_source\_columns} = \text{len(barcodes)}$$
  $$\text{canonical n\_obs} = \text{len(barcodes)}, \quad \text{canonical n\_vars} = \text{len(features)}$$
  Any mismatch results in `ScientificIssueCode.DIMENSION_MISMATCH`.
- **Coordinate & Value Safety**: Coordinates outside $[1, n\_rows]$ or $[1, n\_cols]$ fail with `MALFORMED_INPUT`. Negative, NaN, Infinite, or fractional float counts fail with `NON_COUNT_INPUT`. Near-integral ambiguous floats fail with `AMBIGUOUS_COUNT_SEMANTICS`.
- **Barcodes**: Preserved verbatim without stripping suffixes (e.g. `-1`). Duplicates fail with `DUPLICATE_CELL_IDS`. Blank lines fail with `MALFORMED_INPUT`.
- **Features**: Stable feature IDs must be unique. Duplicate stable IDs fail with `AMBIGUOUS_GENE_IDENTIFIERS`. Duplicate gene symbols with distinct stable IDs are permitted (valid biology).

---

## 4. Feature Identity & Organism Candidate Resolution
- **Canonical `var_names`**: Assigned to stable feature IDs (e.g. Ensembl). Descriptive gene symbols are stored in `adata.var["gene_symbol"]`. Original source feature IDs are stored in `adata.var["source_feature_id"]`.
- **Organism Candidate Resolution**: Reuses Step D conservative policy without arbitrary percentage heuristics:
  - 100% Human Ensembl (`ENSG...`) $\rightarrow$ `Organism.HUMAN`, `GeneIdScheme.ENSEMBL`
  - 100% Mouse Ensembl (`ENSMUSG...`) $\rightarrow$ `Organism.MOUSE`, `GeneIdScheme.ENSEMBL`
  - Mixed Human + Mouse Ensembl $\rightarrow$ `ScientificIssueCode.UNRESOLVED_SPECIES` (`ERROR`)
  - Ensembl mixed with symbols or symbol-only $\rightarrow$ unresolved organism, sets `status = InspectionStatus.NEEDS_MAPPING` (`requiresMapping = True`).
- **Reference Build**: 10x files do not contain authoritative reference build metadata; `referenceBuildCandidate` defaults to `None` without guessing from organism or filenames.

---

## 5. Canonicalization & Slot Hygiene
- **Canonical Output**: Standalone `.h5ad` file written to `output_path`.
- **Policy B Compliant Slots**:
  - `adata.X`: SciPy CSR non-negative integer raw counts in `(cells x genes)` orientation.
  - `adata.obs_names`: Cell barcodes.
  - `adata.var_names`: Stable feature IDs.
  - `adata.layers`: Empty (no duplicate `counts` layer).
  - `adata.obsm`, `adata.varm`, `adata.obsp`, `adata.varp`: Empty.
  - `adata.uns`: Contains strictly approved scientific metadata:
    - `uns["organism"]`
    - `uns["gene_id_scheme"]`
    - `uns["reference_build"]` (if provided)
- **Immutability**: Source files are never modified in place; `output_path` matching any source file path is strictly rejected.
- **Canonical Checksum**: Exact SHA-256 of the written `.h5ad` file returned in `CanonicalizationResult.canonicalChecksum`.
