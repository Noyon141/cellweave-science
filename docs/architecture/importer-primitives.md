# Importer Security, Checksum & Validation Primitives (Phase 3 Step D)

## Overview
This document specifies the shared scientific, cryptographic, and security primitives used by all CellWeave Phase 3 dataset importers. Concrete importers (H5AD, 10x H5, 10x MTX, Generic Matrix) are built on top of these base abstractions in Steps E, F, and G.

---

## 1. Cryptographic Checksum Algorithm
Importers calculate two distinct checksum layers:

### A. Per-File Streaming SHA-256 (`compute_file_sha256`)
- Implemented in `cellweave_science.importers.checksum`.
- Streams files in bounded chunks (default 1 MiB) directly into Python's standard `hashlib.sha256`.
- Bounded memory footprint ($O(1)$ RAM), preventing Out-Of-Memory (OOM) failures on massive matrix files.
- Returns lowercase 64-character hexadecimal string.

### B. Multi-File Aggregate `sourceChecksum` (`compute_source_checksum`)
- **Single-File Formats (`H5AD`, `TENX_H5`)**:
  - Requires exactly 1 file with role `PRIMARY`.
  - `sourceChecksum` = exact per-file SHA-256.
- **Multi-File Formats (`TENX_MTX`, `GENERIC_MATRIX`)**:
  - Rejects duplicate persisted filenames.
  - Formats each member file as:
    ```
    {persistedFileName}\t{byteSize}\t{sha256}\n
    ```
    with UTF-8 encoding and LF (`\n`) line terminator.
  - Sorts file descriptors **lexicographically ascending by exact persisted filename** (case-sensitive).
  - `sourceChecksum` = `SHA-256(concatenated_manifest_bytes)`.
  - Guarantees order-independent, deterministic checksums across all environments.

---

## 2. Physical Inspection Model (`inspect_physical_file`)
- Model: `PhysicalInspectionResult` in `cellweave_science.importers.security`.
- Attributes: `sha256`, `byte_size`, `compression`, `detected_format`.
- Strictly isolated: Never contains database IDs, organization IDs, project IDs, or cloud storage credentials.
- Feeds trusted provenance registration without coupling to infrastructure layers.

---

## 3. Compression Policy & Magic-Byte Detection
- Supported compression values: `NONE`, `GZIP`.
- Detection is based on binary signatures (magic bytes), not file extensions:
  - `GZIP`: `1F 8B`
  - `HDF5`: `\x89HDF\r\n\x1a\n`
  - `MatrixMarket`: `%%MatrixMarket`
- Container distinction:
  - Both `H5AD` and `TENX_H5` share the HDF5 container signature. Physical inspection identifies `HDF5`; format-specific semantic qualification occurs in subsequent importer steps.
- Bounded decompression (`open_bounded_gzip_stream`):
  - Streams decompressed chunks (64 KiB) without whole-file buffering.
  - Enforces `max_uncompressed_bytes` (default 50 GiB) and `max_compression_ratio` (default 250.0) to prevent decompression bombs.

---

## 4. Path & Symlink Security (`validate_trusted_local_path`)
- All local inputs are verified to be existing, regular files.
- Rejects symbolic links by default to prevent arbitrary file access or escape.
- Rejects path strings containing null bytes (`\x00`).

---

## 5. Raw Count Validation Primitive (`validate_count_matrix`)
- Validates that matrix values qualify as raw, unnormalized counts:
  1. **Finite**: No `NaN`, `+Inf`, `-Inf`.
  2. **Non-negative**: Values must be $\ge 0$.
  3. **Integral**:
     - Exact integer dtypes (`np.integer`) are accepted.
     - Floating-point dtypes (`np.floating`) are accepted **only** if every non-zero value satisfies $|x - \text{round}(x)| \le 10^{-5}$ (`FLOAT_INTEGER_TOLERANCE`). Normalized, log-transformed, or continuous data is strictly rejected with `CountValidationError(issue_code=NON_COUNT_INPUT)`.
  4. **Sparse Safety**: Evaluates directly on `matrix.data` without calling `.toarray()` or densifying into memory.

---

## 6. Identifier Validation (`validate_identifiers`)
- Validates cell and gene identifiers:
  - Non-empty, non-whitespace strings.
  - Unique within the dataset.
  - Stable ordering preserved.
  - Duplicate cell IDs are rejected (`ScientificIssueCode.DUPLICATE_CELL_IDS`). No silent suffixing (`-1`, `-2`).

---

## 7. Ensembl Candidate & Organism Detection (`detect_ensembl_organism_candidates`)
- Human candidate: `^ENSG\d+(\.\d+)?$`
- Mouse candidate: `^ENSMUSG\d+(\.\d+)?$`
- If $\ge 50\%$ match human and 0% match mouse: candidate is `Organism.HUMAN`.
- If $\ge 50\%$ match mouse and 0% match human: candidate is `Organism.MOUSE`.
- Mixed or gene symbol identifiers return `None` (unresolved). Organism is never guessed from file names or gene symbols alone.

---

## 8. Role Cardinality Validation (`validate_import_source_set_cardinality`)
- Structural validation for `ImportSourceSet`:
  - `H5AD`: Exactly 1 `PRIMARY`.
  - `TENX_H5`: Exactly 1 `PRIMARY`.
  - `TENX_MTX`: Exactly 1 `MATRIX`, 1 `FEATURES`, 1 `BARCODES` (3 files total).
  - `GENERIC_MATRIX`: Exactly 1 `MATRIX`, 0..1 `CELL_METADATA`, 0..1 `FEATURE_METADATA`.
- Unrelated roles or missing mandatory files yield `ScientificIssue` with `INVALID_IMPORT_CARDINALITY`.

---

## 9. Canonical AnnData Policy B (`build_canonical_anndata`, `validate_canonical_anndata`)
- **Canonical Orientation**: strictly `cells × genes` (`n_obs × n_vars`).
- **Slot Hygiene**:
  - `adata.X`: SciPy CSR sparse matrix containing raw, unnormalized counts.
  - `adata.layers`: Empty. Raw counts are NOT duplicated into `adata.layers["counts"]` during Phase 3.
  - `adata.obsm`, `adata.varm`, `adata.obsp`, `adata.varp`: Empty.
  - `adata.uns`: Minimal controlled metadata (`organism`, `gene_id_scheme`, `reference_build`).
- Validated via `validate_canonical_anndata` before serialization.
