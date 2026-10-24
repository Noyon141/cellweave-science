# Importer Security, Checksum & Validation Primitives (Phase 3 Step D)

## Overview
This document specifies the shared scientific, cryptographic, and security primitives used by all CellWeave Phase 3 dataset importers. Concrete importers (H5AD, 10x H5, 10x MTX, Generic Matrix) are built on top of these base abstractions in Steps E, F, and G.

## Phase 3 exit correction: exact identity and metadata policy

`validate_identifiers()` accepts only existing, nonempty strings without surrounding
whitespace. It rejects nonstrings and whitespace rather than stringifying or
trimming them. Accepted values retain their exact case, Unicode, punctuation,
Ensembl version, and barcode suffix. Duplicate canonical cell/stable feature IDs
remain invalid. Only descriptive/source values explicitly permitted to repeat
(for example gene symbols) use `require_unique=False`.

All importer authority paths apply this policy before assigning canonical axes or
`source_feature_id`. HDF5 byte strings are decoded as strict UTF-8, not converted
from numeric values. Gzip text uses incremental UTF-8 decoding so multibyte
identifiers survive chunk boundaries. Generic text readers preserve physical
string tokens; numeric MTX axis coordinate parsing remains a separate exception.
Explicit metadata joins use exact source tokens without normalization.

Sample source values and mapping `sourceValue` match exactly. Registered
`sampleKey` is validated and retained, never trimmed or invented. A source value
with whitespace cannot silently match a trimmed mapping value. Scientific engine
validation does not replace Postgres as registered sample authority.

`validate_canonical_anndata()` independently rejects extra metadata and never
mutates or sanitizes the object:

- `uns`: only `organism`, `gene_id_scheme`, and optional `reference_build`.
- `obs`: only optional `sample_key`, `condition`, `batch`,
  `biological_replicate`, and `technical_replicate` columns.
- `var`: only `source_feature_id` and optional `gene_symbol` columns.

Existing format-neutral canonical builder callers may omit `source_feature_id`;
importer outputs retain it. When present, its values must satisfy exact string
validation. Existing raw-count/CSR, unique-axis, empty analysis slot, and absent
`raw` checks remain in force. Publication invokes this validator before any
PUT/GET, so prohibited local metadata cannot reach publication transport.

### PHASE 3 REFERENCE-DATA QUALIFICATION GAP

The seven external/reference entries in `reference_data/golden_manifest.json`
still have placeholder-style hashes and are not executed by the manifest harness.
This correction does not change those entries, download datasets, or qualify them.
Passing importer goldens refers to the existing synthetic importer fixtures only.
The subsequent Phase 3 exit audit must retain this qualification gap.

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
  - Supports configurable `DecompressionLimits(max_uncompressed_bytes, max_compression_ratio)`.
  - Distinguishes technical stream safety from business upload policy: Phase 2 owns commercial upload size limits, while Step D primitives guard against stream decompression bombs.
  - Malformed or corrupt compressed streams raise `MalformedCompressedStreamError`.

---

## 4. Path & Symlink Security (`validate_trusted_local_path`)
- All local inputs are verified to be existing, regular files.
- Rejects directories with `ImporterSecurityError`.
- Rejects symbolic links by default to prevent arbitrary file access or escape.
- Rejects path strings containing null bytes (`\x00`).

---

## 5. Raw Count Classification & Validation Model (`classify_count_matrix`, `validate_count_matrix`)
- Expressed via internal `CountClassification` enum:
  - `VALID_COUNTS`:
    - Non-negative, finite integer dtypes (`np.integer`).
    - Non-negative, finite floating dtypes where all values equal their exact mathematical integer conversion (e.g. `0.0`, `1.0`).
  - `AMBIGUOUS`:
    - Floating-point values with tiny non-zero differences within machine epsilon limits ($10 \times \epsilon$). They are not silently accepted as valid raw counts.
    - Yields `CountValidationError(issue_code=AMBIGUOUS_COUNT_SEMANTICS)`.
  - `NOT_COUNTS`:
    - Matrices containing `NaN`, `+Inf`, `-Inf`, negative numbers, or clearly fractional continuous values (e.g. log-normalized).
    - Yields `CountValidationError(issue_code=NON_COUNT_INPUT)`.
- **Sparse Safety**: Evaluates directly on `matrix.data` without calling `.toarray()` or densifying into memory.

---

## 6. Identifier Validation (`validate_identifiers`)
- Validates cell and gene identifiers:
  - Non-empty, non-whitespace strings.
  - Unique within the dataset.
  - Stable ordering preserved.
  - Duplicate cell IDs are rejected (`ScientificIssueCode.DUPLICATE_CELL_IDS`). No silent suffixing (`-1`, `-2`).

---

## 7. Deterministic Ensembl Candidate & Organism Detection (`detect_ensembl_organism_candidates`)
- Consistently human: All identifiers match `^ENSG\d+(\.\d+)?$` $\rightarrow$ `Organism.HUMAN`.
- Consistently mouse: All identifiers match `^ENSMUSG\d+(\.\d+)?$` $\rightarrow$ `Organism.MOUSE`.
- Mixed prefixes (human + mouse) $\rightarrow$ `None` (unresolved).
- Mixed prefixes with gene symbols or random IDs $\rightarrow$ `None` (unresolved).
- Symbol-only or empty identifiers $\rightarrow$ `None` (unresolved).
- Version suffix (e.g. `.7`, `.15`) is tolerated solely for prefix detection; identifiers are not stripped or altered.
- Zero arbitrary or probabilistic thresholds are employed.

---

## 8. Role Cardinality Validation (`validate_import_source_set_cardinality`)
- Structural validation for `ImportSourceSet`:
  - `H5AD`: Exactly 1 `PRIMARY`.
  - `TENX_H5`: Exactly 1 `PRIMARY`.
  - `TENX_MTX`: Exactly 1 `MATRIX`, 1 `FEATURES`, 1 `BARCODES` (3 files total).
  - `GENERIC_MATRIX`: Exactly 1 `MATRIX`, 0..1 `CELL_METADATA`, 0..1 `FEATURE_METADATA`.
- Unrelated roles or missing mandatory files yield `ScientificIssue` with `INVALID_IMPORT_CARDINALITY`, `severity=ERROR`, `userActionRequired=True`.

---

## 9. Canonical AnnData Policy B (`build_canonical_anndata`, `validate_canonical_anndata`)
- **Canonical Orientation**: strictly `cells × genes` (`n_obs × n_vars`).
- **Slot Hygiene**:
  - `adata.X`: SciPy CSR sparse matrix containing raw, unnormalized counts.
  - `adata.layers`: Empty. Raw counts are NOT duplicated into `adata.layers["counts"]` during Phase 3.
  - `adata.obsm`, `adata.varm`, `adata.obsp`, `adata.varp`: Empty.
  - `adata.uns`: Minimal controlled metadata (`organism`, `gene_id_scheme`, `reference_build`).
- Validated via `validate_canonical_anndata` before serialization.
