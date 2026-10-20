# Canonical Artifact Publication (Phase 3 Step H.2)

## 1. Capability-Based Publication Boundary

`cellweave_science.publication` (and `cellweave_science.importers.publication`) implements the science-side canonical artifact publisher (`publish_canonical_artifact`).

Architectural invariants:
- `cellweave-science` holds **no** Cloudflare R2 master credentials, bucket/account environment variables, or `boto3` dependencies.
- `cellweave-science` holds **no** `organizationId` or `projectId` storage authority and never constructs canonical R2 object keys.
- `cellweave-science` consumes a validated local `canonical.h5ad` path and a scoped `CanonicalArtifactPublicationCapabilities` contract issued by the trusted control plane (`cellweave-web`).

---

## 2. Local Canonical Artifact Pre-Publication Validation

Before any network request is attempted, `validate_local_canonical_h5ad_file` verifies:
1. The path exists, is a regular file, and is not a symbolic link (`validate_trusted_local_path(..., allow_symlinks=False)`).
2. Physical file size is strictly greater than `0`.
3. The file is uncompressed (`CompressionFormat.NONE`) and starts with the 8-byte HDF5 magic signature (`\x89HDF\r\n\x1a\n`).
4. The file opens cleanly via `anndata.read_h5ad`.
5. The `AnnData` object passes `validate_canonical_anndata(adata)`:
   - Policy B: raw integer counts stored in CSR sparse `adata.X` (`n_obs > 0`, `n_vars > 0`).
   - Slot hygiene: `adata.layers`, `adata.obsm`, `adata.varm`, `adata.obsp`, and `adata.varp` are empty, and `adata.raw` is `None`.
   - `obs_names` and `var_names` are unique, non-empty string identifiers.
6. `adata.uns["organism"]` and `adata.uns["gene_id_scheme"]` are present and match supported `Organism` and `GeneIdScheme` enums.

---

## 3. Conditional `PUT` & Signed Header Enforcement

`validate_publication_capabilities` and `HttpsCapabilityTransport` enforce:
- `capabilities.datasetVersionId` matches `expected_dataset_version_id` (when provided).
- `capabilities.expiresAt` is a valid timezone-aware UTC timestamp in the future (`CanonicalArtifactCapabilityExpiredError` on expiry).
- Both `putUrl` and `getUrl` use the `https://` scheme and contain no embedded user credentials.
- `capabilities.requiredPutHeaders` contains `"If-None-Match": "*"`.
- Upload bytes are streamed directly from the local file handle in bounded chunks (`DEFAULT_CHUNK_SIZE_BYTES = 1 MiB`) with an explicit `Content-Length` header, avoiding full-file memory buffering.
- HTTP redirects (`301`, `302`, `303`, `307`, `308`) are strictly rejected with `CanonicalArtifactTransportError`.

---

## 4. Read-After-Write Streaming `GET` Verification & Local Mutation Guard

After `PUT` completes (`200/201/204` or `412 Precondition Failed`):
1. `HttpsCapabilityTransport.get_verify_stream` streams the remote object from `capabilities.getUrl` in bounded chunks and computes `remote_sha256` and `remote_byte_size` directly from the byte stream.
2. HTTP `ETag`, `Last-Modified`, `Content-Type`, and custom object metadata headers are never trusted as checksum authority.
3. If `remote_byte_size != local_byte_size` or `remote_sha256 != local_sha256`, `publish_canonical_artifact` raises `CanonicalArtifactIntegrityError`.
4. `LocalFileStatSnapshot` (`byte_size`, `mtime_ns`, `ctime_ns`) and `compute_file_sha256` are checked before upload, after `PUT`, and after `GET` verification. Any concurrent local file modification raises `CanonicalArtifactIntegrityError`.

---

## 5. Idempotent Retry vs Integrity Conflict Semantics

- **`PUBLISHED`**: Returned when conditional `PUT` creates the object (`200`/`201`/`204`) and streaming `GET` verification matches local `byteSize` and `SHA-256`.
- **`ALREADY_PRESENT`**: Returned when conditional `PUT` receives `412 Precondition Failed` and streaming `GET` verification confirms the remote object has identical `byteSize` and `SHA-256`.
- **`CanonicalArtifactIntegrityError`**: Raised when `412 Precondition Failed` is returned for a remote object whose bytes or SHA-256 differ from the local file, or when post-upload verification fails.

---

## 6. Presigned URL Redaction

`redact_capability_url` strips query strings (including `X-Amz-Signature`) and fragments from `putUrl` and `getUrl` before formatting any exception message or log entry.
