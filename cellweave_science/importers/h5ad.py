"""Concrete H5AD format importer conforming to Phase 3 System Design and Policy B.

Implements:
- detect(): Physical HDF5 container check & single PRIMARY file cardinality.
- validate(): Comprehensive structural integrity check (obs_names, dimensions, raw counts).
- inspect(): Extraction of dimensions, metadata columns, deterministic organism/gene scheme detection.
- canonicalize(): Building new canonical AnnData (.h5ad) conforming strictly to Policy B (counts in X).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellweave_science.contracts import (
    CanonicalizationResult,
    GeneIdScheme,
    ImportFileRole,
    ImportInspectionResult,
    InspectionStatus,
    MetadataMappingSubmission,
    Organism,
    ScientificIssue,
    ScientificIssueCode,
    ScientificIssueSeverity,
    SourceFormat,
)
from cellweave_science.importers.canonical import (
    build_canonical_anndata,
    validate_canonical_anndata,
)
from cellweave_science.importers.checksum import compute_file_sha256
from cellweave_science.importers.errors import (
    CountValidationError,
    ImporterSecurityError,
)
from cellweave_science.importers.security import (
    PhysicalContainerFormat,
    inspect_physical_file,
    validate_trusted_local_path,
)
from cellweave_science.importers.sources import ImportSourceSet
from cellweave_science.importers.validation import (
    CountClassification,
    check_ensembl_species_conflict,
    classify_count_matrix,
    detect_ensembl_organism_candidates,
    ensure_csr_matrix,
    has_opposing_ensembl_species,
    validate_identifiers,
    validate_import_source_set_cardinality,
)

# Recognized explicit organism uns keys (case-insensitive audit)
_EXPLICIT_ORGANISM_KEYS = ("organism", "species", "taxon_id")

# Exact normalized metadata values for organism identification (no fuzzy matching)
_ACCEPTED_HUMAN_METADATA_VALUES = frozenset(
    {
        "human",
        "homo sapiens",
        "homo_sapiens",
        "9606",
    }
)

_ACCEPTED_MOUSE_METADATA_VALUES = frozenset(
    {
        "mouse",
        "mus musculus",
        "mus_musculus",
        "10090",
    }
)

_UNSPECIFIED_METADATA_VALUES = frozenset(
    {
        "",
        "unknown",
        "unspecified",
        "na",
        "n/a",
        "none",
        "null",
    }
)

# Recognized explicit stable gene identifier column candidates in var
_CANDIDATE_STABLE_ID_COLUMNS = ("gene_ids", "gene_id", "ensembl_id", "ensembl")

# Recognized explicit descriptive gene symbol column candidates in var
_CANDIDATE_GENE_SYMBOL_COLUMNS = (
    "gene_symbol",
    "symbol",
    "gene_symbols",
    "gene_name",
    "feature_name",
)

# Recognized reference build uns keys
_EXPLICIT_BUILD_KEYS = ("reference_build", "genome", "reference_genome", "assembly")


def _safe_read_h5ad(path: Path) -> ad.AnnData:
    """Read an H5AD file safely, suppressing AnnData's UserWarnings regarding duplicate names."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ad.read_h5ad(path)


class H5ADImporter:
    """Importer for single-cell AnnData H5AD source datasets."""

    def detect(self, source_set: ImportSourceSet) -> bool:
        """Determine if this importer supports the given source file set.

        Requirements:
        1. source_format == SourceFormat.H5AD.
        2. Exactly 1 PRIMARY source file.
        3. Physical container signature is HDF5.
        """
        if source_set.source_format != SourceFormat.H5AD:
            return False

        if len(source_set.files) != 1 or source_set.files[0].role != ImportFileRole.PRIMARY:
            return False

        primary_file = source_set.files[0]
        try:
            inspection = inspect_physical_file(primary_file.local_path)
            return inspection.detected_format == PhysicalContainerFormat.HDF5
        except (ImporterSecurityError, OSError):
            return False

    def validate(self, source_set: ImportSourceSet) -> list[ScientificIssue]:
        """Validate structural integrity and format cardinality before canonicalization."""
        # 1. Structural role cardinality
        issues = validate_import_source_set_cardinality(source_set)
        if issues:
            return issues

        primary_file = source_set.get_primary_file()
        try:
            path = validate_trusted_local_path(primary_file.local_path)
        except ImporterSecurityError as exc:
            return [
                ScientificIssue(
                    code=exc.issue_code,
                    severity=ScientificIssueSeverity.ERROR,
                    message=str(exc),
                    userActionRequired=True,
                )
            ]

        # 2. Physical container verification
        try:
            inspection = inspect_physical_file(path)
            if inspection.detected_format != PhysicalContainerFormat.HDF5:
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.FORMAT_MISMATCH,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"File {path.name} is not an HDF5 container (detected: {inspection.detected_format.value})",
                        userActionRequired=True,
                    )
                ]
        except Exception as exc:  # noqa: BLE001
            return [
                ScientificIssue(
                    code=ScientificIssueCode.MALFORMED_INPUT,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Failed to physically inspect {path.name}: {exc}",
                    userActionRequired=True,
                )
            ]

        # 3. Safe AnnData opening and structural checks
        try:
            adata = _safe_read_h5ad(path)
        except Exception as exc:  # noqa: BLE001
            return [
                ScientificIssue(
                    code=ScientificIssueCode.MALFORMED_INPUT,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Corrupt or malformed H5AD structure in {path.name}: {exc}",
                    userActionRequired=True,
                )
            ]

        # 4. Dimensions check
        if adata.n_obs <= 0 or adata.n_vars <= 0:
            return [
                ScientificIssue(
                    code=ScientificIssueCode.DIMENSION_MISMATCH,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"H5AD has empty dimensions: shape=({adata.n_obs}, {adata.n_vars})",
                    userActionRequired=True,
                )
            ]

        # 5. Cell identifiers uniqueness (obs_names)
        try:
            validate_identifiers(adata.obs_names.tolist(), context="cell IDs (obs_names)")
        except CountValidationError as exc:
            return [
                ScientificIssue(
                    code=exc.issue_code,
                    severity=ScientificIssueSeverity.ERROR,
                    message=str(exc),
                    userActionRequired=True,
                )
            ]

        # 6. Raw count availability check
        try:
            self._resolve_raw_count_matrix(adata)
        except CountValidationError as exc:
            return [
                ScientificIssue(
                    code=exc.issue_code,
                    severity=ScientificIssueSeverity.ERROR,
                    message=str(exc),
                    userActionRequired=True,
                )
            ]

        # 7. Feature identity check
        feature_resolution = self._resolve_feature_identity(adata)
        if feature_resolution["issue"] is not None:
            return [feature_resolution["issue"]]

        return []

    def inspect(self, source_set: ImportSourceSet) -> ImportInspectionResult:
        """Extract dataset dimensions, detect candidate organism/gene schemes, and check if mapping is required."""
        if not source_set.dataset_version_id or not source_set.dataset_version_id.strip():
            raise ImporterSecurityError(
                "Cannot inspect dataset without required dataset_version_id on source_set",
                issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
            )
        version_id = source_set.dataset_version_id.strip()

        issues = self.validate(source_set)
        blocking_issues = [iss for iss in issues if iss.severity == ScientificIssueSeverity.ERROR]

        if blocking_issues:
            return ImportInspectionResult(
                datasetVersionId=version_id,
                sourceFormat=SourceFormat.H5AD,
                status=InspectionStatus.INVALID,
                requiresMapping=False,
                availableCellMetadataColumns=[],
                availableFeatureMetadataColumns=[],
                issues=issues,
                cellCountEstimate=None,
                geneCountEstimate=None,
                detectedOrganism=None,
                detectedGeneIdScheme=None,
                referenceBuildCandidate=None,
            )

        primary_file = source_set.get_primary_file()
        path = validate_trusted_local_path(primary_file.local_path)
        adata = _safe_read_h5ad(path)

        cell_count = int(adata.n_obs)
        gene_count = int(adata.n_vars)
        available_cell_cols = [str(c) for c in adata.obs.columns]
        available_feat_cols = [str(c) for c in adata.var.columns]

        # Feature scheme detection
        feature_res = self._resolve_feature_identity(adata)
        detected_scheme: GeneIdScheme | None = feature_res["scheme"]

        # Organism detection
        organism_res = self._resolve_organism(adata, feature_res["canonical_var_names"])
        detected_org: Organism | None = organism_res["organism"]
        organism_issue: ScientificIssue | None = organism_res["issue"]

        inspection_issues: list[ScientificIssue] = []
        if organism_issue is not None:
            inspection_issues.append(organism_issue)

        # Reference build detection
        ref_build = self._resolve_reference_build(adata)

        # Determine inspection status
        if any(iss.severity == ScientificIssueSeverity.ERROR for iss in inspection_issues):
            status = InspectionStatus.INVALID
            requires_mapping = False
        elif (
            detected_org is None
            or detected_scheme is None
            or any(iss.userActionRequired for iss in inspection_issues)
        ):
            status = InspectionStatus.NEEDS_MAPPING
            requires_mapping = True
        else:
            status = InspectionStatus.VALID
            requires_mapping = False

        return ImportInspectionResult(
            datasetVersionId=version_id,
            sourceFormat=SourceFormat.H5AD,
            status=status,
            requiresMapping=requires_mapping,
            availableCellMetadataColumns=available_cell_cols,
            availableFeatureMetadataColumns=available_feat_cols,
            issues=inspection_issues,
            cellCountEstimate=cell_count,
            geneCountEstimate=gene_count,
            detectedOrganism=detected_org,
            detectedGeneIdScheme=detected_scheme,
            referenceBuildCandidate=ref_build,
        )

    def canonicalize(
        self,
        source_set: ImportSourceSet,
        output_path: str,
        organism: Organism,
        gene_id_scheme: GeneIdScheme,
        reference_build: str | None = None,
        mapping: MetadataMappingSubmission | None = None,
    ) -> CanonicalizationResult:
        """Convert source H5AD dataset to canonical AnnData (.h5ad) conforming to Policy B (counts in X)."""
        issues = self.validate(source_set)
        blocking_issues = [iss for iss in issues if iss.severity == ScientificIssueSeverity.ERROR]
        if blocking_issues:
            raise ImporterSecurityError(
                f"Cannot canonicalize invalid H5AD dataset: {blocking_issues[0].message}",
                issue_code=blocking_issues[0].code,
            )

        primary_file = source_set.get_primary_file()
        source_path = validate_trusted_local_path(primary_file.local_path)
        dest_path = Path(output_path).resolve()

        if source_path == dest_path:
            raise ImporterSecurityError(
                "Source file immutability violation: output_path cannot match source_path",
                issue_code=ScientificIssueCode.CANONICALIZATION_FAILED,
            )

        # Load source AnnData
        adata = _safe_read_h5ad(source_path)

        # 1. Resolve raw count matrix (X vs counts layer vs raw.X)
        count_matrix = self._resolve_raw_count_matrix(adata)

        # 2. Resolve feature identity & var table
        feature_res = self._resolve_feature_identity(adata)
        canonical_var_names = feature_res["canonical_var_names"]
        gene_symbols = feature_res["gene_symbols"]

        # Verify confirmed organism does not conflict with Ensembl ID evidence
        if check_ensembl_species_conflict(canonical_var_names):
            raise ImporterSecurityError(
                "Cannot canonicalize dataset containing conflicting Human and Mouse Ensembl IDs",
                issue_code=ScientificIssueCode.UNRESOLVED_SPECIES,
            )
        if has_opposing_ensembl_species(canonical_var_names, organism):
            raise ImporterSecurityError(
                f"Cannot canonicalize dataset: specified organism {organism.value} conflicts with opposing Ensembl IDs",
                issue_code=ScientificIssueCode.UNRESOLVED_SPECIES,
            )

        # Build clean canonical var DataFrame
        canonical_var = pd.DataFrame(index=canonical_var_names)
        canonical_var["source_feature_id"] = adata.var_names.tolist()
        if gene_symbols is not None:
            canonical_var["gene_symbol"] = gene_symbols

        # 3. Resolve cell metadata & obs table
        clean_obs_names = validate_identifiers(
            adata.obs_names.tolist(), context="cell IDs (obs_names)"
        )
        canonical_obs = pd.DataFrame(index=clean_obs_names)

        # Apply mapped columns if mapping submission provided
        if mapping is not None:
            self._apply_metadata_mapping(canonical_obs, adata.obs, mapping)

        # 4. Construct canonical AnnData via format-neutral Step D builder
        canonical_adata = build_canonical_anndata(
            x=count_matrix,
            obs=canonical_obs,
            var=canonical_var,
            obs_names=clean_obs_names,
            var_names=canonical_var_names,
            organism=organism,
            gene_id_scheme=gene_id_scheme,
            reference_build=reference_build,
        )

        # 5. Validate canonical invariants (Policy B, CSR sparse, slots hygiene)
        validate_canonical_anndata(
            canonical_adata,
            expected_organism=organism,
            expected_gene_id_scheme=gene_id_scheme,
        )

        # 6. Serialize to output destination
        os.makedirs(dest_path.parent, exist_ok=True)
        canonical_adata.write_h5ad(dest_path)

        # 7. Compute exact canonical SHA-256 of the written artifact
        canonical_checksum = compute_file_sha256(dest_path)

        return CanonicalizationResult(
            sourceFormat=SourceFormat.H5AD,
            canonicalFormat="H5AD",
            canonicalChecksum=canonical_checksum,
            organism=organism,
            geneIdScheme=gene_id_scheme,
            cellCount=int(canonical_adata.n_obs),
            geneCount=int(canonical_adata.n_vars),
            warnings=[],
            referenceBuild=reference_build,
            localArtifactPath=str(dest_path),
        )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _resolve_raw_count_matrix(self, adata: ad.AnnData) -> sp.csr_matrix:
        """Resolve authoritative raw unnormalized count matrix from candidate sources.

        Precedence & Conflict Resolution:
        1. Evaluate adata.X:
           - If VALID_COUNTS: candidate_x = X
           - If AMBIGUOUS: candidate_x = AMBIGUOUS
           - If NOT_COUNTS: candidate_x = None
        2. Evaluate explicit counts layer:
           - If adata.layers contains 'counts':
             - Must have shape (n_obs, n_vars)
             - Classify: VALID_COUNTS, AMBIGUOUS, NOT_COUNTS
        3. Evaluate adata.raw:
           - If adata.raw exists, verify obs_names align and evaluate var alignment.
           - Classify raw.X.
        4. Cross-candidate consistency check:
           - If multiple valid count sources exist (e.g. X and counts layer, or X and raw.X),
             they must agree materially. If they conflict, raise AMBIGUOUS_COUNT_SEMANTICS.
        5. If only transformed data exists and zero valid raw counts can be found,
           raise NON_COUNT_INPUT.
        """
        candidate_x: sp.csr_matrix | None = None
        candidate_counts_layer: sp.csr_matrix | None = None
        candidate_raw: sp.csr_matrix | None = None

        x_classification = (
            classify_count_matrix(adata.X)
            if adata.X is not None
            else CountClassification.NOT_COUNTS
        )

        if x_classification == CountClassification.VALID_COUNTS:
            candidate_x = ensure_csr_matrix(adata.X)
        elif x_classification == CountClassification.AMBIGUOUS:
            raise CountValidationError(
                "adata.X expression values are near-integral but ambiguous; raw counts cannot be verified",
                issue_code=ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS,
            )

        # Inspect explicit 'counts' layer
        if adata.layers is not None and "counts" in adata.layers:
            layer_mat = adata.layers["counts"]
            if layer_mat.shape != (adata.n_obs, adata.n_vars):
                raise CountValidationError(
                    f"counts layer shape {layer_mat.shape} does not match AnnData shape ({adata.n_obs}, {adata.n_vars})",
                    issue_code=ScientificIssueCode.DIMENSION_MISMATCH,
                )
            layer_class = classify_count_matrix(layer_mat)
            if layer_class == CountClassification.VALID_COUNTS:
                candidate_counts_layer = ensure_csr_matrix(layer_mat)
            elif layer_class == CountClassification.AMBIGUOUS:
                raise CountValidationError(
                    "layers['counts'] values are near-integral but ambiguous",
                    issue_code=ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS,
                )
            elif layer_class == CountClassification.NOT_COUNTS and candidate_x is None:
                raise CountValidationError(
                    "layers['counts'] does not contain valid raw counts",
                    issue_code=ScientificIssueCode.NON_COUNT_INPUT,
                )

        # Inspect adata.raw
        if adata.raw is not None:
            candidate_raw = self._evaluate_adata_raw(adata)

        # Determine authoritative count source & conflict resolution
        valid_candidates: list[tuple[str, sp.csr_matrix]] = []
        if candidate_x is not None:
            valid_candidates.append(("X", candidate_x))
        if candidate_counts_layer is not None:
            valid_candidates.append(("layers['counts']", candidate_counts_layer))
        if candidate_raw is not None:
            valid_candidates.append(("raw.X", candidate_raw))

        if not valid_candidates:
            raise CountValidationError(
                "No valid raw unnormalized counts found in adata.X, layers['counts'], or adata.raw",
                issue_code=ScientificIssueCode.NON_COUNT_INPUT,
            )

        # If exactly 1 valid candidate exists, return it
        if len(valid_candidates) == 1:
            return valid_candidates[0][1]

        # Multiple candidates exist: verify they agree materially
        base_name, base_matrix = valid_candidates[0]
        for other_name, other_matrix in valid_candidates[1:]:
            if not self._matrices_are_equal(base_matrix, other_matrix):
                raise CountValidationError(
                    f"Conflicting raw-count candidates: {base_name} and {other_name} both contain valid counts but differ",
                    issue_code=ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS,
                )

        # If X is normalized and counts layer is valid, return counts layer
        if candidate_counts_layer is not None and candidate_x is None:
            return candidate_counts_layer

        return base_matrix

    def _evaluate_adata_raw(self, adata: ad.AnnData) -> sp.csr_matrix | None:
        """Evaluate adata.raw feature alignment and count validity without positional guessing."""
        raw = adata.raw
        if raw is None or raw.X is None:
            return None

        # 1. obs_names must match current obs_names exactly
        if list(raw.obs_names) != list(adata.obs_names):
            return None

        # 2. Check feature identity alignment
        raw_var_names = list(raw.var_names)
        cur_var_names = list(adata.var_names)

        raw_mat = raw.X
        raw_classification = classify_count_matrix(raw_mat)
        if raw_classification == CountClassification.AMBIGUOUS:
            raise CountValidationError(
                "adata.raw.X expression values are near-integral but ambiguous; raw counts cannot be verified",
                issue_code=ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS,
            )
        if raw_classification != CountClassification.VALID_COUNTS:
            return None

        raw_csr = ensure_csr_matrix(raw_mat)

        if raw_var_names == cur_var_names:
            # Exact identity and order match
            return raw_csr

        # Check if raw_var_names is a superset of current var_names with stable unique IDs
        cur_var_set = set(cur_var_names)
        if len(cur_var_set) != len(cur_var_names):
            return None

        raw_var_set = set(raw_var_names)
        if cur_var_set.issubset(raw_var_set):
            # Deterministic subset extraction by exact feature identifier
            raw_idx_map = {name: idx for idx, name in enumerate(raw_var_names)}
            ordered_indices = [raw_idx_map[name] for name in cur_var_names]
            subset_csr = raw_csr[:, ordered_indices]
            return ensure_csr_matrix(subset_csr)

        # raw is missing features or ambiguous
        return None

    def _matrices_are_equal(self, a: sp.csr_matrix, b: sp.csr_matrix) -> bool:
        """Sparse-aware matrix equality check without densification."""
        if a.shape != b.shape:
            return False
        # Calculate sparse difference
        diff = a - b
        if diff.nnz == 0:
            return True
        return bool(np.all(diff.data == 0))

    def _resolve_feature_identity(self, adata: ad.AnnData) -> dict[str, Any]:
        """Determine stable canonical var_names, gene symbols, and scheme."""
        source_var_names = [str(x).strip() for x in adata.var_names]

        # 1. Check if source var_names are already consistent Ensembl IDs
        ens_candidate = detect_ensembl_organism_candidates(source_var_names)
        if ens_candidate is not None:
            # var_names are stable Ensembl IDs!
            # Check for descriptive symbol column
            gene_symbols: list[str] | None = None
            for col in _CANDIDATE_GENE_SYMBOL_COLUMNS:
                if col in adata.var.columns:
                    gene_symbols = [str(s).strip() for s in adata.var[col]]
                    break

            return {
                "canonical_var_names": source_var_names,
                "gene_symbols": gene_symbols,
                "scheme": GeneIdScheme.ENSEMBL,
                "issue": None,
            }

        # 2. Check if an explicit stable ID column exists in var
        for col in _CANDIDATE_STABLE_ID_COLUMNS:
            if col in adata.var.columns:
                candidate_ids = [str(x).strip() for x in adata.var[col]]
                try:
                    clean_ids = validate_identifiers(candidate_ids, context=f"var['{col}']")
                except CountValidationError:
                    continue

                if detect_ensembl_organism_candidates(clean_ids) is not None:
                    # Found stable Ensembl ID column!
                    # Source var_names become descriptive gene_symbols
                    return {
                        "canonical_var_names": clean_ids,
                        "gene_symbols": source_var_names,
                        "scheme": GeneIdScheme.ENSEMBL,
                        "issue": None,
                    }

        # 3. Check if var_names are unique and usable under SYMBOL scheme
        try:
            clean_symbols = validate_identifiers(
                source_var_names, context="gene symbols (var_names)"
            )
            return {
                "canonical_var_names": clean_symbols,
                "gene_symbols": clean_symbols,
                "scheme": GeneIdScheme.SYMBOL,
                "issue": None,
            }
        except CountValidationError:
            # Duplicate gene symbols with no alternative stable ID column -> INVALID
            return {
                "canonical_var_names": [],
                "gene_symbols": None,
                "scheme": None,
                "issue": ScientificIssue(
                    code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                    severity=ScientificIssueSeverity.ERROR,
                    message="var_names contain duplicate gene symbols and no stable gene ID column is available",
                    userActionRequired=True,
                ),
            }

    def _resolve_organism(
        self, adata: ad.AnnData, canonical_var_names: list[str]
    ) -> dict[str, Any]:
        """Determine organism from explicit metadata and Ensembl gene identifiers using Step D policy."""
        # 1. First check if gene identifiers themselves contain a conflicting mixture of species
        if check_ensembl_species_conflict(canonical_var_names):
            return {
                "organism": None,
                "issue": ScientificIssue(
                    code=ScientificIssueCode.UNRESOLVED_SPECIES,
                    severity=ScientificIssueSeverity.ERROR,
                    message="Conflicting Ensembl identifiers: dataset contains both Human (ENSG) and Mouse (ENSMUSG) gene IDs",
                    userActionRequired=True,
                ),
            }

        # 2. Check explicit uns metadata against narrow accepted values (no substring matching)
        explicit_org: Organism | None = None
        has_unsupported_org = False
        unsupported_org_val = ""

        if adata.uns is not None:
            for k in _EXPLICIT_ORGANISM_KEYS:
                if k in adata.uns:
                    raw_val = str(adata.uns[k]).strip()
                    norm_val = raw_val.lower()
                    if norm_val in _ACCEPTED_HUMAN_METADATA_VALUES:
                        explicit_org = Organism.HUMAN
                        break
                    if norm_val in _ACCEPTED_MOUSE_METADATA_VALUES:
                        explicit_org = Organism.MOUSE
                        break
                    if norm_val in _UNSPECIFIED_METADATA_VALUES:
                        continue
                    has_unsupported_org = True
                    unsupported_org_val = raw_val
                    break

        if has_unsupported_org:
            return {
                "organism": None,
                "issue": ScientificIssue(
                    code=ScientificIssueCode.UNSUPPORTED_SPECIES,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Explicit source metadata specifies unsupported organism: {unsupported_org_val}",
                    userActionRequired=True,
                ),
            }

        # 3. Check for conflict between explicit metadata and opposing Ensembl IDs
        if explicit_org is not None and has_opposing_ensembl_species(
            canonical_var_names, explicit_org
        ):
            return {
                "organism": None,
                "issue": ScientificIssue(
                    code=ScientificIssueCode.UNRESOLVED_SPECIES,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Organism metadata ({explicit_org.value}) conflicts with opposing Ensembl gene identifiers",
                    userActionRequired=True,
                ),
            }

        # 4. Step D conservative Ensembl candidate detection (requires 100% consistent IDs)
        id_candidate = detect_ensembl_organism_candidates(canonical_var_names)

        # 5. Resolve detected organism
        detected_org = explicit_org or id_candidate

        if detected_org is None:
            return {
                "organism": None,
                "issue": ScientificIssue(
                    code=ScientificIssueCode.UNRESOLVED_SPECIES,
                    severity=ScientificIssueSeverity.WARNING,
                    message="Organism could not be automatically determined; user confirmation required",
                    userActionRequired=True,
                ),
            }

        return {"organism": detected_org, "issue": None}

    def _resolve_reference_build(self, adata: ad.AnnData) -> str | None:
        """Extract explicit reference build string from uns without guessing."""
        if adata.uns is not None:
            for k in _EXPLICIT_BUILD_KEYS:
                if k in adata.uns:
                    val = str(adata.uns[k]).strip()
                    if val:
                        return val
        return None

    def _apply_metadata_mapping(
        self,
        canonical_obs: pd.DataFrame,
        source_obs: pd.DataFrame,
        mapping: MetadataMappingSubmission,
    ) -> None:
        """Map approved cell metadata columns (sampleKey, condition, batch, replicates)."""
        # 1. Sample key mapping
        if mapping.sampleColumn:
            if mapping.sampleColumn not in source_obs.columns:
                raise ImporterSecurityError(
                    f"Requested sampleColumn '{mapping.sampleColumn}' not found in source cell metadata",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            source_vals = source_obs[mapping.sampleColumn].astype(str).tolist()
            if not mapping.sampleMappings:
                raise ImporterSecurityError(
                    f"sampleColumn '{mapping.sampleColumn}' provided without approved sampleMappings; "
                    "source values cannot pass through automatically as sample keys",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            map_dict = {m.sourceValue: m.sampleKey for m in mapping.sampleMappings}
            mapped_keys: list[str] = []
            for v in source_vals:
                target_key = map_dict.get(v)
                if not target_key or not target_key.strip():
                    raise ImporterSecurityError(
                        f"Incomplete sample mapping: source value '{v}' in column '{mapping.sampleColumn}' "
                        "has no approved SampleMappingEntry",
                        issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                    )
                mapped_keys.append(target_key.strip())
            canonical_obs["sample_key"] = mapped_keys
        elif mapping.sampleMappings:
            raise ImporterSecurityError(
                "sampleMappings provided without a corresponding sampleColumn",
                issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
            )

        # 2. Biological condition
        if mapping.conditionColumn:
            if mapping.conditionColumn not in source_obs.columns:
                raise ImporterSecurityError(
                    f"Requested conditionColumn '{mapping.conditionColumn}' not found in source cell metadata",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            canonical_obs["condition"] = source_obs[mapping.conditionColumn].astype(str).tolist()

        # 3. Batch
        if mapping.batchColumn:
            if mapping.batchColumn not in source_obs.columns:
                raise ImporterSecurityError(
                    f"Requested batchColumn '{mapping.batchColumn}' not found in source cell metadata",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            canonical_obs["batch"] = source_obs[mapping.batchColumn].astype(str).tolist()

        # 4. Biological replicate
        if mapping.biologicalReplicateColumn:
            if mapping.biologicalReplicateColumn not in source_obs.columns:
                raise ImporterSecurityError(
                    f"Requested biologicalReplicateColumn '{mapping.biologicalReplicateColumn}' not found in source cell metadata",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            canonical_obs["biological_replicate"] = (
                source_obs[mapping.biologicalReplicateColumn].astype(str).tolist()
            )

        # 5. Technical replicate
        if mapping.technicalReplicateColumn:
            if mapping.technicalReplicateColumn not in source_obs.columns:
                raise ImporterSecurityError(
                    f"Requested technicalReplicateColumn '{mapping.technicalReplicateColumn}' not found in source cell metadata",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            canonical_obs["technical_replicate"] = (
                source_obs[mapping.technicalReplicateColumn].astype(str).tolist()
            )


__all__ = ["H5ADImporter"]
