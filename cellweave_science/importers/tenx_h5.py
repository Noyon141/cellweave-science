"""10x Genomics HDF5 format importer (TENX_H5) conforming to Phase 3 System Design and Policy B.

Implements:
- detect(): Cardinality (PRIMARY x1) and HDF5 physical format sniffing.
- validate(): Comprehensive structural validation of /matrix group, datasets, CSC format,
              Gene Expression modality, raw counts, unique barcodes, and stable IDs.
- inspect(): Extraction of dimensions, modality, Ensembl scheme, and organism candidate.
- canonicalize(): Mandatory transpose from (features x barcodes) to (cells x genes),
                  slot sanitization, clean Policy B AnnData construction, and SHA-256 calculation.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
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
from cellweave_science.importers.tenx_shared import (
    GENE_EXPRESSION_MODALITY,
    TenxImporterConfig,
)
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

_REQUIRED_MATRIX_DATASETS = ("data", "indices", "indptr", "shape", "barcodes")
_REQUIRED_FEATURE_DATASETS = ("id", "name", "feature_type")


class TenxH5Importer:
    """Importer for 10x Genomics HDF5 (TENX_H5) filtered feature barcode matrices."""

    def __init__(self, config: TenxImporterConfig | None = None) -> None:
        if config is not None and not isinstance(config, TenxImporterConfig):
            raise TypeError("config must be an instance of TenxImporterConfig")
        self.config = config

    def detect(self, source_set: ImportSourceSet) -> bool:
        """Determine if this importer supports the given source file set.

        Requirements:
        1. source_format == SourceFormat.TENX_H5.
        2. Exactly 1 PRIMARY source file.
        3. Physical container signature is HDF5.
        """
        if source_set.source_format != SourceFormat.TENX_H5:
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
        """Validate structural integrity, 10x group hierarchy, modality, and raw counts."""
        cardinality_issues = validate_import_source_set_cardinality(source_set)
        if cardinality_issues:
            return cardinality_issues

        primary_file = source_set.get_primary_file()
        path = validate_trusted_local_path(primary_file.local_path)

        # 1. Verify physical container format
        try:
            inspection = inspect_physical_file(path)
            if inspection.detected_format != PhysicalContainerFormat.HDF5:
                return [
                    ScientificIssue(
                        code=ScientificIssueCode.FORMAT_MISMATCH,
                        severity=ScientificIssueSeverity.ERROR,
                        message=f"File {path.name} is not an HDF5 container",
                        userActionRequired=True,
                    )
                ]
        except Exception as exc:  # noqa: BLE001
            return [
                ScientificIssue(
                    code=ScientificIssueCode.MALFORMED_INPUT,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Failed to inspect file {path.name}: {exc}",
                    userActionRequired=True,
                )
            ]

        # 2. Inspect 10x HDF5 group and dataset structure safely
        try:
            with h5py.File(path, "r") as h5:
                # Top-level /matrix group check
                if "matrix" not in h5:
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.MISSING_COMPONENT,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"10x HDF5 file {path.name} missing required top-level group '/matrix'",
                            userActionRequired=True,
                        )
                    ]

                matrix_grp = h5["matrix"]
                if not isinstance(matrix_grp, h5py.Group):
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.MALFORMED_INPUT,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"'/matrix' in {path.name} must be an HDF5 Group",
                            userActionRequired=True,
                        )
                    ]

                # Required datasets inside /matrix
                for req_dset in _REQUIRED_MATRIX_DATASETS:
                    if req_dset not in matrix_grp:
                        return [
                            ScientificIssue(
                                code=ScientificIssueCode.MISSING_COMPONENT,
                                severity=ScientificIssueSeverity.ERROR,
                                message=f"10x HDF5 file {path.name} missing required dataset '/matrix/{req_dset}'",
                                userActionRequired=True,
                            )
                        ]

                # Features group
                if "features" not in matrix_grp or not isinstance(
                    matrix_grp["features"], h5py.Group
                ):
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.MISSING_COMPONENT,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"10x HDF5 file {path.name} missing required group '/matrix/features'",
                            userActionRequired=True,
                        )
                    ]

                feat_grp = matrix_grp["features"]
                for req_feat in _REQUIRED_FEATURE_DATASETS:
                    if req_feat not in feat_grp:
                        return [
                            ScientificIssue(
                                code=ScientificIssueCode.MISSING_COMPONENT,
                                severity=ScientificIssueSeverity.ERROR,
                                message=f"10x HDF5 file {path.name} missing required feature dataset '/matrix/features/{req_feat}'",
                                userActionRequired=True,
                            )
                        ]

                # Shape validation
                shape = matrix_grp["shape"][:]
                if len(shape) != 2:
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.DIMENSION_MISMATCH,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"Invalid matrix shape array length {len(shape)}, expected 2",
                            userActionRequired=True,
                        )
                    ]

                n_features, n_barcodes = int(shape[0]), int(shape[1])
                if n_features <= 0 or n_barcodes <= 0:
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.DIMENSION_MISMATCH,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"10x matrix has non-positive dimensions: ({n_features}, {n_barcodes})",
                            userActionRequired=True,
                        )
                    ]

                # Array length consistency
                indptr = matrix_grp["indptr"][:]
                indices = matrix_grp["indices"][:]
                data = matrix_grp["data"][:]

                if len(indptr) != n_barcodes + 1:
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.DIMENSION_MISMATCH,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"indptr length ({len(indptr)}) does not match barcode count + 1 ({n_barcodes + 1})",
                            userActionRequired=True,
                        )
                    ]

                if len(indices) != len(data):
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.DIMENSION_MISMATCH,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"indices length ({len(indices)}) does not match data length ({len(data)})",
                            userActionRequired=True,
                        )
                    ]

                # Sparse structure bounds check
                if len(data) > 0:
                    if indptr[0] != 0 or indptr[-1] != len(data) or np.any(np.diff(indptr) < 0):
                        return [
                            ScientificIssue(
                                code=ScientificIssueCode.MALFORMED_INPUT,
                                severity=ScientificIssueSeverity.ERROR,
                                message="Invalid sparse indptr monotonic structure in 10x H5",
                                userActionRequired=True,
                            )
                        ]
                    if np.min(indices) < 0 or np.max(indices) >= n_features:
                        return [
                            ScientificIssue(
                                code=ScientificIssueCode.MALFORMED_INPUT,
                                severity=ScientificIssueSeverity.ERROR,
                                message="Sparse indices exceed feature dimension range",
                                userActionRequired=True,
                            )
                        ]

                # Barcode validation
                raw_barcodes = matrix_grp["barcodes"][:]
                if len(raw_barcodes) != n_barcodes:
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.DIMENSION_MISMATCH,
                            severity=ScientificIssueSeverity.ERROR,
                            message=f"Barcode count ({len(raw_barcodes)}) does not match matrix columns ({n_barcodes})",
                            userActionRequired=True,
                        )
                    ]

                clean_barcodes = [
                    str(b.decode("utf-8") if isinstance(b, bytes) else b).strip()
                    for b in raw_barcodes
                ]
                try:
                    validate_identifiers(clean_barcodes, context="cell barcodes (obs_names)")
                except CountValidationError as exc:
                    return [
                        ScientificIssue(
                            code=exc.issue_code,
                            severity=ScientificIssueSeverity.ERROR,
                            message=str(exc),
                            userActionRequired=True,
                        )
                    ]

                # Features validation
                raw_feat_ids = feat_grp["id"][:]
                raw_feat_names = feat_grp["name"][:]
                raw_feat_types = feat_grp["feature_type"][:]

                if (
                    len(raw_feat_ids) != n_features
                    or len(raw_feat_names) != n_features
                    or len(raw_feat_types) != n_features
                ):
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.DIMENSION_MISMATCH,
                            severity=ScientificIssueSeverity.ERROR,
                            message="Feature metadata arrays do not match matrix feature count",
                            userActionRequired=True,
                        )
                    ]

                clean_feat_ids = [
                    str(f.decode("utf-8") if isinstance(f, bytes) else f).strip()
                    for f in raw_feat_ids
                ]
                clean_feat_types = [
                    str(t.decode("utf-8") if isinstance(t, bytes) else t).strip()
                    for t in raw_feat_types
                ]

                # Check for empty identifiers
                for idx, fid in enumerate(clean_feat_ids):
                    if not fid:
                        return [
                            ScientificIssue(
                                code=ScientificIssueCode.MALFORMED_INPUT,
                                severity=ScientificIssueSeverity.ERROR,
                                message=f"Empty feature identifier at row {idx}",
                                userActionRequired=True,
                            )
                        ]

                # Check duplicate stable IDs
                if len(set(clean_feat_ids)) != len(clean_feat_ids):
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS,
                            severity=ScientificIssueSeverity.ERROR,
                            message="Duplicate stable feature identifiers found in 10x feature table",
                            userActionRequired=True,
                        )
                    ]

                # Modality validation: pure Gene Expression only in V1
                unsupported_modalities = {
                    t for t in clean_feat_types if t != GENE_EXPRESSION_MODALITY
                }
                if unsupported_modalities:
                    has_gene = any(t == GENE_EXPRESSION_MODALITY for t in clean_feat_types)
                    if has_gene:
                        msg = (
                            f"Mixed modality detected in 10x H5: Gene Expression combined with "
                            f"{sorted(unsupported_modalities)}. CellWeave V1 supports pure Gene Expression only. "
                            "Silent feature subsetting is prohibited."
                        )
                    else:
                        msg = (
                            f"Unsupported modality in 10x H5: {sorted(unsupported_modalities)}. "
                            "CellWeave V1 supports Gene Expression only."
                        )
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.UNSUPPORTED_MODALITY,
                            severity=ScientificIssueSeverity.ERROR,
                            message=msg,
                            userActionRequired=True,
                        )
                    ]

                # Raw counts validation
                source_csc = sp.csc_matrix(
                    (data, indices, indptr),
                    shape=(n_features, n_barcodes),
                )
                count_class = classify_count_matrix(source_csc)
                if count_class == CountClassification.AMBIGUOUS:
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS,
                            severity=ScientificIssueSeverity.ERROR,
                            message="10x H5 count values are near-integral but ambiguous; raw counts cannot be verified",
                            userActionRequired=True,
                        )
                    ]
                if count_class == CountClassification.NOT_COUNTS:
                    return [
                        ScientificIssue(
                            code=ScientificIssueCode.NON_COUNT_INPUT,
                            severity=ScientificIssueSeverity.ERROR,
                            message="10x H5 expression data contains non-count, negative, NaN, or fractional values",
                            userActionRequired=True,
                        )
                    ]

        except Exception as exc:  # noqa: BLE001
            return [
                ScientificIssue(
                    code=ScientificIssueCode.MALFORMED_INPUT,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Failed to read 10x HDF5 structure from {path.name}: {exc}",
                    userActionRequired=True,
                )
            ]

        return []

    def inspect(self, source_set: ImportSourceSet) -> ImportInspectionResult:
        """Extract dataset dimensions, detect candidate organism/gene scheme, and check if mapping is required."""
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
                sourceFormat=SourceFormat.TENX_H5,
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

        with h5py.File(path, "r") as h5:
            matrix_grp = h5["matrix"]
            shape = matrix_grp["shape"][:]
            n_features, n_barcodes = int(shape[0]), int(shape[1])

            feat_ids_raw = matrix_grp["features"]["id"][:]
            clean_feat_ids = [
                str(f.decode("utf-8") if isinstance(f, bytes) else f).strip() for f in feat_ids_raw
            ]

        # Species conflict check
        inspection_issues: list[ScientificIssue] = []
        if check_ensembl_species_conflict(clean_feat_ids):
            inspection_issues.append(
                ScientificIssue(
                    code=ScientificIssueCode.UNRESOLVED_SPECIES,
                    severity=ScientificIssueSeverity.ERROR,
                    message="Conflicting Ensembl identifiers: 10x features contain both Human (ENSG) and Mouse (ENSMUSG) gene IDs",
                    userActionRequired=True,
                )
            )

        # Ensembl organism candidate resolution using Step D conservative policy
        detected_org: Organism | None = detect_ensembl_organism_candidates(clean_feat_ids)
        detected_scheme: GeneIdScheme | None = None

        if detected_org is not None:
            detected_scheme = GeneIdScheme.ENSEMBL
        else:
            # Check if feature IDs are unique symbols
            try:
                validate_identifiers(clean_feat_ids, context="feature IDs")
                detected_scheme = GeneIdScheme.SYMBOL
            except CountValidationError:
                detected_scheme = None

            if not inspection_issues:
                inspection_issues.append(
                    ScientificIssue(
                        code=ScientificIssueCode.UNRESOLVED_SPECIES,
                        severity=ScientificIssueSeverity.WARNING,
                        message="Organism could not be automatically determined from 10x features; user confirmation required",
                        userActionRequired=True,
                    )
                )

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
            sourceFormat=SourceFormat.TENX_H5,
            status=status,
            requiresMapping=requires_mapping,
            availableCellMetadataColumns=[],
            availableFeatureMetadataColumns=["gene_symbol"],
            issues=inspection_issues,
            cellCountEstimate=n_barcodes,
            geneCountEstimate=n_features,
            detectedOrganism=detected_org,
            detectedGeneIdScheme=detected_scheme,
            referenceBuildCandidate=None,
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
        """Convert source 10x H5 dataset to canonical AnnData conforming to Policy B (counts in X).

        Mandatory transpose: source (features x barcodes) -> canonical (cells x genes).
        """
        issues = self.validate(source_set)
        blocking_issues = [iss for iss in issues if iss.severity == ScientificIssueSeverity.ERROR]
        if blocking_issues:
            raise ImporterSecurityError(
                f"Cannot canonicalize invalid 10x H5 dataset: {blocking_issues[0].message}",
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

        # 1. Read HDF5 matrix data
        with h5py.File(source_path, "r") as h5:
            matrix_grp = h5["matrix"]
            shape = matrix_grp["shape"][:]
            n_features, n_barcodes = int(shape[0]), int(shape[1])

            data = matrix_grp["data"][:]
            indices = matrix_grp["indices"][:]
            indptr = matrix_grp["indptr"][:]

            barcodes_raw = matrix_grp["barcodes"][:]
            feat_ids_raw = matrix_grp["features"]["id"][:]
            feat_names_raw = matrix_grp["features"]["name"][:]

        clean_barcodes = [
            str(b.decode("utf-8") if isinstance(b, bytes) else b).strip() for b in barcodes_raw
        ]
        clean_feat_ids = [
            str(f.decode("utf-8") if isinstance(f, bytes) else f).strip() for f in feat_ids_raw
        ]
        clean_feat_names = [
            str(f.decode("utf-8") if isinstance(f, bytes) else f).strip() for f in feat_names_raw
        ]

        # Species conflict check against confirmed organism
        if check_ensembl_species_conflict(clean_feat_ids):
            raise ImporterSecurityError(
                "Cannot canonicalize dataset containing conflicting Human and Mouse Ensembl IDs",
                issue_code=ScientificIssueCode.UNRESOLVED_SPECIES,
            )
        if has_opposing_ensembl_species(clean_feat_ids, organism):
            raise ImporterSecurityError(
                f"Cannot canonicalize dataset: specified organism {organism.value} conflicts with opposing Ensembl IDs",
                issue_code=ScientificIssueCode.UNRESOLVED_SPECIES,
            )

        # 2. Build source CSC sparse matrix (features x barcodes)
        max_val = float(np.max(data)) if len(data) > 0 else 0.0
        val_dtype = np.int64 if max_val > float(np.iinfo(np.int32).max) else np.int32
        clean_data = data.astype(val_dtype)

        source_csc = sp.csc_matrix(
            (clean_data, indices, indptr),
            shape=(n_features, n_barcodes),
            dtype=val_dtype,
        )

        # 3. Mandatory Transpose: (features x barcodes) -> (cells x genes) = (barcodes x features)
        canonical_csr = ensure_csr_matrix(source_csc.T)

        # Assert orientation invariants
        assert (
            canonical_csr.shape == (n_barcodes, n_features)
        ), f"Transposed matrix shape {canonical_csr.shape} must equal (cells, genes) = ({n_barcodes}, {n_features})"

        # 4. Construct canonical var DataFrame
        canonical_var = pd.DataFrame(index=clean_feat_ids)
        canonical_var["source_feature_id"] = clean_feat_ids
        canonical_var["gene_symbol"] = clean_feat_names

        # 5. Construct canonical obs DataFrame
        clean_obs_names = validate_identifiers(clean_barcodes, context="barcodes (obs_names)")
        canonical_obs = pd.DataFrame(index=clean_obs_names)

        # 10x sources do not provide a cell-level sample metadata dimension.
        # Fail closed if mapping contains inapplicable sample mappings or cell metadata columns.
        if mapping is not None:
            if mapping.sampleColumn:
                raise ImporterSecurityError(
                    f"Cannot apply sampleColumn '{mapping.sampleColumn}' to 10x dataset: "
                    "10x H5 format does not provide cell metadata columns",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if mapping.sampleMappings:
                raise ImporterSecurityError(
                    "Cannot apply sampleMappings to 10x dataset: "
                    "10x H5 format does not provide a source sample dimension; barcodes are cell identities, not sample identities",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if mapping.conditionColumn:
                raise ImporterSecurityError(
                    f"Cannot apply conditionColumn '{mapping.conditionColumn}' to 10x dataset: "
                    "10x H5 format does not provide cell metadata columns",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if mapping.batchColumn:
                raise ImporterSecurityError(
                    f"Cannot apply batchColumn '{mapping.batchColumn}' to 10x dataset: "
                    "10x H5 format does not provide cell metadata columns",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if mapping.biologicalReplicateColumn:
                raise ImporterSecurityError(
                    f"Cannot apply biologicalReplicateColumn '{mapping.biologicalReplicateColumn}' to 10x dataset: "
                    "10x H5 format does not provide cell metadata columns",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )
            if mapping.technicalReplicateColumn:
                raise ImporterSecurityError(
                    f"Cannot apply technicalReplicateColumn '{mapping.technicalReplicateColumn}' to 10x dataset: "
                    "10x H5 format does not provide cell metadata columns",
                    issue_code=ScientificIssueCode.MISSING_REQUIRED_METADATA,
                )

        # 6. Build canonical AnnData via format-neutral Step D builder
        canonical_adata = build_canonical_anndata(
            x=canonical_csr,
            obs=canonical_obs,
            var=canonical_var,
            obs_names=clean_obs_names,
            var_names=clean_feat_ids,
            organism=organism,
            gene_id_scheme=gene_id_scheme,
            reference_build=reference_build,
        )

        # 7. Validate canonical invariants
        validate_canonical_anndata(
            canonical_adata,
            expected_organism=organism,
            expected_gene_id_scheme=gene_id_scheme,
        )

        # 8. Serialize canonical AnnData to disk
        os.makedirs(dest_path.parent, exist_ok=True)
        canonical_adata.write_h5ad(dest_path)

        # 9. Compute exact canonical SHA-256 of written artifact
        canonical_checksum = compute_file_sha256(dest_path)

        return CanonicalizationResult(
            sourceFormat=SourceFormat.TENX_H5,
            canonicalFormat="H5AD",
            canonicalChecksum=canonical_checksum,
            organism=organism,
            geneIdScheme=gene_id_scheme,
            cellCount=int(canonical_adata.n_obs),
            geneCount=int(canonical_adata.n_vars),
            warnings=[],
            referenceBuild=reference_build,
        )


__all__ = ["TenxH5Importer"]
