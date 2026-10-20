"""10x Genomics Matrix Market format importer (TENX_MTX) conforming to Phase 3 System Design and Policy B.

Implements:
- detect(): Cardinality (MATRIX x1, FEATURES x1, BARCODES x1) and Matrix Market header sniffing.
- validate(): Streaming validation of Matrix Market format, modern/legacy features,
              barcodes, Gene Expression modality, dimensions, and raw counts.
- inspect(): Extraction of dimensions, Ensembl scheme, and organism candidate.
- canonicalize(): Mandatory transpose from (features x barcodes) to (cells x genes),
                  slot sanitization, clean Policy B AnnData construction, and SHA-256 calculation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

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
    CompressionFormat,
    PhysicalContainerFormat,
    inspect_physical_file,
    open_bounded_gzip_stream,
    validate_trusted_local_path,
)
from cellweave_science.importers.sources import ImportSourceSet
from cellweave_science.importers.tenx_shared import (
    TenxImporterConfig,
    parse_10x_barcodes,
    parse_10x_features_tsv,
    parse_10x_matrix_market,
    validate_10x_features,
)
from cellweave_science.importers.validation import (
    check_ensembl_species_conflict,
    detect_ensembl_organism_candidates,
    ensure_csr_matrix,
    has_opposing_ensembl_species,
    validate_identifiers,
    validate_import_source_set_cardinality,
)


class TenxMtxImporter:
    """Importer for 10x Genomics Matrix Market (TENX_MTX) triplets."""

    def __init__(self, config: TenxImporterConfig | None = None) -> None:
        self.config = config or TenxImporterConfig()

    def detect(self, source_set: ImportSourceSet) -> bool:
        """Determine if this importer supports the given source file set.

        Requirements:
        1. source_format == SourceFormat.TENX_MTX.
        2. Exactly 3 files: 1 MATRIX, 1 FEATURES, 1 BARCODES.
        3. MATRIX file starts with %%MatrixMarket header (plain or gzip).
        """
        if source_set.source_format != SourceFormat.TENX_MTX:
            return False

        if len(source_set.files) != 3:
            return False

        try:
            matrix_file = source_set.get_single_file_by_role(ImportFileRole.MATRIX)
            _ = source_set.get_single_file_by_role(ImportFileRole.FEATURES)
            _ = source_set.get_single_file_by_role(ImportFileRole.BARCODES)
        except ValueError:
            return False

        try:
            inspection = inspect_physical_file(matrix_file.local_path)
            if inspection.detected_format == PhysicalContainerFormat.MATRIX_MARKET:
                return True
            if inspection.compression == CompressionFormat.GZIP:
                # Sniff first decompressed chunk
                for chunk in open_bounded_gzip_stream(
                    matrix_file.local_path, limits=self.config.decompression_limits
                ):
                    return chunk.startswith(b"%%MatrixMarket")
            return False
        except (ImporterSecurityError, OSError):
            return False

    def validate(self, source_set: ImportSourceSet) -> list[ScientificIssue]:
        """Validate structural integrity, file triplet, modality, coordinates, and raw counts."""
        cardinality_issues = validate_import_source_set_cardinality(source_set)
        if cardinality_issues:
            return cardinality_issues

        try:
            matrix_file = source_set.get_single_file_by_role(ImportFileRole.MATRIX)
            features_file = source_set.get_single_file_by_role(ImportFileRole.FEATURES)
            barcodes_file = source_set.get_single_file_by_role(ImportFileRole.BARCODES)
        except ValueError as exc:
            return [
                ScientificIssue(
                    code=ScientificIssueCode.INVALID_IMPORT_CARDINALITY,
                    severity=ScientificIssueSeverity.ERROR,
                    message=str(exc),
                    userActionRequired=True,
                )
            ]

        # 1. Parse and validate barcodes
        try:
            barcodes = parse_10x_barcodes(
                barcodes_file.local_path, limits=self.config.decompression_limits
            )
        except CountValidationError as exc:
            return [
                ScientificIssue(
                    code=exc.issue_code,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Barcodes validation failed in {barcodes_file.local_path.name}: {exc}",
                    userActionRequired=True,
                )
            ]
        except Exception as exc:  # noqa: BLE001
            return [
                ScientificIssue(
                    code=ScientificIssueCode.MALFORMED_INPUT,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Failed to read barcodes file {barcodes_file.local_path.name}: {exc}",
                    userActionRequired=True,
                )
            ]

        # 2. Parse and validate features (modern or legacy)
        try:
            features = parse_10x_features_tsv(
                features_file.local_path, limits=self.config.decompression_limits
            )
        except CountValidationError as exc:
            return [
                ScientificIssue(
                    code=exc.issue_code,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Features validation failed in {features_file.local_path.name}: {exc}",
                    userActionRequired=True,
                )
            ]
        except Exception as exc:  # noqa: BLE001
            return [
                ScientificIssue(
                    code=ScientificIssueCode.MALFORMED_INPUT,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Failed to read features file {features_file.local_path.name}: {exc}",
                    userActionRequired=True,
                )
            ]

        feature_issues = validate_10x_features(features)
        if feature_issues:
            return feature_issues

        # 3. Parse and validate Matrix Market matrix
        try:
            n_rows, n_cols, _n_entries, _coo = parse_10x_matrix_market(
                matrix_file.local_path, limits=self.config.decompression_limits
            )
        except CountValidationError as exc:
            return [
                ScientificIssue(
                    code=exc.issue_code,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Matrix Market validation failed in {matrix_file.local_path.name}: {exc}",
                    userActionRequired=True,
                )
            ]
        except Exception as exc:  # noqa: BLE001
            return [
                ScientificIssue(
                    code=ScientificIssueCode.MALFORMED_INPUT,
                    severity=ScientificIssueSeverity.ERROR,
                    message=f"Failed to read Matrix Market file {matrix_file.local_path.name}: {exc}",
                    userActionRequired=True,
                )
            ]

        # 4. Mandatory dimension alignment assertion
        if n_rows != len(features):
            return [
                ScientificIssue(
                    code=ScientificIssueCode.DIMENSION_MISMATCH,
                    severity=ScientificIssueSeverity.ERROR,
                    message=(
                        f"Matrix Market row count ({n_rows}) does not match "
                        f"parsed features count ({len(features)}) in {features_file.local_path.name}"
                    ),
                    userActionRequired=True,
                )
            ]

        if n_cols != len(barcodes):
            return [
                ScientificIssue(
                    code=ScientificIssueCode.DIMENSION_MISMATCH,
                    severity=ScientificIssueSeverity.ERROR,
                    message=(
                        f"Matrix Market column count ({n_cols}) does not match "
                        f"parsed barcodes count ({len(barcodes)}) in {barcodes_file.local_path.name}"
                    ),
                    userActionRequired=True,
                )
            ]

        return []

    def inspect(self, source_set: ImportSourceSet) -> ImportInspectionResult:
        """Extract dataset dimensions, detect candidate organism/gene scheme, and check if mapping is required."""
        first_file = source_set.files[0] if source_set.files else None
        version_id = source_set.dataset_version_id or (
            first_file.dataset_file_id if first_file is not None else ""
        )

        issues = self.validate(source_set)
        blocking_issues = [iss for iss in issues if iss.severity == ScientificIssueSeverity.ERROR]

        if blocking_issues:
            return ImportInspectionResult(
                datasetVersionId=version_id,
                sourceFormat=SourceFormat.TENX_MTX,
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

        features_file = source_set.get_single_file_by_role(ImportFileRole.FEATURES)
        barcodes_file = source_set.get_single_file_by_role(ImportFileRole.BARCODES)

        features = parse_10x_features_tsv(
            features_file.local_path, limits=self.config.decompression_limits
        )
        barcodes = parse_10x_barcodes(
            barcodes_file.local_path, limits=self.config.decompression_limits
        )

        clean_feat_ids = [f.feature_id for f in features]
        n_features = len(features)
        n_barcodes = len(barcodes)

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

        # Ensembl candidate resolution using Step D conservative policy
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
            sourceFormat=SourceFormat.TENX_MTX,
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
        """Convert source 10x Matrix Market triplet to canonical AnnData conforming to Policy B (counts in X).

        Mandatory transpose: source (features x barcodes) -> canonical (cells x genes).
        """
        issues = self.validate(source_set)
        blocking_issues = [iss for iss in issues if iss.severity == ScientificIssueSeverity.ERROR]
        if blocking_issues:
            raise ImporterSecurityError(
                f"Cannot canonicalize invalid 10x MTX dataset: {blocking_issues[0].message}",
                issue_code=blocking_issues[0].code,
            )

        dest_path = Path(output_path).resolve()
        for f in source_set.files:
            if validate_trusted_local_path(f.local_path).resolve() == dest_path:
                raise ImporterSecurityError(
                    "Source file immutability violation: output_path cannot match any source file path",
                    issue_code=ScientificIssueCode.CANONICALIZATION_FAILED,
                )

        matrix_file = source_set.get_single_file_by_role(ImportFileRole.MATRIX)
        features_file = source_set.get_single_file_by_role(ImportFileRole.FEATURES)
        barcodes_file = source_set.get_single_file_by_role(ImportFileRole.BARCODES)

        # 1. Parse triplet components
        barcodes = parse_10x_barcodes(
            barcodes_file.local_path, limits=self.config.decompression_limits
        )
        features = parse_10x_features_tsv(
            features_file.local_path, limits=self.config.decompression_limits
        )
        _n_rows, _n_cols, _n_entries, coo_matrix = parse_10x_matrix_market(
            matrix_file.local_path, limits=self.config.decompression_limits
        )

        clean_feat_ids = [f.feature_id for f in features]
        clean_feat_names = [f.gene_symbol for f in features]

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

        n_features = len(features)
        n_barcodes = len(barcodes)

        # 2. Mandatory Transpose: (features x barcodes) -> (cells x genes) = (barcodes x features)
        canonical_csr = ensure_csr_matrix(coo_matrix.T.tocsr())

        # Assert orientation invariants
        assert (
            canonical_csr.shape == (n_barcodes, n_features)
        ), f"Transposed matrix shape {canonical_csr.shape} must equal (cells, genes) = ({n_barcodes}, {n_features})"

        # 3. Construct canonical var DataFrame
        canonical_var = pd.DataFrame(index=clean_feat_ids)
        canonical_var["source_feature_id"] = clean_feat_ids
        canonical_var["gene_symbol"] = clean_feat_names

        # 4. Construct canonical obs DataFrame
        clean_obs_names = validate_identifiers(barcodes, context="barcodes (obs_names)")
        canonical_obs = pd.DataFrame(index=clean_obs_names)

        # Populate sample_key: default SMP_001 if unmapped
        if (
            mapping is not None
            and mapping.sampleColumn
            and mapping.sampleColumn in canonical_obs.columns
        ):
            canonical_obs["sample_key"] = canonical_obs[mapping.sampleColumn].astype(str)
        else:
            canonical_obs["sample_key"] = "SMP_001"

        if mapping is not None:
            if mapping.conditionColumn and mapping.conditionColumn in canonical_obs.columns:
                canonical_obs["condition"] = canonical_obs[mapping.conditionColumn].astype(str)
            if mapping.batchColumn and mapping.batchColumn in canonical_obs.columns:
                canonical_obs["batch"] = canonical_obs[mapping.batchColumn].astype(str)
            if (
                mapping.biologicalReplicateColumn
                and mapping.biologicalReplicateColumn in canonical_obs.columns
            ):
                canonical_obs["biological_replicate"] = canonical_obs[
                    mapping.biologicalReplicateColumn
                ].astype(str)
            if (
                mapping.technicalReplicateColumn
                and mapping.technicalReplicateColumn in canonical_obs.columns
            ):
                canonical_obs["technical_replicate"] = canonical_obs[
                    mapping.technicalReplicateColumn
                ].astype(str)

        # 5. Build canonical AnnData via format-neutral Step D builder
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

        # 6. Validate canonical invariants
        validate_canonical_anndata(
            canonical_adata,
            expected_organism=organism,
            expected_gene_id_scheme=gene_id_scheme,
        )

        # 7. Serialize canonical AnnData to disk
        os.makedirs(dest_path.parent, exist_ok=True)
        canonical_adata.write_h5ad(dest_path)

        # 8. Compute exact canonical SHA-256 of written artifact
        canonical_checksum = compute_file_sha256(dest_path)

        return CanonicalizationResult(
            sourceFormat=SourceFormat.TENX_MTX,
            canonicalFormat="H5AD",
            canonicalChecksum=canonical_checksum,
            organism=organism,
            geneIdScheme=gene_id_scheme,
            cellCount=int(canonical_adata.n_obs),
            geneCount=int(canonical_adata.n_vars),
            warnings=[],
            referenceBuild=reference_build,
        )


__all__ = ["TenxMtxImporter"]
