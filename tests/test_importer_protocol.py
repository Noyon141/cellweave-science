"""Tests for DatasetImporter protocol and ImportSourceSet structures."""

from __future__ import annotations

from pathlib import Path

import pytest

from cellweave_science.contracts import (
    CanonicalizationResult,
    GeneIdScheme,
    ImportFileRole,
    ImportInspectionResult,
    InspectionStatus,
    MatrixOrientation,
    MetadataMappingSubmission,
    Organism,
    SampleMappingEntry,
    ScientificIssue,
    SourceFormat,
)
from cellweave_science.importers import DatasetImporter, ImportSourceFile, ImportSourceSet


class MockImporter:
    """Mock implementation conforming to DatasetImporter protocol."""

    def detect(self, source_set: ImportSourceSet) -> bool:
        return bool(
            source_set.source_format
            in (
                SourceFormat.H5AD,
                SourceFormat.TENX_H5,
                SourceFormat.TENX_MTX,
                SourceFormat.GENERIC_MATRIX,
            )
        )

    def validate(self, _source_set: ImportSourceSet) -> list[ScientificIssue]:
        return []

    def inspect(self, source_set: ImportSourceSet) -> ImportInspectionResult:
        return ImportInspectionResult(
            datasetVersionId="dsv_018f6c52-7e23-7185-9856-f046dc0320a2",
            sourceFormat=source_set.source_format,
            status=InspectionStatus.VALID,
            cellCountEstimate=1000,
            geneCountEstimate=20000,
            detectedOrganism=Organism.HUMAN,
            detectedGeneIdScheme=GeneIdScheme.ENSEMBL,
            referenceBuildCandidate="GRCh38",
            requiresMapping=False,
            availableCellMetadataColumns=["batch", "sample"],
            availableFeatureMetadataColumns=["gene_id", "gene_symbol"],
            issues=[],
        )

    def canonicalize(
        self,
        source_set: ImportSourceSet,
        output_path: str,
        organism: Organism,
        gene_id_scheme: GeneIdScheme,
        reference_build: str | None = None,
        _mapping: MetadataMappingSubmission | None = None,
    ) -> CanonicalizationResult:
        return CanonicalizationResult(
            sourceFormat=source_set.source_format,
            canonicalFormat="H5AD",
            canonicalChecksum="a" * 64,
            organism=organism,
            geneIdScheme=gene_id_scheme,
            referenceBuild=reference_build,
            cellCount=1000,
            geneCount=20000,
            warnings=[],
            localArtifactPath=output_path,
        )


@pytest.mark.unit
def test_dataset_importer_protocol_conformance() -> None:
    importer = MockImporter()
    assert isinstance(importer, DatasetImporter)


@pytest.mark.unit
def test_h5ad_single_file_source_set(tmp_path: Path) -> None:
    test_file = tmp_path / "matrix.h5ad"
    test_file.touch()

    source_file = ImportSourceFile(
        dataset_file_id="dsf_001",
        role=ImportFileRole.PRIMARY,
        local_path=test_file,
    )
    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(source_file,),
    )

    primary = source_set.get_primary_file()
    assert primary.dataset_file_id == "dsf_001"
    assert primary.role == ImportFileRole.PRIMARY
    assert primary.local_path == test_file


@pytest.mark.unit
def test_tenx_h5_single_file_source_set(tmp_path: Path) -> None:
    test_file = tmp_path / "filtered_feature_bc_matrix.h5"
    test_file.touch()

    source_file = ImportSourceFile(
        dataset_file_id="dsf_h5",
        role=ImportFileRole.PRIMARY,
        local_path=test_file,
    )
    source_set = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(source_file,),
    )

    primary = source_set.get_primary_file()
    assert primary.dataset_file_id == "dsf_h5"
    assert primary.role == ImportFileRole.PRIMARY
    assert primary.local_path == test_file


@pytest.mark.unit
def test_tenx_mtx_multi_file_source_set(tmp_path: Path) -> None:
    mtx_file = tmp_path / "matrix.mtx.gz"
    feat_file = tmp_path / "features.tsv.gz"
    bar_file = tmp_path / "barcodes.tsv.gz"
    for f in (mtx_file, feat_file, bar_file):
        f.touch()

    source_set = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile(
                dataset_file_id="dsf_mtx",
                role=ImportFileRole.MATRIX,
                local_path=mtx_file,
            ),
            ImportSourceFile(
                dataset_file_id="dsf_feat",
                role=ImportFileRole.FEATURES,
                local_path=feat_file,
            ),
            ImportSourceFile(
                dataset_file_id="dsf_bar",
                role=ImportFileRole.BARCODES,
                local_path=bar_file,
            ),
        ),
    )

    matrix = source_set.get_single_file_by_role(ImportFileRole.MATRIX)
    assert matrix.dataset_file_id == "dsf_mtx"

    features = source_set.get_single_file_by_role(ImportFileRole.FEATURES)
    assert features.dataset_file_id == "dsf_feat"

    barcodes = source_set.get_single_file_by_role(ImportFileRole.BARCODES)
    assert barcodes.dataset_file_id == "dsf_bar"


@pytest.mark.unit
def test_generic_matrix_with_metadata_source_set(tmp_path: Path) -> None:
    mtx_file = tmp_path / "counts.csv"
    cell_meta = tmp_path / "cell_metadata.tsv"
    gene_meta = tmp_path / "gene_metadata.tsv"
    for f in (mtx_file, cell_meta, gene_meta):
        f.touch()

    source_set = ImportSourceSet(
        source_format=SourceFormat.GENERIC_MATRIX,
        files=(
            ImportSourceFile(
                dataset_file_id="dsf_mat",
                role=ImportFileRole.MATRIX,
                local_path=mtx_file,
            ),
            ImportSourceFile(
                dataset_file_id="dsf_cell_meta",
                role=ImportFileRole.CELL_METADATA,
                local_path=cell_meta,
            ),
            ImportSourceFile(
                dataset_file_id="dsf_feat_meta",
                role=ImportFileRole.FEATURE_METADATA,
                local_path=gene_meta,
            ),
        ),
    )

    matrix = source_set.get_single_file_by_role(ImportFileRole.MATRIX)
    assert matrix.dataset_file_id == "dsf_mat"

    cell_m = source_set.get_single_file_by_role(ImportFileRole.CELL_METADATA)
    assert cell_m.dataset_file_id == "dsf_cell_meta"

    feat_m = source_set.get_single_file_by_role(ImportFileRole.FEATURE_METADATA)
    assert feat_m.dataset_file_id == "dsf_feat_meta"


@pytest.mark.unit
def test_mock_importer_inspection_and_canonicalization(tmp_path: Path) -> None:
    importer = MockImporter()
    test_file = tmp_path / "test.h5ad"
    test_file.touch()

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(
            ImportSourceFile(
                dataset_file_id="dsf_001",
                role=ImportFileRole.PRIMARY,
                local_path=test_file,
            ),
        ),
    )

    assert importer.detect(source_set) is True
    inspection = importer.inspect(source_set)
    assert inspection.status == InspectionStatus.VALID
    assert inspection.cellCountEstimate == 1000
    assert "batch" in inspection.availableCellMetadataColumns
    assert "gene_symbol" in inspection.availableFeatureMetadataColumns

    out_file = str(tmp_path / "canonical.h5ad")
    mapping = MetadataMappingSubmission(
        datasetVersionId="dsv_018f6c52-7e23-7185-9856-f046dc0320a2",
        organism=Organism.HUMAN,
        geneIdScheme=GeneIdScheme.ENSEMBL,
        referenceBuild="GRCh38",
        orientation=MatrixOrientation.CELLS_BY_GENES,
        cellIdColumn="cell_id",
        geneIdColumn="gene_id",
        geneSymbolColumn="gene_symbol",
        sampleColumn="sample",
        conditionColumn="disease",
        batchColumn="batch",
        biologicalReplicateColumn="bio_rep",
        technicalReplicateColumn="tech_rep",
        sampleMappings=[
            SampleMappingEntry(sourceValue="S1", sampleKey="ctrl_rep1"),
        ],
    )
    canon = importer.canonicalize(
        source_set=source_set,
        output_path=out_file,
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        reference_build="GRCh38",
        _mapping=mapping,
    )
    assert canon.canonicalFormat == "H5AD"
    assert canon.cellCount == 1000
    assert canon.canonicalChecksum == "a" * 64
