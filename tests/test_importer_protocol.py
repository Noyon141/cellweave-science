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
    MetadataMappingSubmission,
    Organism,
    ScientificIssue,
    SourceFormat,
)
from cellweave_science.importers import DatasetImporter, ImportSourceFile, ImportSourceSet


class MockImporter:
    """Mock implementation conforming to DatasetImporter protocol."""

    def detect(self, source_set: ImportSourceSet) -> bool:
        return bool(source_set.source_format == SourceFormat.H5AD)

    def validate(self, _source_set: ImportSourceSet) -> list[ScientificIssue]:
        return []

    def inspect(self, _source_set: ImportSourceSet) -> ImportInspectionResult:
        return ImportInspectionResult(
            datasetVersionId="dsv_018f6c52-7e23-7185-9856-f046dc0320a2",
            sourceFormat=SourceFormat.H5AD,
            status=InspectionStatus.VALID,
            cellCountEstimate=1000,
            geneCountEstimate=20000,
            detectedOrganism=Organism.HUMAN,
            detectedGeneIdScheme=GeneIdScheme.ENSEMBL,
            referenceBuildCandidate="GRCh38",
            requiresMapping=False,
            availableMetadataColumns=["batch", "sample"],
            issues=[],
        )

    def canonicalize(
        self,
        _source_set: ImportSourceSet,
        output_path: str,
        organism: Organism,
        gene_id_scheme: GeneIdScheme,
        reference_build: str | None = None,
        _mapping: MetadataMappingSubmission | None = None,
    ) -> CanonicalizationResult:
        return CanonicalizationResult(
            sourceFormat=SourceFormat.H5AD,
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
def test_single_file_source_set(tmp_path: Path) -> None:
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
def test_multi_file_source_set(tmp_path: Path) -> None:
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

    out_file = str(tmp_path / "canonical.h5ad")
    canon = importer.canonicalize(
        _source_set=source_set,
        output_path=out_file,
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        reference_build="GRCh38",
    )
    assert canon.canonicalFormat == "H5AD"
    assert canon.cellCount == 1000
    assert canon.canonicalChecksum == "a" * 64
