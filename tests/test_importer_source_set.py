"""Unit tests for Phase 3 Step D ImportSourceSet role cardinality validation."""

from __future__ import annotations

from pathlib import Path

from cellweave_science.contracts import ImportFileRole, ScientificIssueCode, SourceFormat
from cellweave_science.importers.sources import ImportSourceFile, ImportSourceSet
from cellweave_science.importers.validation import validate_import_source_set_cardinality


def test_h5ad_valid_cardinality(tmp_path: Path) -> None:
    """H5AD with 1 PRIMARY file passes validation."""
    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(
            ImportSourceFile(
                dataset_file_id="dsf_1",
                role=ImportFileRole.PRIMARY,
                local_path=tmp_path / "matrix.h5ad",
            ),
        ),
    )
    assert len(validate_import_source_set_cardinality(source_set)) == 0


def test_tenx_h5_valid_cardinality(tmp_path: Path) -> None:
    """TENX_H5 with 1 PRIMARY file passes validation."""
    source_set = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(
            ImportSourceFile(
                dataset_file_id="dsf_1",
                role=ImportFileRole.PRIMARY,
                local_path=tmp_path / "filtered_feature_bc_matrix.h5",
            ),
        ),
    )
    assert len(validate_import_source_set_cardinality(source_set)) == 0


def test_tenx_mtx_valid_cardinality(tmp_path: Path) -> None:
    """TENX_MTX with 1 MATRIX, 1 FEATURES, 1 BARCODES passes validation."""
    source_set = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile(
                dataset_file_id="dsf_m",
                role=ImportFileRole.MATRIX,
                local_path=tmp_path / "matrix.mtx.gz",
            ),
            ImportSourceFile(
                dataset_file_id="dsf_f",
                role=ImportFileRole.FEATURES,
                local_path=tmp_path / "features.tsv.gz",
            ),
            ImportSourceFile(
                dataset_file_id="dsf_b",
                role=ImportFileRole.BARCODES,
                local_path=tmp_path / "barcodes.tsv.gz",
            ),
        ),
    )
    assert len(validate_import_source_set_cardinality(source_set)) == 0


def test_generic_matrix_valid_combinations(tmp_path: Path) -> None:
    """GENERIC_MATRIX supports matrix only, +cell metadata, +feature metadata, or +both."""
    f_mat = ImportSourceFile(
        dataset_file_id="dsf_m",
        role=ImportFileRole.MATRIX,
        local_path=tmp_path / "counts.csv",
    )
    f_cell = ImportSourceFile(
        dataset_file_id="dsf_c",
        role=ImportFileRole.CELL_METADATA,
        local_path=tmp_path / "cells.csv",
    )
    f_feat = ImportSourceFile(
        dataset_file_id="dsf_f",
        role=ImportFileRole.FEATURE_METADATA,
        local_path=tmp_path / "genes.csv",
    )

    # 1. Matrix only
    assert (
        len(
            validate_import_source_set_cardinality(
                ImportSourceSet(SourceFormat.GENERIC_MATRIX, (f_mat,))
            )
        )
        == 0
    )

    # 2. Matrix + cell metadata
    assert (
        len(
            validate_import_source_set_cardinality(
                ImportSourceSet(SourceFormat.GENERIC_MATRIX, (f_mat, f_cell))
            )
        )
        == 0
    )

    # 3. Matrix + feature metadata
    assert (
        len(
            validate_import_source_set_cardinality(
                ImportSourceSet(SourceFormat.GENERIC_MATRIX, (f_mat, f_feat))
            )
        )
        == 0
    )

    # 4. Matrix + both metadata
    assert (
        len(
            validate_import_source_set_cardinality(
                ImportSourceSet(SourceFormat.GENERIC_MATRIX, (f_mat, f_cell, f_feat))
            )
        )
        == 0
    )


def test_h5ad_invalid_cardinality_matrix(tmp_path: Path) -> None:
    """Test H5AD invalid cardinalities: 0 PRIMARY, 2 PRIMARY, extra role."""
    f_prim1 = ImportSourceFile("dsf_1", ImportFileRole.PRIMARY, tmp_path / "m1.h5ad")
    f_prim2 = ImportSourceFile("dsf_2", ImportFileRole.PRIMARY, tmp_path / "m2.h5ad")
    f_mat = ImportSourceFile("dsf_m", ImportFileRole.MATRIX, tmp_path / "m.mtx")

    # 0 PRIMARY
    s0 = ImportSourceSet(SourceFormat.H5AD, (f_mat,))
    issues0 = validate_import_source_set_cardinality(s0)
    assert len(issues0) == 1 and issues0[0].code == ScientificIssueCode.INVALID_IMPORT_CARDINALITY

    # 2 PRIMARY
    s2 = ImportSourceSet(SourceFormat.H5AD, (f_prim1, f_prim2))
    issues2 = validate_import_source_set_cardinality(s2)
    assert len(issues2) == 1 and issues2[0].code == ScientificIssueCode.INVALID_IMPORT_CARDINALITY

    # Extra role (PRIMARY + MATRIX)
    se = ImportSourceSet(SourceFormat.H5AD, (f_prim1, f_mat))
    issues_e = validate_import_source_set_cardinality(se)
    assert len(issues_e) == 1 and issues_e[0].code == ScientificIssueCode.INVALID_IMPORT_CARDINALITY


def test_tenx_h5_invalid_cardinality_matrix(tmp_path: Path) -> None:
    """Test TENX_H5 invalid cardinalities: 0 PRIMARY, 2 PRIMARY, extra role."""
    f_prim1 = ImportSourceFile("dsf_1", ImportFileRole.PRIMARY, tmp_path / "f1.h5")
    f_prim2 = ImportSourceFile("dsf_2", ImportFileRole.PRIMARY, tmp_path / "f2.h5")
    f_bar = ImportSourceFile("dsf_b", ImportFileRole.BARCODES, tmp_path / "b.tsv")

    # 0 PRIMARY
    s0 = ImportSourceSet(SourceFormat.TENX_H5, (f_bar,))
    assert (
        validate_import_source_set_cardinality(s0)[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # 2 PRIMARY
    s2 = ImportSourceSet(SourceFormat.TENX_H5, (f_prim1, f_prim2))
    assert (
        validate_import_source_set_cardinality(s2)[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # Extra role
    se = ImportSourceSet(SourceFormat.TENX_H5, (f_prim1, f_bar))
    assert (
        validate_import_source_set_cardinality(se)[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )


def test_tenx_mtx_invalid_cardinality_matrix(tmp_path: Path) -> None:
    """Test TENX_MTX invalid cardinalities: missing MATRIX, FEATURES, BARCODES, duplicates, extra metadata."""
    f_mat1 = ImportSourceFile("m1", ImportFileRole.MATRIX, tmp_path / "m1.mtx.gz")
    f_mat2 = ImportSourceFile("m2", ImportFileRole.MATRIX, tmp_path / "m2.mtx.gz")
    f_feat1 = ImportSourceFile("f1", ImportFileRole.FEATURES, tmp_path / "f1.tsv.gz")
    f_feat2 = ImportSourceFile("f2", ImportFileRole.FEATURES, tmp_path / "f2.tsv.gz")
    f_bar1 = ImportSourceFile("b1", ImportFileRole.BARCODES, tmp_path / "b1.tsv.gz")
    f_bar2 = ImportSourceFile("b2", ImportFileRole.BARCODES, tmp_path / "b2.tsv.gz")
    f_cell = ImportSourceFile("c", ImportFileRole.CELL_METADATA, tmp_path / "meta.csv")

    # Missing MATRIX
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.TENX_MTX, (f_feat1, f_bar1))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # Missing FEATURES
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.TENX_MTX, (f_mat1, f_bar1))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # Missing BARCODES
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.TENX_MTX, (f_mat1, f_feat1))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # Duplicate MATRIX
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.TENX_MTX, (f_mat1, f_mat2, f_feat1, f_bar1))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # Duplicate FEATURES
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.TENX_MTX, (f_mat1, f_feat1, f_feat2, f_bar1))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # Duplicate BARCODES
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.TENX_MTX, (f_mat1, f_feat1, f_bar1, f_bar2))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # Extra metadata role in 10x MTX
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.TENX_MTX, (f_mat1, f_feat1, f_bar1, f_cell))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )


def test_generic_matrix_invalid_cardinality_matrix(tmp_path: Path) -> None:
    """Test GENERIC_MATRIX invalid cardinalities: 0 MATRIX, 2 MATRIX, 2 CELL_METADATA, 2 FEATURE_METADATA, invalid roles."""
    f_mat1 = ImportSourceFile("m1", ImportFileRole.MATRIX, tmp_path / "c1.csv")
    f_mat2 = ImportSourceFile("m2", ImportFileRole.MATRIX, tmp_path / "c2.csv")
    f_cell1 = ImportSourceFile("c1", ImportFileRole.CELL_METADATA, tmp_path / "cell1.csv")
    f_cell2 = ImportSourceFile("c2", ImportFileRole.CELL_METADATA, tmp_path / "cell2.csv")
    f_feat1 = ImportSourceFile("f1", ImportFileRole.FEATURE_METADATA, tmp_path / "feat1.csv")
    f_feat2 = ImportSourceFile("f2", ImportFileRole.FEATURE_METADATA, tmp_path / "feat2.csv")
    f_prim = ImportSourceFile("p", ImportFileRole.PRIMARY, tmp_path / "p.h5ad")
    f_bar = ImportSourceFile("b", ImportFileRole.BARCODES, tmp_path / "b.tsv")

    # 0 MATRIX
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.GENERIC_MATRIX, (f_cell1,))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # 2 MATRIX
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.GENERIC_MATRIX, (f_mat1, f_mat2))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # 2 CELL_METADATA
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.GENERIC_MATRIX, (f_mat1, f_cell1, f_cell2))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # 2 FEATURE_METADATA
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.GENERIC_MATRIX, (f_mat1, f_feat1, f_feat2))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # PRIMARY present
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.GENERIC_MATRIX, (f_mat1, f_prim))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )

    # BARCODES present
    assert (
        validate_import_source_set_cardinality(
            ImportSourceSet(SourceFormat.GENERIC_MATRIX, (f_mat1, f_bar))
        )[0].code
        == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
    )
