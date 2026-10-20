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
    issues = validate_import_source_set_cardinality(source_set)
    assert len(issues) == 0


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
    issues = validate_import_source_set_cardinality(source_set)
    assert len(issues) == 0


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
    issues = validate_import_source_set_cardinality(source_set)
    assert len(issues) == 0


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


def test_invalid_cardinalities(tmp_path: Path) -> None:
    """Missing required roles or extraneous roles produce blocking issues."""
    f_mat = ImportSourceFile(
        dataset_file_id="dsf_m",
        role=ImportFileRole.MATRIX,
        local_path=tmp_path / "matrix.mtx",
    )
    f_prim = ImportSourceFile(
        dataset_file_id="dsf_p",
        role=ImportFileRole.PRIMARY,
        local_path=tmp_path / "extra.h5ad",
    )

    # H5AD missing PRIMARY
    s1 = ImportSourceSet(SourceFormat.H5AD, (f_mat,))
    issues1 = validate_import_source_set_cardinality(s1)
    assert len(issues1) == 1
    assert issues1[0].code == ScientificIssueCode.INVALID_IMPORT_CARDINALITY

    # TENX_MTX with extra PRIMARY file
    s2 = ImportSourceSet(SourceFormat.TENX_MTX, (f_mat, f_prim))
    issues2 = validate_import_source_set_cardinality(s2)
    assert len(issues2) == 1
    assert issues2[0].code == ScientificIssueCode.INVALID_IMPORT_CARDINALITY
