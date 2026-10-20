"""Comprehensive unit, golden, and regression tests for Phase 3 Step G / G.2 Generic Matrix Importer:
- GenericMatrixImporter (GENERIC_MATRIX)

Test matrix:
1. Protocol conformance & Configuration Safety (satisfies DatasetImporter, fail-closed limits).
2. Delimiter & Container Sniffing (CSV, TSV, MTX, ambiguous delimiters, comment handling).
3. Delimiter Extension-Agnostic Policy (content-based detection; .csv vs .tsv produce identical interpretations).
4. Format Detection & Cardinality (valid roles MATRIX, optional CELL_METADATA / FEATURE_METADATA).
5. Validation logic (structural errors, non-numeric tables, count anomalies, duplicate column headers).
6. Inspection logic (datasetVersionId authority, NEEDS_MAPPING vs INVALID status, column discovery).
7. Golden CSV Tests (CELLS_BY_GENES via matrixCellIdColumn & GENES_BY_CELLS via matrixFeatureIdColumn).
8. Golden TSV Tests (plain and bounded gzip).
9. Golden MTX Tests (CELLS_BY_GENES & GENES_BY_CELLS with explicit cellAxisIndexColumn & featureAxisIndexColumn).
10. Reordered MTX Metadata Golden Test (Section 16: non-square 3x2 GENES_BY_CELLS with reordered metadata).
11. MTX Axis Coordinate Error Tests (Section 17: zero, negative, fractional, duplicate, missing, gap, out-of-range, non-numeric, blank/null).
12. MTX Row-Order Independence Tests (Section 18: multiple permutations produce identical canonical output).
13. CSV Misleading-First-Column Regression Tests (Section 19: 'batch,ENSG1,ENSG2' without/with matrixCellIdColumn).
14. GENES_BY_CELLS Explicit Feature Row Tests (Section 20: 'probe_label,CELL_A,CELL_B' without/with matrixFeatureIdColumn).
15. Feature metadata join test matrix (explicit featureMetadataJoinColumn, reordered, duplicate, missing, extra rows).
16. Cell metadata join test matrix (explicit cellMetadataJoinColumn, translated cellIdColumn, reordered, duplicate, missing, extra rows).
17. Strict Sample Mapping Authority (no source pass-through, complete mapping requirement).
18. Biological annotation mappings (condition, batch, biological/technical replicates).
19. Canonical obs schema tests (complete mapping vs minimal mapping; no axis index helper columns persisted).
20. Canonical var schema tests (stable IDs, source_feature_id, gene_symbol only when mapped; no axis index columns persisted).
21. Stable ID + duplicate symbol test (valid when stable IDs are unique).
22. Organism, Gene ID Scheme, & Reference build resolution tests.
23. Raw count strictness (integer, integral floats, fractional, negative, NaN/Inf, ambiguous near-integral).
24. Policy B canonicalization slot hygiene (CSR in X, empty layers/obsm/varm, controlled uns).
25. Source immutability & path collision safety.
26. Exact round-trip assertions for CSV and MTX fixtures.
27. RangeIndex & Header Authority regression tests.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellweave_science.contracts import (
    GeneIdScheme,
    ImportFileRole,
    InspectionStatus,
    MatrixOrientation,
    MetadataMappingSubmission,
    Organism,
    SampleMappingEntry,
    ScientificIssueCode,
    SourceFormat,
)
from cellweave_science.importers import (
    DatasetImporter,
    GenericMatrixConfig,
    GenericMatrixImporter,
    ImportSourceFile,
    ImportSourceSet,
    compute_file_sha256,
    read_delimited_table,
    sniff_delimiter,
)
from cellweave_science.importers.canonical import validate_canonical_anndata
from cellweave_science.importers.errors import (
    CountValidationError,
    ImporterSecurityError,
)
from cellweave_science.importers.security import DecompressionLimits

DEFAULT_TEST_LIMITS = DecompressionLimits(
    max_uncompressed_bytes=100 * 1024 * 1024,
    max_compression_ratio=200.0,
)
DEFAULT_TEST_CONFIG = GenericMatrixConfig(decompression_limits=DEFAULT_TEST_LIMITS)


# =========================================================================
# HELPER FUNCTIONS TO BUILD SYNTHETIC INPUTS & MAPPINGS
# =========================================================================


def create_csv_matrix(
    file_path: Path,
    data: list[list[Any]],
    headers: list[str],
    *,
    sep: str = ",",
    compress: bool = False,
) -> Path:
    """Create a delimited matrix file (CSV or TSV, plain or gzip)."""
    df = pd.DataFrame(data, columns=headers)
    if compress:
        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            df.to_csv(f, sep=sep, index=False)
    else:
        df.to_csv(file_path, sep=sep, index=False)
    return file_path


def create_mtx_file(
    file_path: Path,
    n_rows: int,
    n_cols: int,
    entries: list[tuple[int, int, int | float]],
    *,
    compress: bool = False,
) -> Path:
    """Create a Matrix Market coordinate file."""
    lines = [
        "%%MatrixMarket matrix coordinate integer general\n",
        f"{n_rows} {n_cols} {len(entries)}\n",
    ]
    for r, c, v in entries:
        lines.append(f"{r} {c} {v}\n")
    content = "".join(lines)

    if compress:
        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        file_path.write_text(content, encoding="utf-8")
    return file_path


def create_source_file(path: Path, role: ImportFileRole) -> ImportSourceFile:
    return ImportSourceFile(
        dataset_file_id=f"file-{path.name}",
        role=role,
        local_path=path,
    )


def create_source_set(
    files: list[ImportSourceFile],
    dataset_version_id: str = "ds-ver-test-001",
    source_format: SourceFormat = SourceFormat.GENERIC_MATRIX,
) -> ImportSourceSet:
    return ImportSourceSet(
        source_format=source_format,
        files=tuple(files),
        dataset_version_id=dataset_version_id,
    )


def make_mapping(
    *,
    dataset_version_id: str = "ds-ver-test-001",
    organism: Organism = Organism.HUMAN,
    gene_id_scheme: GeneIdScheme = GeneIdScheme.ENSEMBL,
    orientation: MatrixOrientation | None = None,
    cell_id_column: str | None = None,
    gene_id_column: str | None = None,
    gene_symbol_column: str | None = None,
    matrix_cell_id_column: str | None = None,
    matrix_feature_id_column: str | None = None,
    cell_metadata_join_column: str | None = None,
    feature_metadata_join_column: str | None = None,
    cell_axis_index_column: str | None = None,
    feature_axis_index_column: str | None = None,
    sample_column: str | None = None,
    sample_mappings: list[SampleMappingEntry] | None = None,
    condition_column: str | None = None,
    batch_column: str | None = None,
    biological_replicate_column: str | None = None,
    technical_replicate_column: str | None = None,
    reference_build: str | None = None,
) -> MetadataMappingSubmission:
    """Helper to instantiate fully valid MetadataMappingSubmission with explicit Step G.1 fields."""
    return MetadataMappingSubmission(
        datasetVersionId=dataset_version_id,
        organism=organism,
        geneIdScheme=gene_id_scheme,
        orientation=orientation,
        cellIdColumn=cell_id_column,
        geneIdColumn=gene_id_column,
        geneSymbolColumn=gene_symbol_column,
        matrixCellIdColumn=matrix_cell_id_column,
        matrixFeatureIdColumn=matrix_feature_id_column,
        cellMetadataJoinColumn=cell_metadata_join_column,
        featureMetadataJoinColumn=feature_metadata_join_column,
        cellAxisIndexColumn=cell_axis_index_column,
        featureAxisIndexColumn=feature_axis_index_column,
        sampleColumn=sample_column,
        sampleMappings=sample_mappings,
        conditionColumn=condition_column,
        batchColumn=batch_column,
        biologicalReplicateColumn=biological_replicate_column,
        technicalReplicateColumn=technical_replicate_column,
        referenceBuild=reference_build,
    )


# =========================================================================
# 1. PROTOCOL CONFORMANCE & CONFIGURATION SAFETY
# =========================================================================


def test_generic_matrix_importer_protocol_conformance() -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    assert isinstance(importer, DatasetImporter)
    assert hasattr(importer, "detect")
    assert hasattr(importer, "validate")
    assert hasattr(importer, "inspect")
    assert hasattr(importer, "canonicalize")


def test_generic_matrix_config_fail_closed() -> None:
    with pytest.raises(TypeError, match="decompression_limits"):
        GenericMatrixConfig(decompression_limits=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="config must be an instance of GenericMatrixConfig"):
        GenericMatrixImporter(None)  # type: ignore[arg-type]


# =========================================================================
# 2. DELIMITER & CONTAINER SNIFFING
# =========================================================================


def test_sniff_delimiter_comma_and_tab(tmp_path: Path) -> None:
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("cell_id,ENSG000001,ENSG000002\ncell_1,10,20\n", encoding="utf-8")
    assert sniff_delimiter(csv_file, DEFAULT_TEST_LIMITS) == ","

    tsv_file = tmp_path / "test.tsv"
    tsv_file.write_text("cell_id\tENSG000001\tENSG000002\ncell_1\t10\t20\n", encoding="utf-8")
    assert sniff_delimiter(tsv_file, DEFAULT_TEST_LIMITS) == "\t"


def test_sniff_delimiter_gzip(tmp_path: Path) -> None:
    gz_csv = tmp_path / "test.csv.gz"
    with gzip.open(gz_csv, "wt", encoding="utf-8") as f:
        f.write("cell_id,g1,g2\nc1,5,10\n")
    assert sniff_delimiter(gz_csv, DEFAULT_TEST_LIMITS) == ","


def test_sniff_delimiter_empty_or_comments_fails_closed(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(CountValidationError) as exc:
        sniff_delimiter(empty_file, DEFAULT_TEST_LIMITS)
    assert exc.value.issue_code == ScientificIssueCode.MALFORMED_INPUT

    comments_only = tmp_path / "comments.txt"
    comments_only.write_text("# header\n# another comment\n", encoding="utf-8")
    with pytest.raises(CountValidationError) as exc:
        sniff_delimiter(comments_only, DEFAULT_TEST_LIMITS)
    assert exc.value.issue_code == ScientificIssueCode.MALFORMED_INPUT


def test_sniff_delimiter_ambiguous_fails_closed(tmp_path: Path) -> None:
    ambig_file = tmp_path / "ambig.txt"
    ambig_file.write_text("cell_id\tcol1,col2\n", encoding="utf-8")
    with pytest.raises(CountValidationError) as exc:
        sniff_delimiter(ambig_file, DEFAULT_TEST_LIMITS)
    assert exc.value.issue_code == ScientificIssueCode.FORMAT_MISMATCH


# =========================================================================
# 3. DELIMITER EXTENSION-AGNOSTIC REGRESSION TESTS
# =========================================================================


def test_delimiter_extension_agnostic_single_column(tmp_path: Path) -> None:
    """Content-based detection regression test:

    Plain single-column content named .csv vs named .tsv must not produce two different
    semantic interpretations merely because of the extension.
    """
    content = "token_column\nval_1\nval_2\nval_3\n"
    csv_file = tmp_path / "single_column.csv"
    tsv_file = tmp_path / "single_column.tsv"

    csv_file.write_text(content, encoding="utf-8")
    tsv_file.write_text(content, encoding="utf-8")

    delim_csv = sniff_delimiter(csv_file, DEFAULT_TEST_LIMITS)
    delim_tsv = sniff_delimiter(tsv_file, DEFAULT_TEST_LIMITS)
    assert delim_csv == delim_tsv

    df_csv, _ = read_delimited_table(csv_file, DEFAULT_TEST_LIMITS)
    df_tsv, _ = read_delimited_table(tsv_file, DEFAULT_TEST_LIMITS)
    pd.testing.assert_frame_equal(df_csv, df_tsv)
    assert list(df_csv.columns) == ["token_column"]
    assert df_csv["token_column"].tolist() == ["val_1", "val_2", "val_3"]

    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    issues_csv = importer.validate(
        create_source_set([create_source_file(csv_file, ImportFileRole.MATRIX)])
    )
    issues_tsv = importer.validate(
        create_source_set([create_source_file(tsv_file, ImportFileRole.MATRIX)])
    )

    assert any(iss.code == ScientificIssueCode.MALFORMED_INPUT for iss in issues_csv)
    assert any(iss.code == ScientificIssueCode.MALFORMED_INPUT for iss in issues_tsv)
    assert [iss.code for iss in issues_csv] == [iss.code for iss in issues_tsv]


# =========================================================================
# 4. FORMAT DETECTION & CARDINALITY
# =========================================================================


def test_detect_supported_formats(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    csv_file = tmp_path / "mat.csv"
    csv_file.write_text("cell_id,g1,g2\nc1,1,2\n", encoding="utf-8")
    source_set_csv = create_source_set([create_source_file(csv_file, ImportFileRole.MATRIX)])
    assert importer.detect(source_set_csv) is True

    tsv_file = tmp_path / "mat.tsv"
    tsv_file.write_text("cell_id\tg1\tg2\nc1\t1\t2\n", encoding="utf-8")
    source_set_tsv = create_source_set([create_source_file(tsv_file, ImportFileRole.MATRIX)])
    assert importer.detect(source_set_tsv) is True

    mtx_file = tmp_path / "mat.mtx"
    create_mtx_file(mtx_file, 2, 2, [(1, 1, 5)])
    source_set_mtx = create_source_set([create_source_file(mtx_file, ImportFileRole.MATRIX)])
    assert importer.detect(source_set_mtx) is True

    mtx_gz = tmp_path / "mat.mtx.gz"
    create_mtx_file(mtx_gz, 2, 2, [(1, 1, 5)], compress=True)
    source_set_mtx_gz = create_source_set([create_source_file(mtx_gz, ImportFileRole.MATRIX)])
    assert importer.detect(source_set_mtx_gz) is True


def test_detect_rejects_unsupported_formats_and_cardinality(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    csv_file = tmp_path / "mat.csv"
    csv_file.write_text("c,g\n1,2\n", encoding="utf-8")
    non_generic_set = create_source_set(
        [create_source_file(csv_file, ImportFileRole.MATRIX)],
        source_format=SourceFormat.H5AD,
    )
    assert importer.detect(non_generic_set) is False

    primary_set = create_source_set([create_source_file(csv_file, ImportFileRole.PRIMARY)])
    assert importer.detect(primary_set) is False

    bc_file = tmp_path / "barcodes.tsv"
    bc_file.write_text("c1\n", encoding="utf-8")
    invalid_roles_set = create_source_set(
        [
            create_source_file(csv_file, ImportFileRole.MATRIX),
            create_source_file(bc_file, ImportFileRole.BARCODES),
        ]
    )
    assert importer.detect(invalid_roles_set) is False

    csv2 = tmp_path / "mat2.csv"
    csv2.write_text("c,g\n1,2\n", encoding="utf-8")
    multi_matrix_set = create_source_set(
        [
            create_source_file(csv_file, ImportFileRole.MATRIX),
            create_source_file(csv2, ImportFileRole.MATRIX),
        ]
    )
    assert importer.detect(multi_matrix_set) is False


# =========================================================================
# 5. VALIDATION LOGIC & DUPLICATE IDENTIFIERS
# =========================================================================


def test_validate_cardinality_violations(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    f = tmp_path / "mat.csv"
    f.write_text("c,g\n1,2\n", encoding="utf-8")

    empty_set = create_source_set([])
    issues = importer.validate(empty_set)
    assert any(iss.code == ScientificIssueCode.MISSING_COMPONENT for iss in issues)

    f2 = tmp_path / "mat2.csv"
    f2.write_text("c,g\n1,2\n", encoding="utf-8")
    multi_set = create_source_set(
        [
            create_source_file(f, ImportFileRole.MATRIX),
            create_source_file(f2, ImportFileRole.MATRIX),
        ]
    )
    issues = importer.validate(multi_set)
    assert any(iss.code == ScientificIssueCode.INVALID_IMPORT_CARDINALITY for iss in issues)


def test_validate_non_numeric_and_empty_tables(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    bad_mat = tmp_path / "bad_mat.csv"
    bad_mat.write_text("cell_id,g1,g2\nc1,text,10\n", encoding="utf-8")
    source_set = create_source_set([create_source_file(bad_mat, ImportFileRole.MATRIX)])
    issues = importer.validate(source_set)
    assert any(iss.code == ScientificIssueCode.MALFORMED_INPUT for iss in issues)

    single_col = tmp_path / "single_col.csv"
    single_col.write_text("cell_id\nc1\nc2\n", encoding="utf-8")
    single_set = create_source_set([create_source_file(single_col, ImportFileRole.MATRIX)])
    issues = importer.validate(single_set)
    assert any(iss.code == ScientificIssueCode.MALFORMED_INPUT for iss in issues)


def test_validate_duplicate_authoritative_matrix_ids(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    # Duplicate column identifiers (row 0 header) detected in validate()
    dup_cols = tmp_path / "dup_cols.csv"
    dup_cols.write_text("cell_id,g1,g1\nc1,10,20\nc2,30,40\n", encoding="utf-8")
    issues_col = importer.validate(
        create_source_set([create_source_file(dup_cols, ImportFileRole.MATRIX)])
    )
    assert any(iss.code == ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS for iss in issues_col)

    # Duplicate row identifiers in explicit matrixCellIdColumn rejected in canonicalize()
    dup_rows = tmp_path / "dup_rows.csv"
    dup_rows.write_text(
        "cell_id,ENSG00000124762,ENSG00000142168\nc1,10,20\nc1,30,40\n", encoding="utf-8"
    )
    with pytest.raises((CountValidationError, ImporterSecurityError)) as exc_row:
        importer.canonicalize(
            source_set=create_source_set([create_source_file(dup_rows, ImportFileRole.MATRIX)]),
            output_path=str(tmp_path / "out_dup_row.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=make_mapping(
                orientation=MatrixOrientation.CELLS_BY_GENES,
                matrix_cell_id_column="cell_id",
            ),
        )
    assert isinstance(exc_row.value, CountValidationError | ImporterSecurityError)
    assert exc_row.value.issue_code == ScientificIssueCode.DUPLICATE_CELL_IDS


def test_validate_raw_counts(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    frac_mat = tmp_path / "frac_mat.csv"
    frac_mat.write_text("cell_id,g1,g2\nc1,1.5,2.0\n", encoding="utf-8")
    frac_set = create_source_set([create_source_file(frac_mat, ImportFileRole.MATRIX)])
    issues = importer.validate(frac_set)
    assert any(iss.code == ScientificIssueCode.NON_COUNT_INPUT for iss in issues)

    neg_mat = tmp_path / "neg_mat.csv"
    neg_mat.write_text("cell_id,g1,g2\nc1,-1,2\n", encoding="utf-8")
    neg_set = create_source_set([create_source_file(neg_mat, ImportFileRole.MATRIX)])
    issues = importer.validate(neg_set)
    assert any(iss.code == ScientificIssueCode.NON_COUNT_INPUT for iss in issues)

    ambig_mat = tmp_path / "ambig_mat.csv"
    ambig_val = repr(1.0 + np.finfo(np.float64).eps * 2.0)
    ambig_mat.write_text(f"cell_id,g1,g2\nc1,{ambig_val},2\n", encoding="utf-8")
    ambig_set = create_source_set([create_source_file(ambig_mat, ImportFileRole.MATRIX)])
    issues = importer.validate(ambig_set)
    assert any(iss.code == ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS for iss in issues)


# =========================================================================
# 6. INSPECTION LOGIC & datasetVersionId AUTHORITY
# =========================================================================


def test_inspect_dataset_version_id_authority_distinct_ids(tmp_path: Path) -> None:
    """Proves inspect() strictly uses source_set.dataset_version_id and never file IDs."""
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    csv_file = tmp_path / "mat.csv"
    csv_file.write_text("cell_id,g1,g2\nc1,10,20\n", encoding="utf-8")

    source_file = ImportSourceFile(
        dataset_file_id="dsf_generic_matrix_123",
        role=ImportFileRole.MATRIX,
        local_path=csv_file,
    )
    source_set = ImportSourceSet(
        source_format=SourceFormat.GENERIC_MATRIX,
        files=(source_file,),
        dataset_version_id="dsv_generic_version_456",
    )

    inspection = importer.inspect(source_set)
    assert inspection.datasetVersionId == "dsv_generic_version_456"
    assert inspection.datasetVersionId != "dsf_generic_matrix_123"


def test_inspect_dataset_version_id_missing_fails_closed(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    csv_file = tmp_path / "mat.csv"
    csv_file.write_text("cell_id,g1,g2\nc1,1,2\n", encoding="utf-8")

    source_set_no_ver = ImportSourceSet(
        source_format=SourceFormat.GENERIC_MATRIX,
        files=(create_source_file(csv_file, ImportFileRole.MATRIX),),
        dataset_version_id=None,
    )
    with pytest.raises(ImporterSecurityError) as exc:
        importer.inspect(source_set_no_ver)
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA

    source_set_blank_ver = ImportSourceSet(
        source_format=SourceFormat.GENERIC_MATRIX,
        files=(create_source_file(csv_file, ImportFileRole.MATRIX),),
        dataset_version_id="   ",
    )
    with pytest.raises(ImporterSecurityError) as exc:
        importer.inspect(source_set_blank_ver)
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


def test_inspect_needs_mapping_status_and_metadata_discovery(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    mat_file = tmp_path / "mat.csv"
    mat_file.write_text("cell_id,ENSG00000124762,ENSG00000142168\nc1,10,20\n", encoding="utf-8")

    cell_meta = tmp_path / "cell_metadata.csv"
    cell_meta.write_text("cell_id,sample_id,condition,batch\nc1,s1,treated,b1\n", encoding="utf-8")

    feat_meta = tmp_path / "feat_metadata.csv"
    feat_meta.write_text("gene_id,gene_symbol\nENSG00000124762,CDKN1A\n", encoding="utf-8")

    source_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
            create_source_file(feat_meta, ImportFileRole.FEATURE_METADATA),
        ]
    )

    inspection = importer.inspect(source_set)
    assert inspection.status == InspectionStatus.NEEDS_MAPPING
    assert inspection.requiresMapping is True
    assert any(iss.code == ScientificIssueCode.AMBIGUOUS_ORIENTATION for iss in inspection.issues)
    assert inspection.availableCellMetadataColumns == ["cell_id", "sample_id", "condition", "batch"]
    assert inspection.availableFeatureMetadataColumns == ["gene_id", "gene_symbol"]


# =========================================================================
# 7. INSPECTION STATUS MATRIX (Section 26)
# =========================================================================


def test_inspection_status_matrix(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    # 1. Orientation unresolved => NEEDS_MAPPING + AMBIGUOUS_ORIENTATION
    p_valid = tmp_path / "valid.csv"
    p_valid.write_text("cell_id,ENSG00000124762\nc1,10\n", encoding="utf-8")
    res1 = importer.inspect(create_source_set([create_source_file(p_valid, ImportFileRole.MATRIX)]))
    assert res1.status == InspectionStatus.NEEDS_MAPPING
    assert res1.requiresMapping is True
    assert any(iss.code == ScientificIssueCode.AMBIGUOUS_ORIENTATION for iss in res1.issues)

    # 2. Malformed matrix => INVALID
    p_malformed = tmp_path / "malformed.csv"
    p_malformed.write_text("cell_id,g1\nc1,not_a_number\n", encoding="utf-8")
    res2 = importer.inspect(
        create_source_set([create_source_file(p_malformed, ImportFileRole.MATRIX)])
    )
    assert res2.status == InspectionStatus.INVALID
    assert res2.requiresMapping is False

    # 3. Non-count matrix => INVALID
    p_frac = tmp_path / "frac.csv"
    p_frac.write_text("cell_id,g1\nc1,2.718\n", encoding="utf-8")
    res3 = importer.inspect(create_source_set([create_source_file(p_frac, ImportFileRole.MATRIX)]))
    assert res3.status == InspectionStatus.INVALID
    assert res3.requiresMapping is False

    # 4. Duplicate column headers => INVALID
    p_dup_hdr = tmp_path / "dup_hdr.csv"
    p_dup_hdr.write_text("cell_id,g1,g1\nc1,1,2\nc2,3,4\n", encoding="utf-8")
    res4 = importer.inspect(
        create_source_set([create_source_file(p_dup_hdr, ImportFileRole.MATRIX)])
    )
    assert res4.status == InspectionStatus.INVALID
    assert res4.requiresMapping is False
    assert any(iss.code == ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS for iss in res4.issues)

    # 5. Generic MTX missing CELL_METADATA / FEATURE_METADATA / axis mappings => NEEDS_MAPPING
    mtx_file = tmp_path / "mat.mtx"
    create_mtx_file(mtx_file, 2, 2, [(1, 1, 10)])
    res_mtx_no_meta = importer.inspect(
        create_source_set([create_source_file(mtx_file, ImportFileRole.MATRIX)])
    )
    assert res_mtx_no_meta.status == InspectionStatus.NEEDS_MAPPING
    assert res_mtx_no_meta.requiresMapping is True

    # 6. Generic MTX with malformed / duplicate axis coordinates => INVALID when inspected with mapping
    cell_meta_dup_coord = tmp_path / "cell_dup_coord.csv"
    create_csv_matrix(
        cell_meta_dup_coord,
        data=[[1, "C1"], [1, "C2"]],
        headers=["cell_idx", "cell_id"],
    )
    feat_meta_valid = tmp_path / "feat_valid.csv"
    create_csv_matrix(
        feat_meta_valid,
        data=[[1, "ENSG00000124762"], [2, "ENSG00000142168"]],
        headers=["feat_idx", "gene_id"],
    )
    res_mtx_bad_coord = importer.inspect(
        create_source_set(
            [
                create_source_file(mtx_file, ImportFileRole.MATRIX),
                create_source_file(cell_meta_dup_coord, ImportFileRole.CELL_METADATA),
                create_source_file(feat_meta_valid, ImportFileRole.FEATURE_METADATA),
            ]
        ),
        mapping=make_mapping(
            orientation=MatrixOrientation.CELLS_BY_GENES,
            cell_axis_index_column="cell_idx",
            feature_axis_index_column="feat_idx",
            cell_id_column="cell_id",
            gene_id_column="gene_id",
        ),
    )
    assert res_mtx_bad_coord.status == InspectionStatus.INVALID
    assert res_mtx_bad_coord.requiresMapping is False


# =========================================================================
# 8. GOLDEN CSV TESTS: CELLS_BY_GENES & GENES_BY_CELLS
# =========================================================================


def test_golden_csv_cells_by_genes(tmp_path: Path) -> None:
    """Non-square 2 cells x 3 genes matrix (CELLS_BY_GENES) with explicit matrixCellIdColumn."""
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    csv_file = tmp_path / "counts_cells_by_genes.csv"
    create_csv_matrix(
        csv_file,
        data=[
            ["cell_A", 10, 20, 30],
            ["cell_B", 40, 50, 60],
        ],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168", "ENSG00000182463"],
    )

    source_set = create_source_set([create_source_file(csv_file, ImportFileRole.MATRIX)])
    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
    )

    out_h5ad = tmp_path / "out_cells_by_genes.h5ad"
    result = importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )

    assert result.cellCount == 2
    assert result.geneCount == 3
    assert out_h5ad.exists()

    adata = ad.read_h5ad(out_h5ad)
    assert adata.n_obs == 2
    assert adata.n_vars == 3
    assert list(adata.obs_names) == ["cell_A", "cell_B"]
    assert list(adata.var_names) == ["ENSG00000124762", "ENSG00000142168", "ENSG00000182463"]
    assert sp.issparse(adata.X)
    assert adata.X.format == "csr"

    dense_x = adata.X.toarray()
    expected = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    np.testing.assert_array_equal(dense_x, expected)

    validate_canonical_anndata(
        adata,
        expected_organism=Organism.HUMAN,
        expected_gene_id_scheme=GeneIdScheme.ENSEMBL,
    )


def test_golden_csv_genes_by_cells(tmp_path: Path) -> None:
    """Non-square 3 genes x 2 cells matrix (GENES_BY_CELLS) with explicit matrixFeatureIdColumn."""
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    csv_file = tmp_path / "counts_genes_by_cells.csv"
    create_csv_matrix(
        csv_file,
        data=[
            ["ENSG00000124762", 10, 40],
            ["ENSG00000142168", 20, 50],
            ["ENSG00000182463", 30, 60],
        ],
        headers=["gene_id", "cell_A", "cell_B"],
    )

    source_set = create_source_set([create_source_file(csv_file, ImportFileRole.MATRIX)])
    mapping = make_mapping(
        orientation=MatrixOrientation.GENES_BY_CELLS,
        matrix_feature_id_column="gene_id",
    )

    out_h5ad = tmp_path / "out_genes_by_cells.h5ad"
    result = importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )

    assert result.cellCount == 2
    assert result.geneCount == 3
    assert out_h5ad.exists()

    adata = ad.read_h5ad(out_h5ad)
    assert adata.n_obs == 2
    assert adata.n_vars == 3
    assert list(adata.obs_names) == ["cell_A", "cell_B"]
    assert list(adata.var_names) == ["ENSG00000124762", "ENSG00000142168", "ENSG00000182463"]
    assert sp.issparse(adata.X)
    assert adata.X.format == "csr"

    dense_x = adata.X.toarray()
    expected = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    np.testing.assert_array_equal(dense_x, expected)

    validate_canonical_anndata(
        adata,
        expected_organism=Organism.HUMAN,
        expected_gene_id_scheme=GeneIdScheme.ENSEMBL,
    )


# =========================================================================
# 9. GOLDEN TSV TESTS (PLAIN & GZIP)
# =========================================================================


def test_golden_tsv_plain_and_gzip(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    tsv_gz = tmp_path / "counts.tsv.gz"
    create_csv_matrix(
        tsv_gz,
        data=[
            ["cell_1", 100, 200],
            ["cell_2", 300, 400],
        ],
        headers=["cell_id", "ENSMUSG00000024046", "ENSMUSG00000019929"],
        sep="\t",
        compress=True,
    )

    source_set = create_source_set([create_source_file(tsv_gz, ImportFileRole.MATRIX)])
    mapping = make_mapping(
        organism=Organism.MOUSE,
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
    )

    out_h5ad = tmp_path / "out_tsv.h5ad"
    result = importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.MOUSE,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )

    assert result.cellCount == 2
    assert result.geneCount == 2
    adata = ad.read_h5ad(out_h5ad)
    assert list(adata.obs_names) == ["cell_1", "cell_2"]
    assert list(adata.var_names) == ["ENSMUSG00000024046", "ENSMUSG00000019929"]
    np.testing.assert_array_equal(adata.X.toarray(), np.array([[100, 200], [300, 400]]))


# =========================================================================
# 10. GOLDEN MTX TESTS & REORDERED MTX METADATA GOLDEN TEST (Section 16)
# =========================================================================


def test_golden_mtx_cells_by_genes(tmp_path: Path) -> None:
    """Matrix Market 2 cells x 3 genes with explicit axis coordinates."""
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    mtx_file = tmp_path / "matrix.mtx"
    create_mtx_file(
        mtx_file,
        n_rows=2,
        n_cols=3,
        entries=[
            (1, 1, 10),
            (1, 3, 30),
            (2, 2, 50),
            (2, 3, 60),
        ],
    )

    cell_meta = tmp_path / "cells.csv"
    create_csv_matrix(
        cell_meta,
        data=[[1, "C1", "sample_A"], [2, "C2", "sample_B"]],
        headers=["mtx_row_idx", "cell_id", "sample_source"],
    )

    feat_meta = tmp_path / "genes.csv"
    create_csv_matrix(
        feat_meta,
        data=[
            [1, "ENSG00000124762", "CDKN1A"],
            [2, "ENSG00000142168", "SOD1"],
            [3, "ENSG00000182463", "ACTB"],
        ],
        headers=["mtx_col_idx", "gene_id", "symbol"],
    )

    source_set = create_source_set(
        [
            create_source_file(mtx_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
            create_source_file(feat_meta, ImportFileRole.FEATURE_METADATA),
        ]
    )

    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        cell_axis_index_column="mtx_row_idx",
        feature_axis_index_column="mtx_col_idx",
        cell_id_column="cell_id",
        gene_id_column="gene_id",
        gene_symbol_column="symbol",
        sample_column="sample_source",
        sample_mappings=[
            SampleMappingEntry(sourceValue="sample_A", sampleKey="SMP-001"),
            SampleMappingEntry(sourceValue="sample_B", sampleKey="SMP-002"),
        ],
    )

    out_h5ad = tmp_path / "out_mtx_cells_by_genes.h5ad"
    result = importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )

    assert result.cellCount == 2
    assert result.geneCount == 3

    adata = ad.read_h5ad(out_h5ad)
    assert list(adata.obs_names) == ["C1", "C2"]
    assert list(adata.var_names) == ["ENSG00000124762", "ENSG00000142168", "ENSG00000182463"]
    assert list(adata.obs["sample_key"]) == ["SMP-001", "SMP-002"]
    assert list(adata.var["gene_symbol"]) == ["CDKN1A", "SOD1", "ACTB"]
    assert "mtx_row_idx" not in adata.obs.columns
    assert "mtx_col_idx" not in adata.var.columns

    dense_x = adata.X.toarray()
    expected = np.array([[10, 0, 30], [0, 50, 60]], dtype=np.int32)
    np.testing.assert_array_equal(dense_x, expected)


def test_golden_mtx_genes_by_cells(tmp_path: Path) -> None:
    """Matrix Market 3 rows (genes) x 2 cols (cells) with transpose and explicit axis coordinates."""
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    mtx_file = tmp_path / "matrix_gc.mtx.gz"
    create_mtx_file(
        mtx_file,
        n_rows=3,
        n_cols=2,
        entries=[
            (1, 1, 10),
            (3, 1, 30),
            (2, 2, 50),
            (3, 2, 60),
        ],
        compress=True,
    )

    cell_meta = tmp_path / "cells.csv"
    create_csv_matrix(
        cell_meta,
        data=[[1, "C1"], [2, "C2"]],
        headers=["mtx_col_idx", "cell_id"],
    )

    feat_meta = tmp_path / "genes.csv"
    create_csv_matrix(
        feat_meta,
        data=[
            [1, "ENSG00000124762"],
            [2, "ENSG00000142168"],
            [3, "ENSG00000182463"],
        ],
        headers=["mtx_row_idx", "gene_id"],
    )

    source_set = create_source_set(
        [
            create_source_file(mtx_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
            create_source_file(feat_meta, ImportFileRole.FEATURE_METADATA),
        ]
    )

    mapping = make_mapping(
        orientation=MatrixOrientation.GENES_BY_CELLS,
        cell_axis_index_column="mtx_col_idx",
        feature_axis_index_column="mtx_row_idx",
        cell_id_column="cell_id",
        gene_id_column="gene_id",
    )

    out_h5ad = tmp_path / "out_mtx_genes_by_cells.h5ad"
    result = importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )

    assert result.cellCount == 2
    assert result.geneCount == 3

    adata = ad.read_h5ad(out_h5ad)
    assert list(adata.obs_names) == ["C1", "C2"]
    assert list(adata.var_names) == ["ENSG00000124762", "ENSG00000142168", "ENSG00000182463"]

    dense_x = adata.X.toarray()
    expected = np.array([[10, 0, 30], [0, 50, 60]], dtype=np.int32)
    np.testing.assert_array_equal(dense_x, expected)


def test_reordered_mtx_metadata_golden(tmp_path: Path) -> None:
    """Section 16 Mandatory Golden Test: Non-square 3 features x 2 cells MTX (GENES_BY_CELLS)
    with deliberately reordered CELL_METADATA and FEATURE_METADATA.

    CELL_METADATA:
      2,CELL_B
      1,CELL_A
    FEATURE_METADATA:
      3,probe_gamma,ENSG00000182463,ACTB
      1,probe_alpha,ENSG00000124762,CDKN1A
      2,probe_beta,ENSG00000142168,SOD1
    """
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    # 3 features (rows 1..3) x 2 cells (cols 1..2)
    # Row 1 (probe_alpha / ENSG00000124762): col 1 (CELL_A)=11, col 2 (CELL_B)=12
    # Row 2 (probe_beta  / ENSG00000142168): col 1 (CELL_A)=21, col 2 (CELL_B)=22
    # Row 3 (probe_gamma / ENSG00000182463): col 1 (CELL_A)=31, col 2 (CELL_B)=32
    mtx_file = tmp_path / "reordered_golden.mtx"
    create_mtx_file(
        mtx_file,
        n_rows=3,
        n_cols=2,
        entries=[
            (1, 1, 11),
            (1, 2, 12),
            (2, 1, 21),
            (2, 2, 22),
            (3, 1, 31),
            (3, 2, 32),
        ],
    )

    cell_meta = tmp_path / "cell_meta_reordered.csv"
    create_csv_matrix(
        cell_meta,
        data=[
            [2, "CELL_B"],
            [1, "CELL_A"],
        ],
        headers=["matrix_cell_index", "cell_id"],
    )

    feat_meta = tmp_path / "feat_meta_reordered.csv"
    create_csv_matrix(
        feat_meta,
        data=[
            [3, "probe_gamma", "ENSG00000182463", "ACTB"],
            [1, "probe_alpha", "ENSG00000124762", "CDKN1A"],
            [2, "probe_beta", "ENSG00000142168", "SOD1"],
        ],
        headers=["matrix_feature_index", "probe_label", "stable_gene_id", "gene_symbol"],
    )

    source_set = create_source_set(
        [
            create_source_file(mtx_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
            create_source_file(feat_meta, ImportFileRole.FEATURE_METADATA),
        ]
    )

    mapping = make_mapping(
        orientation=MatrixOrientation.GENES_BY_CELLS,
        cell_axis_index_column="matrix_cell_index",
        feature_axis_index_column="matrix_feature_index",
        cell_id_column="cell_id",
        feature_metadata_join_column="probe_label",
        gene_id_column="stable_gene_id",
        gene_symbol_column="gene_symbol",
    )

    out_h5ad = tmp_path / "out_reordered_mtx_golden.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )

    adata = ad.read_h5ad(out_h5ad)
    assert list(adata.obs_names) == ["CELL_A", "CELL_B"]
    assert list(adata.var_names) == ["ENSG00000124762", "ENSG00000142168", "ENSG00000182463"]
    assert list(adata.var["source_feature_id"]) == ["probe_alpha", "probe_beta", "probe_gamma"]
    assert list(adata.var["gene_symbol"]) == ["CDKN1A", "SOD1", "ACTB"]
    assert "matrix_cell_index" not in adata.obs.columns
    assert "matrix_feature_index" not in adata.var.columns

    # Canonical shape is 2 cells x 3 features:
    # CELL_A: [11, 21, 31], CELL_B: [12, 22, 32]
    expected_x = np.array([[11, 21, 31], [12, 22, 32]], dtype=np.int32)
    np.testing.assert_array_equal(adata.X.toarray(), expected_x)


def test_mtx_without_metadata_or_axis_columns_fails_closed(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mtx_file = tmp_path / "matrix.mtx"
    create_mtx_file(mtx_file, 2, 2, [(1, 1, 5)])

    source_set = create_source_set([create_source_file(mtx_file, ImportFileRole.MATRIX)])
    mapping = make_mapping(orientation=MatrixOrientation.CELLS_BY_GENES)

    out_h5ad = tmp_path / "out.h5ad"
    with pytest.raises(ImporterSecurityError) as exc:
        importer.canonicalize(
            source_set=source_set,
            output_path=str(out_h5ad),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


# =========================================================================
# 11. MTX AXIS ERROR TESTS (Section 17) & ROW-ORDER INDEPENDENCE (Section 18)
# =========================================================================


@pytest.mark.parametrize(
    "bad_coords",
    [
        ["0", "1"],  # zero
        ["-1", "2"],  # negative
        ["1.5", "2"],  # fractional
        ["1", "1"],  # duplicate
        ["1"],  # missing index (only 1 of 2)
        ["1", "3"],  # gap / out-of-range for n=2
        ["1", "2", "3"],  # extra / out-of-range coordinate
        ["abc", "2"],  # non-numeric
        ["", "2"],  # blank/null
    ],
)
def test_mtx_axis_error_cases_both_cell_and_feature(tmp_path: Path, bad_coords: list[str]) -> None:
    """Section 17: Test zero, negative, fractional, duplicate, missing, gap,
    out-of-range, non-numeric, and blank/null for BOTH cellAxisIndexColumn and featureAxisIndexColumn.
    All must fail closed.
    """
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mtx_file = tmp_path / "mtx_2x2.mtx"
    create_mtx_file(mtx_file, 2, 2, [(1, 1, 10), (2, 2, 20)])

    valid_cells = tmp_path / "valid_cells.csv"
    create_csv_matrix(
        valid_cells,
        data=[["1", "C1"], ["2", "C2"]],
        headers=["cell_idx", "cell_id"],
    )
    valid_feats = tmp_path / "valid_feats.csv"
    create_csv_matrix(
        valid_feats,
        data=[["1", "ENSG00000124762"], ["2", "ENSG00000142168"]],
        headers=["feat_idx", "gene_id"],
    )

    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        cell_axis_index_column="cell_idx",
        feature_axis_index_column="feat_idx",
        cell_id_column="cell_id",
        gene_id_column="gene_id",
    )

    # 1. Bad cellAxisIndexColumn
    bad_cells = tmp_path / "bad_cells.csv"
    bad_cell_rows = [[c, f"C_{i}"] for i, c in enumerate(bad_coords)]
    create_csv_matrix(bad_cells, data=bad_cell_rows, headers=["cell_idx", "cell_id"])
    src_bad_cell = create_source_set(
        [
            create_source_file(mtx_file, ImportFileRole.MATRIX),
            create_source_file(bad_cells, ImportFileRole.CELL_METADATA),
            create_source_file(valid_feats, ImportFileRole.FEATURE_METADATA),
        ]
    )
    with pytest.raises((CountValidationError, ImporterSecurityError)):
        importer.canonicalize(
            source_set=src_bad_cell,
            output_path=str(tmp_path / "fail_cell_axis.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )

    # 2. Bad featureAxisIndexColumn
    bad_feats = tmp_path / "bad_feats.csv"
    ensembl_pool = ["ENSG00000124762", "ENSG00000142168", "ENSG00000182463"]
    bad_feat_rows = [[c, ensembl_pool[i]] for i, c in enumerate(bad_coords)]
    create_csv_matrix(bad_feats, data=bad_feat_rows, headers=["feat_idx", "gene_id"])
    src_bad_feat = create_source_set(
        [
            create_source_file(mtx_file, ImportFileRole.MATRIX),
            create_source_file(valid_cells, ImportFileRole.CELL_METADATA),
            create_source_file(bad_feats, ImportFileRole.FEATURE_METADATA),
        ]
    )
    with pytest.raises((CountValidationError, ImporterSecurityError)):
        importer.canonicalize(
            source_set=src_bad_feat,
            output_path=str(tmp_path / "fail_feat_axis.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )


def test_mtx_row_order_independence(tmp_path: Path) -> None:
    """Section 18: Arbitrarily reorder CELL_METADATA and FEATURE_METADATA rows multiple ways;
    canonical output must remain identical in obs_names, var_names, source_feature_id, and matrix coordinates.
    """
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    mtx_file = tmp_path / "mtx_3x3.mtx"
    create_mtx_file(
        mtx_file,
        n_rows=3,
        n_cols=3,
        entries=[
            (1, 1, 11),
            (1, 2, 12),
            (2, 3, 23),
            (3, 1, 31),
            (3, 3, 33),
        ],
    )

    cell_rows = [
        [1, "CELL_1", "cond_1"],
        [2, "CELL_2", "cond_2"],
        [3, "CELL_3", "cond_3"],
    ]
    feat_rows = [
        [1, "p1", "ENSG00000124762", "CDKN1A"],
        [2, "p2", "ENSG00000142168", "SOD1"],
        [3, "p3", "ENSG00000182463", "ACTB"],
    ]

    permutations = [
        ([0, 1, 2], [0, 1, 2]),
        ([2, 0, 1], [1, 2, 0]),
        ([1, 2, 0], [2, 1, 0]),
    ]

    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        cell_axis_index_column="cell_idx",
        feature_axis_index_column="feat_idx",
        cell_id_column="cell_id",
        feature_metadata_join_column="probe_id",
        gene_id_column="gene_id",
        gene_symbol_column="symbol",
        condition_column="cond",
    )

    baseline_adata: Any = None
    for idx, (c_perm, f_perm) in enumerate(permutations):
        c_file = tmp_path / f"cells_perm_{idx}.csv"
        f_file = tmp_path / f"feats_perm_{idx}.csv"
        create_csv_matrix(
            c_file,
            data=[cell_rows[i] for i in c_perm],
            headers=["cell_idx", "cell_id", "cond"],
        )
        create_csv_matrix(
            f_file,
            data=[feat_rows[i] for i in f_perm],
            headers=["feat_idx", "probe_id", "gene_id", "symbol"],
        )
        src = create_source_set(
            [
                create_source_file(mtx_file, ImportFileRole.MATRIX),
                create_source_file(c_file, ImportFileRole.CELL_METADATA),
                create_source_file(f_file, ImportFileRole.FEATURE_METADATA),
            ]
        )
        out_path = tmp_path / f"out_perm_{idx}.h5ad"
        importer.canonicalize(
            source_set=src,
            output_path=str(out_path),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
        adata = ad.read_h5ad(out_path)
        if baseline_adata is None:
            baseline_adata = adata
        else:
            assert list(adata.obs_names) == list(baseline_adata.obs_names)
            assert list(adata.var_names) == list(baseline_adata.var_names)
            assert list(adata.var["source_feature_id"]) == list(
                baseline_adata.var["source_feature_id"]
            )
            assert list(adata.var["gene_symbol"]) == list(baseline_adata.var["gene_symbol"])
            assert list(adata.obs["condition"]) == list(baseline_adata.obs["condition"])
            np.testing.assert_array_equal(adata.X.toarray(), baseline_adata.X.toarray())


# =========================================================================
# 12. CSV MISLEADING-FIRST-COLUMN (Section 19) & GENES_BY_CELLS FEATURE ROW (Section 20)
# =========================================================================


def test_csv_misleading_first_column_regression(tmp_path: Path) -> None:
    """Section 19:
    batch,ENSG00000124762,ENSG00000142168
    A,10,20
    B,30,40

    Without matrixCellIdColumn = 'batch', do NOT treat A/B as cell IDs:
    - inspect(): NEEDS_MAPPING
    - canonicalize(): MISSING_REQUIRED_METADATA
    With explicit matrixCellIdColumn = 'batch':
    - A/B become source cell IDs by explicit user authority.
    """
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    csv_file = tmp_path / "misleading_batch.csv"
    csv_file.write_text(
        "batch,ENSG00000124762,ENSG00000142168\nA,10,20\nB,30,40\n", encoding="utf-8"
    )
    source_set = create_source_set([create_source_file(csv_file, ImportFileRole.MATRIX)])

    insp = importer.inspect(source_set)
    assert insp.status == InspectionStatus.NEEDS_MAPPING
    assert insp.requiresMapping is True

    # Without matrixCellIdColumn -> fails closed with MISSING_REQUIRED_METADATA
    with pytest.raises(ImporterSecurityError) as exc:
        importer.canonicalize(
            source_set=source_set,
            output_path=str(tmp_path / "fail_no_matrix_cell_col.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=make_mapping(orientation=MatrixOrientation.CELLS_BY_GENES),
        )
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA

    # With explicit matrixCellIdColumn = "batch" -> succeeds by explicit user authority
    out_ok = tmp_path / "explicit_batch_as_cell.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_ok),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=make_mapping(
            orientation=MatrixOrientation.CELLS_BY_GENES,
            matrix_cell_id_column="batch",
        ),
    )
    adata = ad.read_h5ad(out_ok)
    assert list(adata.obs_names) == ["A", "B"]
    assert list(adata.var_names) == ["ENSG00000124762", "ENSG00000142168"]


def test_genes_by_cells_explicit_feature_row(tmp_path: Path) -> None:
    """Section 20:
    probe_label,CELL_A,CELL_B
    probe_1,10,20
    probe_2,30,40

    Require matrixFeatureIdColumn = 'probe_label'.
    Without it: NEEDS_MAPPING / MISSING_REQUIRED_METADATA.
    With it + FEATURE_METADATA mapping probe_1/probe_2 to stable IDs: canonical var_names use stable IDs.
    """
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    csv_file = tmp_path / "probes_by_cells.csv"
    csv_file.write_text(
        "probe_label,CELL_A,CELL_B\nprobe_1,10,20\nprobe_2,30,40\n", encoding="utf-8"
    )

    feat_meta = tmp_path / "probe_meta.csv"
    create_csv_matrix(
        feat_meta,
        data=[
            ["probe_1", "ENSG00000124762", "CDKN1A"],
            ["probe_2", "ENSG00000142168", "SOD1"],
        ],
        headers=["probe_key", "ensembl_id", "symbol"],
    )

    source_set = create_source_set(
        [
            create_source_file(csv_file, ImportFileRole.MATRIX),
            create_source_file(feat_meta, ImportFileRole.FEATURE_METADATA),
        ]
    )

    insp = importer.inspect(source_set)
    assert insp.status == InspectionStatus.NEEDS_MAPPING

    # Without matrixFeatureIdColumn -> fails closed
    with pytest.raises(ImporterSecurityError) as exc:
        importer.canonicalize(
            source_set=source_set,
            output_path=str(tmp_path / "fail_no_feat_col.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=make_mapping(
                orientation=MatrixOrientation.GENES_BY_CELLS,
                feature_metadata_join_column="probe_key",
                gene_id_column="ensembl_id",
            ),
        )
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA

    # With explicit matrixFeatureIdColumn = "probe_label" -> succeeds
    out_ok = tmp_path / "probes_resolved.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_ok),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=make_mapping(
            orientation=MatrixOrientation.GENES_BY_CELLS,
            matrix_feature_id_column="probe_label",
            feature_metadata_join_column="probe_key",
            gene_id_column="ensembl_id",
            gene_symbol_column="symbol",
        ),
    )
    adata = ad.read_h5ad(out_ok)
    assert list(adata.obs_names) == ["CELL_A", "CELL_B"]
    assert list(adata.var_names) == ["ENSG00000124762", "ENSG00000142168"]
    assert list(adata.var["source_feature_id"]) == ["probe_1", "probe_2"]
    assert list(adata.var["gene_symbol"]) == ["CDKN1A", "SOD1"]
    np.testing.assert_array_equal(adata.X.toarray(), np.array([[10, 30], [20, 40]]))


# =========================================================================
# 13. FEATURE METADATA JOIN TEST MATRIX (A-E)
# =========================================================================


def test_feature_metadata_join_matrix_cases(tmp_path: Path) -> None:
    """Test full matrix of FEATURE_METADATA joining policies using explicit featureMetadataJoinColumn:

    A. Perfect same-order identity match
    B. Reordered relative to matrix => deterministic realignment
    C. Duplicate metadata join key => reject (AMBIGUOUS_GENE_IDENTIFIERS)
    D. Matrix feature missing from metadata => reject (MISSING_REQUIRED_METADATA)
    E. Extra unrelated metadata rows => ignored, matrix features resolve.
    """
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10, 20]],
        headers=["cell_id", "probe_1", "probe_2"],
    )

    # A & E: Extra rows and same order
    feat_meta_extra = tmp_path / "feat_extra.csv"
    create_csv_matrix(
        feat_meta_extra,
        data=[
            ["probe_1", "ENSG00000124762", "CDKN1A"],
            ["probe_2", "ENSG00000142168", "SOD1"],
            ["probe_unrelated", "ENSG00000999999", "EXTRA"],
        ],
        headers=["probe_id", "ensembl_id", "symbol"],
    )

    src_extra = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(feat_meta_extra, ImportFileRole.FEATURE_METADATA),
        ]
    )
    mapping_extra = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        feature_metadata_join_column="probe_id",
        gene_id_column="ensembl_id",
        gene_symbol_column="symbol",
    )
    out_extra = tmp_path / "out_extra.h5ad"
    importer.canonicalize(
        source_set=src_extra,
        output_path=str(out_extra),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping_extra,
    )
    adata_extra = ad.read_h5ad(out_extra)
    assert list(adata_extra.var_names) == ["ENSG00000124762", "ENSG00000142168"]
    assert list(adata_extra.var["source_feature_id"]) == ["probe_1", "probe_2"]
    assert list(adata_extra.var["gene_symbol"]) == ["CDKN1A", "SOD1"]

    # B: Reordered metadata relative to matrix
    feat_meta_reordered = tmp_path / "feat_reordered.csv"
    create_csv_matrix(
        feat_meta_reordered,
        data=[
            ["probe_2", "ENSG00000142168", "SOD1"],
            ["probe_1", "ENSG00000124762", "CDKN1A"],
        ],
        headers=["probe_id", "ensembl_id", "symbol"],
    )
    src_reord = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(feat_meta_reordered, ImportFileRole.FEATURE_METADATA),
        ]
    )
    out_reord = tmp_path / "out_reord.h5ad"
    importer.canonicalize(
        source_set=src_reord,
        output_path=str(out_reord),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping_extra,
    )
    adata_reord = ad.read_h5ad(out_reord)
    assert list(adata_reord.var_names) == ["ENSG00000124762", "ENSG00000142168"]
    assert list(adata_reord.var["gene_symbol"]) == ["CDKN1A", "SOD1"]

    # C: Duplicate metadata join key
    feat_meta_dup = tmp_path / "feat_dup.csv"
    create_csv_matrix(
        feat_meta_dup,
        data=[
            ["probe_1", "ENSG00000124762", "CDKN1A"],
            ["probe_1", "ENSG00000142168", "SOD1"],
        ],
        headers=["probe_id", "ensembl_id", "symbol"],
    )
    src_dup = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(feat_meta_dup, ImportFileRole.FEATURE_METADATA),
        ]
    )
    with pytest.raises((CountValidationError, ImporterSecurityError)) as exc_dup:
        importer.canonicalize(
            source_set=src_dup,
            output_path=str(tmp_path / "out_fail.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping_extra,
        )
    assert isinstance(exc_dup.value, CountValidationError | ImporterSecurityError)
    assert exc_dup.value.issue_code == ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS

    # D: Matrix feature missing from metadata
    feat_meta_missing = tmp_path / "feat_missing.csv"
    create_csv_matrix(
        feat_meta_missing,
        data=[
            ["probe_1", "ENSG00000124762", "CDKN1A"],
        ],
        headers=["probe_id", "ensembl_id", "symbol"],
    )
    src_miss = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(feat_meta_missing, ImportFileRole.FEATURE_METADATA),
        ]
    )
    with pytest.raises(ImporterSecurityError) as exc_miss:
        importer.canonicalize(
            source_set=src_miss,
            output_path=str(tmp_path / "out_fail.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping_extra,
        )
    assert exc_miss.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


# =========================================================================
# 14. CELL METADATA JOIN TEST MATRIX (A-E)
# =========================================================================


def test_cell_metadata_join_matrix_cases(tmp_path: Path) -> None:
    """Test full matrix of CELL_METADATA joining policies using explicit cellMetadataJoinColumn:

    A. Perfect same-order identity match
    B. Reordered relative to matrix => deterministic realignment
    C. Duplicate cell key => reject (DUPLICATE_CELL_IDS)
    D. Missing matrix cell => reject (MISSING_REQUIRED_METADATA)
    E. Extra unrelated metadata row => extra row ignored, matrix cells resolve
    F. Translated canonical cellIdColumn distinct from cellMetadataJoinColumn.
    """
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[
            ["cell_A", 1, 2],
            ["cell_B", 3, 4],
        ],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )

    # A, E & F: Extra rows, same order, and translated canonical cellIdColumn
    cell_meta_extra = tmp_path / "cells_extra.csv"
    create_csv_matrix(
        cell_meta_extra,
        data=[
            ["cell_A", "CANON_CELL_A", "cond_A"],
            ["cell_B", "CANON_CELL_B", "cond_B"],
            ["cell_unrelated", "CANON_CELL_X", "cond_X"],
        ],
        headers=["cell_id", "canonical_id", "cond"],
    )
    src_extra = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta_extra, ImportFileRole.CELL_METADATA),
        ]
    )
    mapping_extra = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        cell_metadata_join_column="cell_id",
        cell_id_column="canonical_id",
        condition_column="cond",
    )
    out_extra = tmp_path / "out_cell_extra.h5ad"
    importer.canonicalize(
        source_set=src_extra,
        output_path=str(out_extra),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping_extra,
    )
    adata_extra = ad.read_h5ad(out_extra)
    assert list(adata_extra.obs_names) == ["CANON_CELL_A", "CANON_CELL_B"]
    assert list(adata_extra.obs["condition"]) == ["cond_A", "cond_B"]

    # B: Reordered metadata relative to matrix (with omitted cellIdColumn -> keeps source tokens)
    cell_meta_reordered = tmp_path / "cells_reordered.csv"
    create_csv_matrix(
        cell_meta_reordered,
        data=[
            ["cell_B", "cond_B"],
            ["cell_A", "cond_A"],
        ],
        headers=["cell_id", "cond"],
    )
    src_reord = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta_reordered, ImportFileRole.CELL_METADATA),
        ]
    )
    mapping_reord = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        cell_metadata_join_column="cell_id",
        condition_column="cond",
    )
    out_reord = tmp_path / "out_cell_reord.h5ad"
    importer.canonicalize(
        source_set=src_reord,
        output_path=str(out_reord),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping_reord,
    )
    adata_reord = ad.read_h5ad(out_reord)
    assert list(adata_reord.obs_names) == ["cell_A", "cell_B"]
    assert list(adata_reord.obs["condition"]) == ["cond_A", "cond_B"]

    # C: Duplicate cell key
    cell_meta_dup = tmp_path / "cells_dup.csv"
    create_csv_matrix(
        cell_meta_dup,
        data=[
            ["cell_A", "cond_1"],
            ["cell_A", "cond_2"],
        ],
        headers=["cell_id", "cond"],
    )
    src_dup = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta_dup, ImportFileRole.CELL_METADATA),
        ]
    )
    with pytest.raises((CountValidationError, ImporterSecurityError)) as exc_dup:
        importer.canonicalize(
            source_set=src_dup,
            output_path=str(tmp_path / "out_fail.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping_reord,
        )
    assert isinstance(exc_dup.value, CountValidationError | ImporterSecurityError)
    assert exc_dup.value.issue_code == ScientificIssueCode.DUPLICATE_CELL_IDS

    # D: Missing matrix cell
    cell_meta_missing = tmp_path / "cells_missing.csv"
    create_csv_matrix(
        cell_meta_missing,
        data=[
            ["cell_A", "cond_A"],
        ],
        headers=["cell_id", "cond"],
    )
    src_miss = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta_missing, ImportFileRole.CELL_METADATA),
        ]
    )
    with pytest.raises(ImporterSecurityError) as exc_miss:
        importer.canonicalize(
            source_set=src_miss,
            output_path=str(tmp_path / "out_fail.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping_reord,
        )
    assert exc_miss.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


# =========================================================================
# 15. STRICT SAMPLE MAPPING AUTHORITY
# =========================================================================


def test_sample_mapping_strict_authority_and_no_passthrough(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10, 20], ["c2", 30, 40]],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )

    cell_meta = tmp_path / "cells.csv"
    create_csv_matrix(
        cell_meta,
        data=[["c1", "source_sample_X"], ["c2", "source_sample_Y"]],
        headers=["cell_id", "sample_col"],
    )

    source_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
        ]
    )

    # 1. sampleColumn provided without sampleMappings: FAIL CLOSED
    mapping_no_entries = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        cell_metadata_join_column="cell_id",
        sample_column="sample_col",
    )
    out_h5ad = tmp_path / "out_sample.h5ad"
    with pytest.raises(ImporterSecurityError) as exc:
        importer.canonicalize(
            source_set=source_set,
            output_path=str(out_h5ad),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping_no_entries,
        )
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA

    # 2. Incomplete sample mapping: c2's sample has no entry: FAIL CLOSED
    mapping_partial = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        cell_metadata_join_column="cell_id",
        sample_column="sample_col",
        sample_mappings=[
            SampleMappingEntry(sourceValue="source_sample_X", sampleKey="SMP-001"),
        ],
    )
    with pytest.raises(ImporterSecurityError) as exc:
        importer.canonicalize(
            source_set=source_set,
            output_path=str(out_h5ad),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping_partial,
        )
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA

    # 3. Complete approved sample mapping: SUCCESS
    mapping_complete = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        cell_metadata_join_column="cell_id",
        sample_column="sample_col",
        sample_mappings=[
            SampleMappingEntry(sourceValue="source_sample_X", sampleKey="SMP-001"),
            SampleMappingEntry(sourceValue="source_sample_Y", sampleKey="SMP-002"),
        ],
    )
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping_complete,
    )
    adata = ad.read_h5ad(out_h5ad)
    assert list(adata.obs["sample_key"]) == ["SMP-001", "SMP-002"]


def test_sample_mappings_without_sample_column_fails_closed(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10, 20]],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )
    cell_meta = tmp_path / "cells.csv"
    create_csv_matrix(
        cell_meta,
        data=[["c1", "s1"]],
        headers=["cell_id", "s_col"],
    )
    source_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
        ]
    )

    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        cell_metadata_join_column="cell_id",
        sample_column=None,
        sample_mappings=[SampleMappingEntry(sourceValue="s1", sampleKey="SMP-1")],
    )
    out_h5ad = tmp_path / "out_sample.h5ad"
    with pytest.raises(ImporterSecurityError) as exc:
        importer.canonicalize(
            source_set=source_set,
            output_path=str(out_h5ad),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


# =========================================================================
# 16. BIOLOGICAL ANNOTATIONS & CANONICAL OBS SCHEMA TESTS
# =========================================================================


def test_canonical_obs_schema_complete_and_minimal(tmp_path: Path) -> None:
    """Proves exact obs columns when all mappings are supplied vs minimal mapping."""
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10, 20]],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )

    cell_meta = tmp_path / "cells.csv"
    create_csv_matrix(
        cell_meta,
        data=[["c1", "s1", "treated", "batch_1", "bio_1", "tech_1", "unmapped_extra"]],
        headers=[
            "cell_id",
            "sample_col",
            "cond_col",
            "batch_col",
            "bio_col",
            "tech_col",
            "extra_col",
        ],
    )

    source_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
        ]
    )

    # 1. Complete mapping: assert exact columns
    mapping_complete = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        cell_metadata_join_column="cell_id",
        sample_column="sample_col",
        sample_mappings=[SampleMappingEntry(sourceValue="s1", sampleKey="SMP-001")],
        condition_column="cond_col",
        batch_column="batch_col",
        biological_replicate_column="bio_col",
        technical_replicate_column="tech_col",
    )
    out_complete = tmp_path / "out_complete.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_complete),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping_complete,
    )
    adata_complete = ad.read_h5ad(out_complete)
    expected_obs_cols = [
        "sample_key",
        "condition",
        "batch",
        "biological_replicate",
        "technical_replicate",
    ]
    assert list(adata_complete.obs.columns) == expected_obs_cols
    assert "extra_col" not in adata_complete.obs.columns

    # 2. Minimal mapping: zero optional columns
    mapping_minimal = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
    )
    out_minimal = tmp_path / "out_minimal.h5ad"
    importer.canonicalize(
        source_set=create_source_set([create_source_file(mat_file, ImportFileRole.MATRIX)]),
        output_path=str(out_minimal),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping_minimal,
    )
    adata_minimal = ad.read_h5ad(out_minimal)
    assert list(adata_minimal.obs.columns) == []


# =========================================================================
# 17. CANONICAL VAR SCHEMA & STABLE ID + DUPLICATE SYMBOLS TEST
# =========================================================================


def test_stable_id_with_duplicate_symbols_valid(tmp_path: Path) -> None:
    """Proves duplicate symbols are valid when stable canonical IDs are unique."""
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10, 20]],
        headers=["cell_id", "feature_a", "feature_b"],
    )

    feat_meta = tmp_path / "genes.csv"
    create_csv_matrix(
        feat_meta,
        data=[
            ["feature_a", "ENSG00000124762", "DUP"],
            ["feature_b", "ENSG00000142168", "DUP"],
        ],
        headers=["src_id", "stable_id", "symbol"],
    )

    source_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(feat_meta, ImportFileRole.FEATURE_METADATA),
        ]
    )

    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        feature_metadata_join_column="src_id",
        gene_id_column="stable_id",
        gene_symbol_column="symbol",
    )

    out_h5ad = tmp_path / "out_dup_symbol.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )

    adata = ad.read_h5ad(out_h5ad)
    assert list(adata.var_names) == ["ENSG00000124762", "ENSG00000142168"]
    assert list(adata.var["source_feature_id"]) == ["feature_a", "feature_b"]
    assert list(adata.var["gene_symbol"]) == ["DUP", "DUP"]
    assert list(adata.var.columns) == ["source_feature_id", "gene_symbol"]


def test_canonical_var_schema_controlled(tmp_path: Path) -> None:
    """Without geneSymbolColumn, canonical var must contain ONLY source_feature_id."""
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10, 20]],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )
    source_set = create_source_set([create_source_file(mat_file, ImportFileRole.MATRIX)])
    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
    )
    out_h5ad = tmp_path / "out_no_symbol.h5ad"

    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )
    adata = ad.read_h5ad(out_h5ad)
    assert list(adata.var.columns) == ["source_feature_id"]
    assert "gene_symbol" not in adata.var.columns


def test_symbol_only_scheme_and_duplicate_symbols_fails_closed(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    # 1. Unique symbols under GeneIdScheme.SYMBOL: SUCCESS
    mat_file = tmp_path / "mat_sym.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10, 20]],
        headers=["cell_id", "CDKN1A", "ACTB"],
    )
    source_set = create_source_set([create_source_file(mat_file, ImportFileRole.MATRIX)])
    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        gene_id_scheme=GeneIdScheme.SYMBOL,
    )
    out_h5ad = tmp_path / "out_sym.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.SYMBOL,
        mapping=mapping,
    )
    adata = ad.read_h5ad(out_h5ad)
    assert list(adata.var_names) == ["CDKN1A", "ACTB"]

    # 2. Duplicate symbols under GeneIdScheme.SYMBOL: FAILS CLOSED without make_unique()
    mat_dup_sym = tmp_path / "mat_dup_sym.csv"
    create_csv_matrix(
        mat_dup_sym,
        data=[["c1", 10, 20]],
        headers=["cell_id", "CDKN1A", "CDKN1A"],
    )
    with pytest.raises((CountValidationError, ImporterSecurityError)) as exc:
        importer.canonicalize(
            source_set=create_source_set([create_source_file(mat_dup_sym, ImportFileRole.MATRIX)]),
            output_path=str(tmp_path / "out_fail.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.SYMBOL,
            mapping=mapping,
        )
    assert isinstance(exc.value, CountValidationError | ImporterSecurityError)
    assert exc.value.issue_code == ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS


# =========================================================================
# 18. ORGANISM, GENE SCHEME & REFERENCE BUILD RESOLUTION
# =========================================================================


def test_organism_resolution_policy_step_d(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    mat_h = tmp_path / "human.csv"
    create_csv_matrix(
        mat_h, data=[["c1", 10, 20]], headers=["cell_id", "ENSG00000124762", "ENSG00000142168"]
    )
    res_h = importer.inspect(create_source_set([create_source_file(mat_h, ImportFileRole.MATRIX)]))
    assert res_h.detectedOrganism == Organism.HUMAN
    assert res_h.detectedGeneIdScheme == GeneIdScheme.ENSEMBL

    mat_m = tmp_path / "mouse.csv"
    create_csv_matrix(
        mat_m,
        data=[["c1", 10, 20]],
        headers=["cell_id", "ENSMUSG00000024046", "ENSMUSG00000019929"],
    )
    res_m = importer.inspect(create_source_set([create_source_file(mat_m, ImportFileRole.MATRIX)]))
    assert res_m.detectedOrganism == Organism.MOUSE
    assert res_m.detectedGeneIdScheme == GeneIdScheme.ENSEMBL

    mat_mix = tmp_path / "mix.csv"
    create_csv_matrix(
        mat_mix, data=[["c1", 10, 20]], headers=["cell_id", "ENSG00000124762", "ENSMUSG00000024046"]
    )
    res_mix = importer.inspect(
        create_source_set([create_source_file(mat_mix, ImportFileRole.MATRIX)])
    )
    assert res_mix.status == InspectionStatus.INVALID
    assert any(iss.code == ScientificIssueCode.UNRESOLVED_SPECIES for iss in res_mix.issues)

    mat_sym = tmp_path / "symbols.csv"
    create_csv_matrix(mat_sym, data=[["c1", 10, 20]], headers=["cell_id", "CDKN1A", "ACTB"])
    res_sym = importer.inspect(
        create_source_set([create_source_file(mat_sym, ImportFileRole.MATRIX)])
    )
    assert res_sym.detectedOrganism is None
    assert res_sym.detectedGeneIdScheme is None


def test_reference_build_preservation(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file, data=[["c1", 10, 20]], headers=["cell_id", "ENSG00000124762", "ENSG00000142168"]
    )
    source_set = create_source_set([create_source_file(mat_file, ImportFileRole.MATRIX)])

    mapping_with_ref = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        reference_build="GRCh38.p14",
    )
    out_ref = tmp_path / "out_ref.h5ad"
    res1 = importer.canonicalize(
        source_set=source_set,
        output_path=str(out_ref),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping_with_ref,
    )
    assert res1.referenceBuild == "GRCh38.p14"
    adata_ref = ad.read_h5ad(out_ref)
    assert adata_ref.uns["reference_build"] == "GRCh38.p14"

    mapping_no_ref = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
    )
    out_noref = tmp_path / "out_noref.h5ad"
    res2 = importer.canonicalize(
        source_set=source_set,
        output_path=str(out_noref),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping_no_ref,
    )
    assert res2.referenceBuild is None
    adata_noref = ad.read_h5ad(out_noref)
    assert "reference_build" not in adata_noref.uns


# =========================================================================
# 19. RAW COUNT STRICTNESS & POLICY B SLOT HYGIENE
# =========================================================================


def test_integral_floats_accepted(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10.0, 20.0]],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )
    source_set = create_source_set([create_source_file(mat_file, ImportFileRole.MATRIX)])
    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
    )
    out_h5ad = tmp_path / "out_float.h5ad"

    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )
    adata = ad.read_h5ad(out_h5ad)
    assert np.issubdtype(adata.X.dtype, np.integer)
    np.testing.assert_array_equal(adata.X.toarray(), np.array([[10, 20]]))


def test_slot_hygiene_and_uns_provenance(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10, 20]],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )
    source_set = create_source_set([create_source_file(mat_file, ImportFileRole.MATRIX)])
    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
    )
    out_h5ad = tmp_path / "out_hygiene.h5ad"

    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )

    adata = ad.read_h5ad(out_h5ad)
    assert sp.issparse(adata.X)
    assert adata.X.format == "csr"
    assert len(adata.layers) == 0
    assert len(adata.obsm) == 0
    assert len(adata.varm) == 0
    assert len(adata.obsp) == 0
    assert len(adata.varp) == 0

    assert set(adata.uns.keys()) == {"organism", "gene_id_scheme"}
    assert adata.uns["organism"] == Organism.HUMAN.value
    assert adata.uns["gene_id_scheme"] == GeneIdScheme.ENSEMBL.value


# =========================================================================
# 20. SOURCE IMMUTABILITY, PATH COLLISION SAFETY & EXACT ROUND-TRIP
# =========================================================================


def test_output_path_collision_fails_closed(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10, 20]],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )
    source_set = create_source_set([create_source_file(mat_file, ImportFileRole.MATRIX)])
    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
    )

    with pytest.raises(ImporterSecurityError) as exc:
        importer.canonicalize(
            source_set=source_set,
            output_path=str(mat_file),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
    assert exc.value.issue_code == ScientificIssueCode.CANONICALIZATION_FAILED


def test_source_files_remain_immutable(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10, 20]],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )
    cell_meta = tmp_path / "cells.csv"
    create_csv_matrix(cell_meta, data=[["c1", "s1"]], headers=["cell_id", "sample_col"])

    source_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
        ]
    )

    sha_mat_before = compute_file_sha256(mat_file)
    sha_cell_before = compute_file_sha256(cell_meta)

    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        cell_metadata_join_column="cell_id",
        sample_column="sample_col",
        sample_mappings=[SampleMappingEntry(sourceValue="s1", sampleKey="SMP-1")],
    )
    out_h5ad = tmp_path / "out.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )

    sha_mat_after = compute_file_sha256(mat_file)
    sha_cell_after = compute_file_sha256(cell_meta)

    assert sha_mat_before == sha_mat_after
    assert sha_cell_before == sha_cell_after


def test_exact_round_trip_csv_and_mtx(tmp_path: Path) -> None:
    """Exact round-trip verification: write canonical.h5ad, read back, assert exact slots."""
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    # 1. CSV Round-Trip
    csv_file = tmp_path / "round_csv.csv"
    create_csv_matrix(
        csv_file,
        data=[
            ["cell_1", 10, 20],
            ["cell_2", 30, 40],
        ],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )
    cell_meta_csv = tmp_path / "cell_meta_csv.csv"
    create_csv_matrix(
        cell_meta_csv,
        data=[["cell_1", "treated"], ["cell_2", "control"]],
        headers=["cell_id", "condition_source"],
    )

    csv_set = create_source_set(
        [
            create_source_file(csv_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta_csv, ImportFileRole.CELL_METADATA),
        ]
    )
    csv_mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="cell_id",
        cell_metadata_join_column="cell_id",
        condition_column="condition_source",
        reference_build="GRCh38",
    )
    csv_out = tmp_path / "canonical_csv.h5ad"
    importer.canonicalize(
        source_set=csv_set,
        output_path=str(csv_out),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=csv_mapping,
    )
    adata_csv = ad.read_h5ad(csv_out)
    assert list(adata_csv.obs_names) == ["cell_1", "cell_2"]
    assert list(adata_csv.obs.columns) == ["condition"]
    assert list(adata_csv.obs["condition"]) == ["treated", "control"]
    assert list(adata_csv.var_names) == ["ENSG00000124762", "ENSG00000142168"]
    assert list(adata_csv.var["source_feature_id"]) == ["ENSG00000124762", "ENSG00000142168"]
    assert "gene_symbol" not in adata_csv.var.columns
    assert adata_csv.uns["organism"] == "HUMAN"
    assert adata_csv.uns["gene_id_scheme"] == "ENSEMBL"
    assert adata_csv.uns["reference_build"] == "GRCh38"
    assert sp.isspmatrix_csr(adata_csv.X)
    np.testing.assert_array_equal(adata_csv.X.toarray(), np.array([[10, 20], [30, 40]]))
    validate_canonical_anndata(
        adata_csv, expected_organism=Organism.HUMAN, expected_gene_id_scheme=GeneIdScheme.ENSEMBL
    )

    # 2. MTX Round-Trip
    mtx_file = tmp_path / "round_mtx.mtx"
    create_mtx_file(mtx_file, 2, 2, [(1, 1, 5), (2, 2, 15)])
    cell_meta_mtx = tmp_path / "cell_meta_mtx.csv"
    create_csv_matrix(
        cell_meta_mtx,
        data=[[1, "C_A"], [2, "C_B"]],
        headers=["mtx_row", "cell_id"],
    )
    feat_meta_mtx = tmp_path / "feat_meta_mtx.csv"
    create_csv_matrix(
        feat_meta_mtx,
        data=[[1, "ENSG00000124762", "CDKN1A"], [2, "ENSG00000142168", "SOD1"]],
        headers=["mtx_col", "gene_id", "symbol"],
    )

    mtx_set = create_source_set(
        [
            create_source_file(mtx_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta_mtx, ImportFileRole.CELL_METADATA),
            create_source_file(feat_meta_mtx, ImportFileRole.FEATURE_METADATA),
        ]
    )
    mtx_mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        cell_axis_index_column="mtx_row",
        feature_axis_index_column="mtx_col",
        cell_id_column="cell_id",
        gene_id_column="gene_id",
        gene_symbol_column="symbol",
    )
    mtx_out = tmp_path / "canonical_mtx.h5ad"
    importer.canonicalize(
        source_set=mtx_set,
        output_path=str(mtx_out),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mtx_mapping,
    )
    adata_mtx = ad.read_h5ad(mtx_out)
    assert list(adata_mtx.obs_names) == ["C_A", "C_B"]
    assert list(adata_mtx.obs.columns) == []
    assert list(adata_mtx.var_names) == ["ENSG00000124762", "ENSG00000142168"]
    assert list(adata_mtx.var["source_feature_id"]) == ["ENSG00000124762", "ENSG00000142168"]
    assert list(adata_mtx.var["gene_symbol"]) == ["CDKN1A", "SOD1"]
    assert adata_mtx.uns["organism"] == "HUMAN"
    assert adata_mtx.uns["gene_id_scheme"] == "ENSEMBL"
    assert sp.isspmatrix_csr(adata_mtx.X)
    np.testing.assert_array_equal(adata_mtx.X.toarray(), np.array([[5, 0], [0, 15]]))
    validate_canonical_anndata(
        adata_mtx, expected_organism=Organism.HUMAN, expected_gene_id_scheme=GeneIdScheme.ENSEMBL
    )


# =========================================================================
# 21. RANGEINDEX, HEADER AUTHORITY & MTX IDENTITY PROVENANCE TESTS
# =========================================================================


def test_rangeindex_for_cells_by_genes_forbidden(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    csv_file = tmp_path / "no_row_id_cells_by_genes.csv"
    csv_file.write_text("ENSG00000124762,ENSG00000142168\n10,20\n30,40\n", encoding="utf-8")

    df, _ = read_delimited_table(csv_file, DEFAULT_TEST_LIMITS)
    assert isinstance(df.index, pd.RangeIndex)

    source_set = create_source_set([create_source_file(csv_file, ImportFileRole.MATRIX)])

    insp = importer.inspect(source_set)
    assert insp.status == InspectionStatus.NEEDS_MAPPING
    assert insp.requiresMapping is True

    mapping = make_mapping(orientation=MatrixOrientation.CELLS_BY_GENES)
    out_h5ad = tmp_path / "fail_range_cbg.h5ad"
    with pytest.raises(ImporterSecurityError) as exc:
        importer.canonicalize(
            source_set=source_set,
            output_path=str(out_h5ad),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA
    assert not out_h5ad.exists()


def test_rangeindex_for_genes_by_cells_forbidden(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    tsv_file = tmp_path / "no_row_id_genes_by_cells.tsv"
    tsv_file.write_text("cell_A\tcell_B\n10\t20\n30\t40\n50\t60\n", encoding="utf-8")

    df, _ = read_delimited_table(tsv_file, DEFAULT_TEST_LIMITS)
    assert isinstance(df.index, pd.RangeIndex)

    source_set = create_source_set([create_source_file(tsv_file, ImportFileRole.MATRIX)])

    insp = importer.inspect(source_set)
    assert insp.status == InspectionStatus.NEEDS_MAPPING
    assert insp.requiresMapping is True

    mapping = make_mapping(orientation=MatrixOrientation.GENES_BY_CELLS)
    out_h5ad = tmp_path / "fail_range_gbc.h5ad"
    with pytest.raises(ImporterSecurityError) as exc:
        importer.canonicalize(
            source_set=source_set,
            output_path=str(out_h5ad),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA
    assert not out_h5ad.exists()


def test_header_authority_regressions(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    blank_col = tmp_path / "blank_col.csv"
    blank_col.write_text("cell_id,,ENSG00000142168\nc1,10,20\nc2,30,40\n", encoding="utf-8")
    src_blank = create_source_set([create_source_file(blank_col, ImportFileRole.MATRIX)])
    issues_blank = importer.validate(src_blank)
    assert any(iss.code == ScientificIssueCode.MALFORMED_INPUT for iss in issues_blank)
    assert importer.inspect(src_blank).status == InspectionStatus.INVALID
    with pytest.raises(ImporterSecurityError) as exc_blank:
        importer.canonicalize(
            source_set=src_blank,
            output_path=str(tmp_path / "out_blank.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=make_mapping(
                orientation=MatrixOrientation.CELLS_BY_GENES,
                matrix_cell_id_column="cell_id",
            ),
        )
    assert exc_blank.value.issue_code == ScientificIssueCode.MALFORMED_INPUT

    dup_hdr = tmp_path / "dup_hdr.csv"
    dup_hdr.write_text(
        "cell_id,ENSG00000124762,ENSG00000124762\nc1,10,20\nc2,30,40\n", encoding="utf-8"
    )
    src_dup = create_source_set([create_source_file(dup_hdr, ImportFileRole.MATRIX)])
    issues_dup = importer.validate(src_dup)
    assert any(iss.code == ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS for iss in issues_dup)
    assert importer.inspect(src_dup).status == InspectionStatus.INVALID
    with pytest.raises(ImporterSecurityError) as exc_dup:
        importer.canonicalize(
            source_set=src_dup,
            output_path=str(tmp_path / "out_dup.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=make_mapping(
                orientation=MatrixOrientation.CELLS_BY_GENES,
                matrix_cell_id_column="cell_id",
            ),
        )
    assert exc_dup.value.issue_code == ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS

    unnamed_col = tmp_path / "unnamed_col.csv"
    unnamed_col.write_text(
        "cell_id,Unnamed: 1,ENSG00000142168\nc1,10,20\nc2,30,40\n", encoding="utf-8"
    )
    src_unnamed = create_source_set([create_source_file(unnamed_col, ImportFileRole.MATRIX)])
    issues_unnamed = importer.validate(src_unnamed)
    assert any(iss.code == ScientificIssueCode.MALFORMED_INPUT for iss in issues_unnamed)
    assert importer.inspect(src_unnamed).status == InspectionStatus.INVALID

    leading_comma = tmp_path / "leading_comma.csv"
    leading_comma.write_text(
        ",ENSG00000124762,ENSG00000142168\nc1,10,20\nc2,30,40\n", encoding="utf-8"
    )
    src_leading = create_source_set([create_source_file(leading_comma, ImportFileRole.MATRIX)])
    assert importer.inspect(src_leading).status == InspectionStatus.INVALID
    with pytest.raises(ImporterSecurityError) as exc_leading:
        importer.canonicalize(
            source_set=src_leading,
            output_path=str(tmp_path / "out_leading.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=make_mapping(orientation=MatrixOrientation.CELLS_BY_GENES),
        )
    assert exc_leading.value.issue_code == ScientificIssueCode.MALFORMED_INPUT


def test_generic_mtx_without_identity_metadata_inspect_and_canonicalize(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mtx_file = tmp_path / "bare_matrix.mtx"
    create_mtx_file(mtx_file, 3, 2, [(1, 1, 10), (2, 2, 20), (3, 1, 30)])

    source_set = create_source_set([create_source_file(mtx_file, ImportFileRole.MATRIX)])

    insp = importer.inspect(source_set)
    assert insp.status == InspectionStatus.NEEDS_MAPPING
    assert insp.requiresMapping is True

    for orient in (MatrixOrientation.CELLS_BY_GENES, MatrixOrientation.GENES_BY_CELLS):
        out_h5ad = tmp_path / f"out_{orient.value}.h5ad"
        with pytest.raises(ImporterSecurityError) as exc:
            importer.canonicalize(
                source_set=source_set,
                output_path=str(out_h5ad),
                organism=Organism.HUMAN,
                gene_id_scheme=GeneIdScheme.ENSEMBL,
                mapping=make_mapping(orientation=orient),
            )
        assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA
        assert not out_h5ad.exists()


def test_generic_mtx_with_source_feature_ids_and_stable_gene_ids(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    cell_meta = tmp_path / "cells_custom.csv"
    create_csv_matrix(
        cell_meta,
        data=[
            [1, "donor1_well_A01"],
            [2, "donor1_well_B04"],
        ],
        headers=["cell_axis_idx", "custom_cell_token"],
    )

    feat_meta = tmp_path / "features_custom.csv"
    create_csv_matrix(
        feat_meta,
        data=[
            [1, "probe_alpha", "ENSG00000124762", "CDKN1A"],
            [2, "probe_beta", "ENSG00000142168", "SOD1"],
            [3, "probe_gamma", "ENSG00000182463", "ACTB"],
        ],
        headers=["feat_axis_idx", "matrix_probe_id", "ensembl_stable_id", "hgnc_symbol"],
    )

    # Case 1: 2 x 3 MTX (CELLS_BY_GENES — no transpose)
    mtx_cbg = tmp_path / "mtx_2x3.mtx"
    create_mtx_file(
        mtx_cbg,
        n_rows=2,
        n_cols=3,
        entries=[
            (1, 1, 11),
            (1, 3, 13),
            (2, 2, 22),
            (2, 3, 23),
        ],
    )
    src_cbg = create_source_set(
        [
            create_source_file(mtx_cbg, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
            create_source_file(feat_meta, ImportFileRole.FEATURE_METADATA),
        ]
    )
    map_cbg = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        cell_axis_index_column="cell_axis_idx",
        feature_axis_index_column="feat_axis_idx",
        cell_id_column="custom_cell_token",
        feature_metadata_join_column="matrix_probe_id",
        gene_id_column="ensembl_stable_id",
        gene_symbol_column="hgnc_symbol",
    )
    out_cbg = tmp_path / "out_cbg.h5ad"
    importer.canonicalize(
        source_set=src_cbg,
        output_path=str(out_cbg),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=map_cbg,
    )
    adata_cbg = ad.read_h5ad(out_cbg)
    assert list(adata_cbg.obs_names) == ["donor1_well_A01", "donor1_well_B04"]
    assert list(adata_cbg.var_names) == ["ENSG00000124762", "ENSG00000142168", "ENSG00000182463"]
    assert list(adata_cbg.var["source_feature_id"]) == ["probe_alpha", "probe_beta", "probe_gamma"]
    assert list(adata_cbg.var["gene_symbol"]) == ["CDKN1A", "SOD1", "ACTB"]
    expected_mat = np.array([[11, 0, 13], [0, 22, 23]], dtype=np.int32)
    np.testing.assert_array_equal(adata_cbg.X.toarray(), expected_mat)

    # Case 2: 3 x 2 MTX (GENES_BY_CELLS — explicit transpose to 2 x 3)
    mtx_gbc = tmp_path / "mtx_3x2.mtx"
    create_mtx_file(
        mtx_gbc,
        n_rows=3,
        n_cols=2,
        entries=[
            (1, 1, 11),
            (3, 1, 13),
            (2, 2, 22),
            (3, 2, 23),
        ],
    )
    src_gbc = create_source_set(
        [
            create_source_file(mtx_gbc, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
            create_source_file(feat_meta, ImportFileRole.FEATURE_METADATA),
        ]
    )
    map_gbc = make_mapping(
        orientation=MatrixOrientation.GENES_BY_CELLS,
        cell_axis_index_column="cell_axis_idx",
        feature_axis_index_column="feat_axis_idx",
        cell_id_column="custom_cell_token",
        feature_metadata_join_column="matrix_probe_id",
        gene_id_column="ensembl_stable_id",
        gene_symbol_column="hgnc_symbol",
    )
    out_gbc = tmp_path / "out_gbc.h5ad"
    importer.canonicalize(
        source_set=src_gbc,
        output_path=str(out_gbc),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=map_gbc,
    )
    adata_gbc = ad.read_h5ad(out_gbc)
    assert list(adata_gbc.obs_names) == ["donor1_well_A01", "donor1_well_B04"]
    assert list(adata_gbc.var_names) == ["ENSG00000124762", "ENSG00000142168", "ENSG00000182463"]
    assert list(adata_gbc.var["source_feature_id"]) == ["probe_alpha", "probe_beta", "probe_gamma"]
    assert list(adata_gbc.var["gene_symbol"]) == ["CDKN1A", "SOD1", "ACTB"]
    np.testing.assert_array_equal(adata_gbc.X.toarray(), expected_mat)


def test_cell_metadata_id_join_unrelated_row_positions(tmp_path: Path) -> None:
    """Proves CELL_METADATA joins strictly by cellMetadataJoinColumn value to authoritative matrix cell tokens,
    never by DataFrame row number, using deliberately unrelated row positions and extra unreferenced rows.
    """
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    mat_file = tmp_path / "mat_cells.csv"
    create_csv_matrix(
        mat_file,
        data=[
            ["cell_Z", 1, 2],
            ["cell_A", 3, 4],
            ["cell_M", 5, 6],
        ],
        headers=["matrix_cid", "ENSG00000124762", "ENSG00000142168"],
    )

    cell_meta = tmp_path / "meta_unrelated_pos.csv"
    create_csv_matrix(
        cell_meta,
        data=[
            ["noise_0", "unreferenced_1", "batch_X", "ctrl"],
            ["noise_1", "cell_M", "batch_3", "stim"],
            ["noise_2", "unreferenced_2", "batch_Y", "ctrl"],
            ["noise_3", "cell_Z", "batch_1", "ctrl"],
            ["noise_4", "cell_A", "batch_2", "stim"],
        ],
        headers=["row_tag", "authoritative_cell_key", "batch_col", "cond_col"],
    )

    source_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
        ]
    )

    bad_mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="matrix_cid",
        cell_metadata_join_column="non_existent_col",
        batch_column="batch_col",
    )
    with pytest.raises(ImporterSecurityError) as exc:
        importer.canonicalize(
            source_set=source_set,
            output_path=str(tmp_path / "fail_col.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=bad_mapping,
        )
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA

    valid_mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        matrix_cell_id_column="matrix_cid",
        cell_metadata_join_column="authoritative_cell_key",
        batch_column="batch_col",
        condition_column="cond_col",
    )
    out_h5ad = tmp_path / "realigned_cells.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=valid_mapping,
    )
    adata = ad.read_h5ad(out_h5ad)
    assert list(adata.obs_names) == ["cell_Z", "cell_A", "cell_M"]
    assert list(adata.obs["batch"]) == ["batch_1", "batch_2", "batch_3"]
    assert list(adata.obs["condition"]) == ["ctrl", "stim", "stim"]
    np.testing.assert_array_equal(adata.X.toarray(), np.array([[1, 2], [3, 4], [5, 6]]))
