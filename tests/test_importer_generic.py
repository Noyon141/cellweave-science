"""Comprehensive unit, golden, and regression tests for Phase 3 Step G Generic Matrix Importer:
- GenericMatrixImporter (GENERIC_MATRIX)

Test matrix:
1. Protocol conformance & Configuration Safety (satisfies DatasetImporter, fail-closed limits).
2. Delimiter & Container Sniffing (CSV, TSV, MTX, ambiguous delimiters, comment handling).
3. Delimiter Extension-Agnostic Policy (content-based detection; .csv vs .tsv produce identical interpretations).
4. Format Detection & Cardinality (valid roles MATRIX, optional CELL_METADATA / FEATURE_METADATA).
5. Validation logic (structural errors, non-numeric tables, dimension mismatch, count anomalies, duplicate IDs).
6. Inspection logic (datasetVersionId authority, NEEDS_MAPPING status, column discovery).
7. Inspection status matrix (NEEDS_MAPPING vs INVALID across failure modes).
8. Golden CSV Tests (CELLS_BY_GENES & GENES_BY_CELLS with exact non-square matrix verification).
9. Golden TSV Tests (plain and bounded gzip).
10. Golden MTX Tests (coordinate format with external CELL_METADATA and FEATURE_METADATA).
11. Feature metadata join test matrix (A: same-order, B: reordered, C: duplicate, D: missing, E: extra rows).
12. Cell metadata join test matrix (A: same-order, B: reordered, C: duplicate, D: missing, E: extra rows).
13. Strict Sample Mapping Authority (no source pass-through, complete mapping requirement).
14. Biological annotation mappings (condition, batch, biological/technical replicates).
15. Canonical obs schema tests (complete mapping vs minimal mapping with zero optional columns).
16. Canonical var schema tests (stable IDs, source_feature_id, gene_symbol only when mapped).
17. Stable ID + duplicate symbol test (valid when stable IDs are unique).
18. Organism and Gene ID Scheme resolution tests.
19. Reference build preservation (explicit preserved, missing is None).
20. Raw count strictness (integer, integral floats, fractional, negative, NaN/Inf, ambiguous near-integral).
21. Policy B canonicalization slot hygiene (CSR in X, empty layers/obsm/varm, controlled uns).
22. Source immutability & path collision safety.
23. Exact round-trip assert for CSV and MTX fixtures.
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
    sample_column: str | None = None,
    sample_mappings: list[SampleMappingEntry] | None = None,
    condition_column: str | None = None,
    batch_column: str | None = None,
    biological_replicate_column: str | None = None,
    technical_replicate_column: str | None = None,
    reference_build: str | None = None,
) -> MetadataMappingSubmission:
    """Helper to instantiate fully valid MetadataMappingSubmission with defaults."""
    return MetadataMappingSubmission(
        datasetVersionId=dataset_version_id,
        organism=organism,
        geneIdScheme=gene_id_scheme,
        orientation=orientation,
        cellIdColumn=cell_id_column,
        geneIdColumn=gene_id_column,
        geneSymbolColumn=gene_symbol_column,
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

    # Both infer identical delimiter strictly from content
    delim_csv = sniff_delimiter(csv_file, DEFAULT_TEST_LIMITS)
    delim_tsv = sniff_delimiter(tsv_file, DEFAULT_TEST_LIMITS)
    assert delim_csv == delim_tsv

    # Both parse into identical single-column DataFrames
    df_csv, _ = read_delimited_table(csv_file, DEFAULT_TEST_LIMITS)
    df_tsv, _ = read_delimited_table(tsv_file, DEFAULT_TEST_LIMITS)
    pd.testing.assert_frame_equal(df_csv, df_tsv)
    assert list(df_csv.columns) == ["token_column"]
    assert df_csv["token_column"].tolist() == ["val_1", "val_2", "val_3"]

    # When evaluated as expression matrices, both fail identically with MALFORMED_INPUT
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

    # 1. Plain CSV
    csv_file = tmp_path / "mat.csv"
    csv_file.write_text("cell_id,g1,g2\nc1,1,2\n", encoding="utf-8")
    source_set_csv = create_source_set([create_source_file(csv_file, ImportFileRole.MATRIX)])
    assert importer.detect(source_set_csv) is True

    # 2. Plain TSV
    tsv_file = tmp_path / "mat.tsv"
    tsv_file.write_text("cell_id\tg1\tg2\nc1\t1\t2\n", encoding="utf-8")
    source_set_tsv = create_source_set([create_source_file(tsv_file, ImportFileRole.MATRIX)])
    assert importer.detect(source_set_tsv) is True

    # 3. Plain MTX
    mtx_file = tmp_path / "mat.mtx"
    create_mtx_file(mtx_file, 2, 2, [(1, 1, 5)])
    source_set_mtx = create_source_set([create_source_file(mtx_file, ImportFileRole.MATRIX)])
    assert importer.detect(source_set_mtx) is True

    # 4. Gzipped MTX
    mtx_gz = tmp_path / "mat.mtx.gz"
    create_mtx_file(mtx_gz, 2, 2, [(1, 1, 5)], compress=True)
    source_set_mtx_gz = create_source_set([create_source_file(mtx_gz, ImportFileRole.MATRIX)])
    assert importer.detect(source_set_mtx_gz) is True


def test_detect_rejects_unsupported_formats_and_cardinality(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    # Non-generic format
    csv_file = tmp_path / "mat.csv"
    csv_file.write_text("c,g\n1,2\n", encoding="utf-8")
    non_generic_set = create_source_set(
        [create_source_file(csv_file, ImportFileRole.MATRIX)],
        source_format=SourceFormat.H5AD,
    )
    assert importer.detect(non_generic_set) is False

    # Disallowed role: PRIMARY instead of MATRIX
    primary_set = create_source_set([create_source_file(csv_file, ImportFileRole.PRIMARY)])
    assert importer.detect(primary_set) is False

    # Disallowed role: BARCODES present
    bc_file = tmp_path / "barcodes.tsv"
    bc_file.write_text("c1\n", encoding="utf-8")
    invalid_roles_set = create_source_set(
        [
            create_source_file(csv_file, ImportFileRole.MATRIX),
            create_source_file(bc_file, ImportFileRole.BARCODES),
        ]
    )
    assert importer.detect(invalid_roles_set) is False

    # Multiple MATRIX files
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

    # Missing MATRIX
    empty_set = create_source_set([])
    issues = importer.validate(empty_set)
    assert any(iss.code == ScientificIssueCode.MISSING_COMPONENT for iss in issues)

    # Multiple MATRIX
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

    # Non-numeric values in expression table
    bad_mat = tmp_path / "bad_mat.csv"
    bad_mat.write_text("cell_id,g1,g2\nc1,text,10\n", encoding="utf-8")
    source_set = create_source_set([create_source_file(bad_mat, ImportFileRole.MATRIX)])
    issues = importer.validate(source_set)
    assert any(iss.code == ScientificIssueCode.MALFORMED_INPUT for iss in issues)

    # Single column table (no genes)
    single_col = tmp_path / "single_col.csv"
    single_col.write_text("cell_id\nc1\nc2\n", encoding="utf-8")
    single_set = create_source_set([create_source_file(single_col, ImportFileRole.MATRIX)])
    issues = importer.validate(single_set)
    assert any(iss.code == ScientificIssueCode.MALFORMED_INPUT for iss in issues)


def test_validate_duplicate_authoritative_matrix_ids(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    # Duplicate row identifiers (column 0)
    dup_rows = tmp_path / "dup_rows.csv"
    dup_rows.write_text("cell_id,g1,g2\nc1,10,20\nc1,30,40\n", encoding="utf-8")
    issues_row = importer.validate(
        create_source_set([create_source_file(dup_rows, ImportFileRole.MATRIX)])
    )
    assert any(iss.code == ScientificIssueCode.DUPLICATE_CELL_IDS for iss in issues_row)

    # Duplicate column identifiers (row 0 header)
    dup_cols = tmp_path / "dup_cols.csv"
    dup_cols.write_text("cell_id,g1,g1\nc1,10,20\nc2,30,40\n", encoding="utf-8")
    issues_col = importer.validate(
        create_source_set([create_source_file(dup_cols, ImportFileRole.MATRIX)])
    )
    assert any(iss.code == ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS for iss in issues_col)


def test_validate_raw_counts(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    # Fractional values (log-normalized / CPM)
    frac_mat = tmp_path / "frac_mat.csv"
    frac_mat.write_text("cell_id,g1,g2\nc1,1.5,2.0\n", encoding="utf-8")
    frac_set = create_source_set([create_source_file(frac_mat, ImportFileRole.MATRIX)])
    issues = importer.validate(frac_set)
    assert any(iss.code == ScientificIssueCode.NON_COUNT_INPUT for iss in issues)

    # Negative values
    neg_mat = tmp_path / "neg_mat.csv"
    neg_mat.write_text("cell_id,g1,g2\nc1,-1,2\n", encoding="utf-8")
    neg_set = create_source_set([create_source_file(neg_mat, ImportFileRole.MATRIX)])
    issues = importer.validate(neg_set)
    assert any(iss.code == ScientificIssueCode.NON_COUNT_INPUT for iss in issues)

    # Ambiguous near-integral values
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

    # Missing dataset_version_id
    source_set_no_ver = ImportSourceSet(
        source_format=SourceFormat.GENERIC_MATRIX,
        files=(create_source_file(csv_file, ImportFileRole.MATRIX),),
        dataset_version_id=None,
    )
    with pytest.raises(ImporterSecurityError) as exc:
        importer.inspect(source_set_no_ver)
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA

    # Whitespace only dataset_version_id
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
# 7. INSPECTION STATUS MATRIX
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

    # 4. Duplicate authoritative IDs => INVALID
    p_dup = tmp_path / "dup.csv"
    p_dup.write_text("cell_id,g1,g2\nc1,1,2\nc1,3,4\n", encoding="utf-8")
    res4 = importer.inspect(create_source_set([create_source_file(p_dup, ImportFileRole.MATRIX)]))
    assert res4.status == InspectionStatus.INVALID
    assert res4.requiresMapping is False
    assert any(iss.code == ScientificIssueCode.DUPLICATE_CELL_IDS for iss in res4.issues)


# =========================================================================
# 8. GOLDEN CSV TESTS: CELLS_BY_GENES & GENES_BY_CELLS
# =========================================================================


def test_golden_csv_cells_by_genes(tmp_path: Path) -> None:
    """Non-square 2 cells x 3 genes matrix (CELLS_BY_GENES).

    Matrix:
             ENSG00000124762  ENSG00000142168  ENSG00000182463
    cell_A          10               20               30
    cell_B          40               50               60
    """
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
        cell_id_column="cell_id",
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
    """Non-square 3 genes x 2 cells matrix (GENES_BY_CELLS).

    Transpose must yield 2 cells x 3 genes.
    Source:
    gene_id          cell_A  cell_B
    ENSG00000124762    10      40
    ENSG00000142168    20      50
    ENSG00000182463    30      60
    """
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
        gene_id_column="gene_id",
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

    # Transposed result should match cell_A: [10, 20, 30], cell_B: [40, 50, 60]
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

    # Gzipped TSV
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
        cell_id_column="cell_id",
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
# 10. GOLDEN MTX TESTS
# =========================================================================


def test_golden_mtx_cells_by_genes(tmp_path: Path) -> None:
    """Matrix Market 2 cells x 3 genes with external metadata."""
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
        data=[["C1", "sample_A"], ["C2", "sample_B"]],
        headers=["cell_id", "sample_source"],
    )

    feat_meta = tmp_path / "genes.csv"
    create_csv_matrix(
        feat_meta,
        data=[
            ["ENSG00000124762", "CDKN1A"],
            ["ENSG00000142168", "SOD1"],
            ["ENSG00000182463", "ACTB"],
        ],
        headers=["gene_id", "symbol"],
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

    dense_x = adata.X.toarray()
    expected = np.array([[10, 0, 30], [0, 50, 60]], dtype=np.int32)
    np.testing.assert_array_equal(dense_x, expected)


def test_golden_mtx_genes_by_cells(tmp_path: Path) -> None:
    """Matrix Market 3 rows (genes) x 2 cols (cells) with transpose."""
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
        data=[["C1"], ["C2"]],
        headers=["cell_id"],
    )

    feat_meta = tmp_path / "genes.csv"
    create_csv_matrix(
        feat_meta,
        data=[
            ["ENSG00000124762"],
            ["ENSG00000142168"],
            ["ENSG00000182463"],
        ],
        headers=["gene_id"],
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


def test_mtx_without_metadata_fails_closed(tmp_path: Path) -> None:
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
# 11. FEATURE METADATA JOIN TEST MATRIX (A-E)
# =========================================================================


def test_feature_metadata_join_matrix_cases(tmp_path: Path) -> None:
    """Test full matrix of FEATURE_METADATA joining policies:

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
            ["probe_unrelated", "ENSG00000999999", "EXTRA"],  # Extra unreferenced row
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
    # Alignment strictly matches matrix columns order [probe_1, probe_2]
    assert list(adata_reord.var_names) == ["ENSG00000124762", "ENSG00000142168"]
    assert list(adata_reord.var["gene_symbol"]) == ["CDKN1A", "SOD1"]

    # C: Duplicate metadata join key
    feat_meta_dup = tmp_path / "feat_dup.csv"
    create_csv_matrix(
        feat_meta_dup,
        data=[
            ["probe_1", "ENSG00000124762", "CDKN1A"],
            ["probe_1", "ENSG00000142168", "SOD1"],  # Duplicate probe_1
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
            # probe_2 missing!
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
# 12. CELL METADATA JOIN TEST MATRIX (A-E)
# =========================================================================


def test_cell_metadata_join_matrix_cases(tmp_path: Path) -> None:
    """Test full matrix of CELL_METADATA joining policies:

    A. Perfect same-order identity match
    B. Reordered relative to matrix => deterministic realignment
    C. Duplicate cell key => reject (DUPLICATE_CELL_IDS)
    D. Missing matrix cell => reject (MISSING_REQUIRED_METADATA)
    E. Extra unrelated metadata row => extra row ignored, matrix cells resolve.
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

    # A & E: Extra rows and same order
    cell_meta_extra = tmp_path / "cells_extra.csv"
    create_csv_matrix(
        cell_meta_extra,
        data=[
            ["cell_A", "cond_A"],
            ["cell_B", "cond_B"],
            ["cell_unrelated", "cond_X"],  # Extra unreferenced row
        ],
        headers=["cell_id", "cond"],
    )
    src_extra = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta_extra, ImportFileRole.CELL_METADATA),
        ]
    )
    mapping_extra = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        cell_id_column="cell_id",
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
    assert list(adata_extra.obs_names) == ["cell_A", "cell_B"]
    assert list(adata_extra.obs["condition"]) == ["cond_A", "cond_B"]

    # B: Reordered metadata relative to matrix
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
    out_reord = tmp_path / "out_cell_reord.h5ad"
    importer.canonicalize(
        source_set=src_reord,
        output_path=str(out_reord),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping_extra,
    )
    adata_reord = ad.read_h5ad(out_reord)
    # Alignment matches matrix rows order [cell_A, cell_B]
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
            mapping=mapping_extra,
        )
    assert isinstance(exc_dup.value, CountValidationError | ImporterSecurityError)
    assert exc_dup.value.issue_code == ScientificIssueCode.DUPLICATE_CELL_IDS

    # D: Missing matrix cell
    cell_meta_missing = tmp_path / "cells_missing.csv"
    create_csv_matrix(
        cell_meta_missing,
        data=[
            ["cell_A", "cond_A"],
            # cell_B missing!
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
            mapping=mapping_extra,
        )
    assert exc_miss.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


# =========================================================================
# 13. STRICT SAMPLE MAPPING AUTHORITY
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
# 14. BIOLOGICAL ANNOTATIONS & CANONICAL OBS SCHEMA TESTS
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
        cell_id_column="cell_id",
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
# 15. CANONICAL VAR SCHEMA & STABLE ID + DUPLICATE SYMBOLS TEST
# =========================================================================


def test_stable_id_with_duplicate_symbols_valid(tmp_path: Path) -> None:
    """Proves duplicate symbols are valid when stable canonical IDs are unique:

    source_feature_id: feature_a, feature_b
    geneIdColumn: ENSG00000124762, ENSG00000142168
    geneSymbolColumn: DUP, DUP
    """
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
            ["feature_b", "ENSG00000142168", "DUP"],  # Duplicate symbol, unique stable ID
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
    mapping = make_mapping(orientation=MatrixOrientation.CELLS_BY_GENES)
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
# 16. ORGANISM, GENE SCHEME & REFERENCE BUILD RESOLUTION
# =========================================================================


def test_organism_resolution_policy_step_d(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    # 100% Human Ensembl IDs => HUMAN candidate
    mat_h = tmp_path / "human.csv"
    create_csv_matrix(
        mat_h, data=[["c1", 10, 20]], headers=["cell_id", "ENSG00000124762", "ENSG00000142168"]
    )
    res_h = importer.inspect(create_source_set([create_source_file(mat_h, ImportFileRole.MATRIX)]))
    assert res_h.detectedOrganism == Organism.HUMAN
    assert res_h.detectedGeneIdScheme == GeneIdScheme.ENSEMBL

    # 100% Mouse Ensembl IDs => MOUSE candidate
    mat_m = tmp_path / "mouse.csv"
    create_csv_matrix(
        mat_m,
        data=[["c1", 10, 20]],
        headers=["cell_id", "ENSMUSG00000024046", "ENSMUSG00000019929"],
    )
    res_m = importer.inspect(create_source_set([create_source_file(mat_m, ImportFileRole.MATRIX)]))
    assert res_m.detectedOrganism == Organism.MOUSE
    assert res_m.detectedGeneIdScheme == GeneIdScheme.ENSEMBL

    # Human + Mouse mixture => conflict / INVALID
    mat_mix = tmp_path / "mix.csv"
    create_csv_matrix(
        mat_mix, data=[["c1", 10, 20]], headers=["cell_id", "ENSG00000124762", "ENSMUSG00000024046"]
    )
    res_mix = importer.inspect(
        create_source_set([create_source_file(mat_mix, ImportFileRole.MATRIX)])
    )
    assert res_mix.status == InspectionStatus.INVALID
    assert any(iss.code == ScientificIssueCode.UNRESOLVED_SPECIES for iss in res_mix.issues)

    # Symbol-only => None (no organism inference)
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

    # 1. Explicit reference build preserved
    mapping_with_ref = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
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

    # 2. Missing reference build remains None (never guessed)
    mapping_no_ref = make_mapping(orientation=MatrixOrientation.CELLS_BY_GENES)
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
# 17. RAW COUNT STRICTNESS
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
    mapping = make_mapping(orientation=MatrixOrientation.CELLS_BY_GENES)
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


# =========================================================================
# 18. POLICY B OUTPUT & SLOT HYGIENE
# =========================================================================


def test_slot_hygiene_and_uns_provenance(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["c1", 10, 20]],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )
    source_set = create_source_set([create_source_file(mat_file, ImportFileRole.MATRIX)])
    mapping = make_mapping(orientation=MatrixOrientation.CELLS_BY_GENES)
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
# 19. SOURCE IMMUTABILITY & PATH COLLISION SAFETY
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
    mapping = make_mapping(orientation=MatrixOrientation.CELLS_BY_GENES)

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


# =========================================================================
# 20. EXACT ROUND-TRIP ASSERTION (CSV & MTX)
# =========================================================================


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
        cell_id_column="cell_id",
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
    create_csv_matrix(cell_meta_mtx, data=[["C_A"], ["C_B"]], headers=["cell_id"])
    feat_meta_mtx = tmp_path / "feat_meta_mtx.csv"
    create_csv_matrix(
        feat_meta_mtx,
        data=[["ENSG00000124762", "CDKN1A"], ["ENSG00000142168", "SOD1"]],
        headers=["gene_id", "symbol"],
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
