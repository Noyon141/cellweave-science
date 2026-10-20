"""Comprehensive unit and golden tests for Phase 3 Step G Generic Matrix Importer:
- GenericMatrixImporter (GENERIC_MATRIX)

Test matrix:
1. Protocol conformance & Configuration Safety (satisfies DatasetImporter, fail-closed limits).
2. Delimiter & Container Sniffing (CSV, TSV, MTX, ambiguous delimiters, comment handling).
3. Format Detection & Cardinality (valid roles MATRIX, optional CELL_METADATA / FEATURE_METADATA).
4. Validation logic (structural errors, non-numeric tables, dimension mismatch, count anomalies).
5. Inspection logic (datasetVersionId requirement, NEEDS_MAPPING status, column discovery).
6. Golden CSV Tests (CELLS_BY_GENES & GENES_BY_CELLS with exact non-square matrix verification).
7. Golden TSV Tests (plain and bounded gzip).
8. Golden MTX Tests (coordinate format with external CELL_METADATA and FEATURE_METADATA).
9. Metadata joins & realignment (out-of-order rows, duplicate IDs, missing IDs fail-closed).
10. Strict Sample Mapping Authority (no source pass-through, complete mapping requirement).
11. Biological annotation mappings (condition, batch, biological/technical replicates).
12. Raw count strictness (integer, integral floats, fractional, negative, NaN/Inf, ambiguous).
13. Policy B canonicalization slot hygiene (CSR in X, empty layers/obsm/varm, controlled uns).
14. Source immutability & path collision safety.
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


def test_sniff_delimiter_no_delimiters_fails_closed(tmp_path: Path) -> None:
    no_delim = tmp_path / "nodelim.txt"
    no_delim.write_text("single_column_token_only\nvalue1\n", encoding="utf-8")
    with pytest.raises(CountValidationError) as exc:
        sniff_delimiter(no_delim, DEFAULT_TEST_LIMITS)
    assert exc.value.issue_code == ScientificIssueCode.FORMAT_MISMATCH


# =========================================================================
# 3. FORMAT DETECTION & CARDINALITY
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
# 4. VALIDATION LOGIC
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

    # Single column table (no delimiter found)
    single_col = tmp_path / "single_col.csv"
    single_col.write_text("cell_id\nc1\nc2\n", encoding="utf-8")
    single_set = create_source_set([create_source_file(single_col, ImportFileRole.MATRIX)])
    issues = importer.validate(single_set)
    assert any(
        iss.code in (ScientificIssueCode.MALFORMED_INPUT, ScientificIssueCode.FORMAT_MISMATCH)
        for iss in issues
    )


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
# 5. INSPECTION LOGIC
# =========================================================================


def test_inspect_dataset_version_id_required(tmp_path: Path) -> None:
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


def test_inspect_unresolved_species_conflict_returns_invalid(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    # Row 0 containing both Human (ENSG) and Mouse (ENSMUSG) IDs
    mat_file = tmp_path / "conflict.csv"
    mat_file.write_text(
        "gene_id,c1,c2\n" "ENSG00000124762,10,20\n" "ENSMUSG00000024046,30,40\n",
        encoding="utf-8",
    )
    source_set = create_source_set([create_source_file(mat_file, ImportFileRole.MATRIX)])
    inspection = importer.inspect(source_set)
    assert inspection.status == InspectionStatus.INVALID
    assert inspection.requiresMapping is False
    assert any(iss.code == ScientificIssueCode.UNRESOLVED_SPECIES for iss in inspection.issues)


# =========================================================================
# 6. GOLDEN CSV TESTS: CELLS_BY_GENES & GENES_BY_CELLS
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
# 7. GOLDEN TSV TESTS (PLAIN & GZIP)
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
# 8. GOLDEN MTX TESTS
# =========================================================================


def test_golden_mtx_cells_by_genes(tmp_path: Path) -> None:
    """Matrix Market 2 cells x 3 genes with external metadata."""
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    # MTX coordinate file: 2 rows (cells), 3 cols (genes), 4 non-zero entries
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
# 9. METADATA JOINS & REALIGNMENT
# =========================================================================


def test_cell_metadata_realignment_and_missing_id_fail_closed(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)

    # Matrix has cells in order: cell_A, cell_B, cell_C
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[
            ["cell_A", 1, 2],
            ["cell_B", 3, 4],
            ["cell_C", 5, 6],
        ],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )

    # Metadata has cells in REVERSE order: cell_C, cell_A, cell_B
    cell_meta = tmp_path / "cells.csv"
    create_csv_matrix(
        cell_meta,
        data=[
            ["cell_C", "c_condition"],
            ["cell_A", "a_condition"],
            ["cell_B", "b_condition"],
        ],
        headers=["meta_cell_id", "condition_source"],
    )

    source_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
        ]
    )

    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        cell_id_column="cell_id",
        condition_column="condition_source",
    )

    out_h5ad = tmp_path / "out_realigned.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )

    adata = ad.read_h5ad(out_h5ad)
    assert list(adata.obs_names) == ["cell_A", "cell_B", "cell_C"]
    assert list(adata.obs["condition"]) == ["a_condition", "b_condition", "c_condition"]

    # Now test MISSING cell ID: cell_C is missing from metadata
    bad_meta = tmp_path / "bad_cells.csv"
    create_csv_matrix(
        bad_meta,
        data=[
            ["cell_A", "a_condition"],
            ["cell_B", "b_condition"],
        ],
        headers=["meta_cell_id", "condition_source"],
    )
    bad_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(bad_meta, ImportFileRole.CELL_METADATA),
        ]
    )
    with pytest.raises(ImporterSecurityError) as exc:
        importer.canonicalize(
            source_set=bad_set,
            output_path=str(out_h5ad),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
    assert exc.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


def test_cell_metadata_duplicate_ids_fails_closed(tmp_path: Path) -> None:
    importer = GenericMatrixImporter(DEFAULT_TEST_CONFIG)
    mat_file = tmp_path / "mat.csv"
    create_csv_matrix(
        mat_file,
        data=[["cell_A", 1, 2], ["cell_B", 3, 4]],
        headers=["cell_id", "ENSG00000124762", "ENSG00000142168"],
    )

    dup_meta = tmp_path / "dup_cells.csv"
    create_csv_matrix(
        dup_meta,
        data=[
            ["cell_A", "cond1"],
            ["cell_A", "cond2"],  # Duplicate!
        ],
        headers=["meta_cell_id", "cond"],
    )

    source_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(dup_meta, ImportFileRole.CELL_METADATA),
        ]
    )
    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        cell_id_column="cell_id",
    )

    out_h5ad = tmp_path / "out_dup.h5ad"
    with pytest.raises(CountValidationError) as exc:
        importer.canonicalize(
            source_set=source_set,
            output_path=str(out_h5ad),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
    assert exc.value.issue_code == ScientificIssueCode.DUPLICATE_CELL_IDS


# =========================================================================
# 10. STRICT SAMPLE MAPPING AUTHORITY
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
# 11. BIOLOGICAL ANNOTATION MAPPINGS
# =========================================================================


def test_biological_annotation_mappings(tmp_path: Path) -> None:
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
        data=[["c1", "ctrl", "batch_1", "bio_rep_1", "tech_rep_1", "unmapped_col"]],
        headers=["cell_id", "cond", "batch_src", "bio", "tech", "unmapped"],
    )

    source_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
        ]
    )

    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        condition_column="cond",
        batch_column="batch_src",
        biological_replicate_column="bio",
        technical_replicate_column="tech",
    )

    out_h5ad = tmp_path / "out_annot.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_h5ad),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )

    adata = ad.read_h5ad(out_h5ad)
    assert adata.obs["condition"].iloc[0] == "ctrl"
    assert adata.obs["batch"].iloc[0] == "batch_1"
    assert adata.obs["biological_replicate"].iloc[0] == "bio_rep_1"
    assert adata.obs["technical_replicate"].iloc[0] == "tech_rep_1"
    # Unmapped column must NOT be present in obs
    assert "unmapped" not in adata.obs.columns


def test_missing_requested_column_fails_closed(tmp_path: Path) -> None:
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
        data=[["c1", "ctrl"]],
        headers=["cell_id", "cond"],
    )
    source_set = create_source_set(
        [
            create_source_file(mat_file, ImportFileRole.MATRIX),
            create_source_file(cell_meta, ImportFileRole.CELL_METADATA),
        ]
    )

    mapping = make_mapping(
        orientation=MatrixOrientation.CELLS_BY_GENES,
        batch_column="non_existent_batch_column",
    )
    out_h5ad = tmp_path / "out_annot.h5ad"
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
# 12. RAW COUNT STRICTNESS
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
# 13. POLICY B OUTPUT & SLOT HYGIENE
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
# 14. SOURCE IMMUTABILITY & PATH COLLISION SAFETY
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

    # Attempting to write canonical output directly onto the source file
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
