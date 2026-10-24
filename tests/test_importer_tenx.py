"""Comprehensive unit and golden tests for Phase 3 Step F 10x Genomics importers:
- TenxH5Importer (TENX_H5)
- TenxMtxImporter (TENX_MTX)

Test matrix:
1. Protocol conformance (satisfies DatasetImporter).
2. H5 detection, structural parsing, missing components, dimensions, corrupt indices.
3. MTX detection, Matrix Market parsing, missing roles, header, dimensions.
4. Modality policy: pure Gene Expression (VALID) vs mixed/unsupported (UNSUPPORTED_MODALITY).
5. Modern features.tsv vs legacy genes.tsv (plain & gzip).
6. Barcodes (plain & gzip, suffix preservation, duplicate/empty rejection).
7. Mandatory orientation transpose golden tests (3 genes x 2 cells -> 2 cells x 3 genes with exact values).
8. Organism resolution matrix: 100% Human, 100% Mouse, mixed conflict, Ensembl+symbols unresolved.
9. Feature ID tests: versioned Ensembl preservation, duplicate stable IDs rejection.
10. Count validation: integers, ambiguous near-integral, fractional, negative.
11. Slot hygiene: layers clean, obsm/varm clean, controlled uns only.
12. Round-trip verification: H5 and MTX canonical artifacts read back and validated.
13. Source immutability: source SHA identical before and after canonicalization.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from cellweave_science.contracts import (
    GeneIdScheme,
    ImportFileRole,
    InspectionStatus,
    MetadataMappingSubmission,
    Organism,
    SampleMappingEntry,
    ScientificIssueCode,
    ScientificIssueSeverity,
    SourceFormat,
)
from cellweave_science.importers import (
    DatasetImporter,
    ImportSourceFile,
    ImportSourceSet,
    TenxH5Importer,
    TenxMtxImporter,
    compute_file_sha256,
)
from cellweave_science.importers.canonical import validate_canonical_anndata
from cellweave_science.importers.errors import ImporterSecurityError
from cellweave_science.importers.security import DecompressionLimits
from cellweave_science.importers.tenx_shared import TenxImporterConfig, parse_10x_barcodes

DEFAULT_TEST_LIMITS = DecompressionLimits(
    max_uncompressed_bytes=100 * 1024 * 1024,
    max_compression_ratio=200.0,
)
DEFAULT_TEST_CONFIG = TenxImporterConfig(decompression_limits=DEFAULT_TEST_LIMITS)


def test_tenx_unicode_identity_survives_gzip_chunk_boundary(tmp_path: Path) -> None:
    # Split the two-byte UTF-8 character across the 64 KiB decompression boundary.
    token = "A" * 65535 + "α-1"
    path = tmp_path / "barcodes.tsv.gz"
    with gzip.open(path, "wb") as stream:
        stream.write((token + "\n").encode("utf-8"))
    limits = DecompressionLimits(max_uncompressed_bytes=100000, max_compression_ratio=10000)
    assert parse_10x_barcodes(path, limits) == [token]


@pytest.mark.parametrize("kind", ["h5", "modern", "legacy"])
@pytest.mark.parametrize("axis", ["barcode", "feature"])
@pytest.mark.parametrize("padding", ["leading", "trailing", "both", "tab", "newline"])
def test_tenx_identity_whitespace_fails_closed(
    tmp_path: Path, kind: str, axis: str, padding: str
) -> None:
    token = "AAACCCAAGT-1" if axis == "barcode" else "ENSG00000141510.7"
    invalid = {
        "leading": " " + token,
        "trailing": token + " ",
        "both": " " + token + " ",
        "tab": "\t" + token,
        "newline": token + "\n",
    }[padding]
    barcode = invalid if axis == "barcode" else "AAACCCAAGT-1"
    feature = invalid if axis == "feature" else "ENSG00000141510.7"
    importer: TenxH5Importer | TenxMtxImporter
    if kind == "h5":
        path = create_synthetic_10x_h5(
            tmp_path / "source.h5",
            [(feature, "TP53", "Gene Expression")],
            [barcode],
            np.array([1]),
            np.array([0]),
            np.array([0, 1]),
        )
        bundle = ImportSourceSet(
            SourceFormat.TENX_H5, (ImportSourceFile("primary", ImportFileRole.PRIMARY, path),)
        )
        importer = TenxH5Importer()
    else:
        features: list[tuple[str, str, str]] | list[tuple[str, str]]
        if kind == "legacy":
            features = [(feature, "TP53")]
        else:
            features = [(feature, "TP53", "Gene Expression")]
        matrix, feature_path, barcode_path = create_synthetic_10x_mtx(
            tmp_path / "triplet",
            features,
            [barcode],
            [(1, 1, 1)],
            legacy_genes=kind == "legacy",
            gzip_features=True,
            gzip_barcodes=True,
        )
        bundle = ImportSourceSet(
            SourceFormat.TENX_MTX,
            (
                ImportSourceFile("m", ImportFileRole.MATRIX, matrix),
                ImportSourceFile("f", ImportFileRole.FEATURES, feature_path),
                ImportSourceFile("b", ImportFileRole.BARCODES, barcode_path),
            ),
        )
        importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    checksums = [compute_file_sha256(file.local_path) for file in bundle.files]
    output = tmp_path / "out.h5ad"
    with pytest.raises(ImporterSecurityError):
        importer.canonicalize(bundle, str(output), Organism.HUMAN, GeneIdScheme.ENSEMBL)
    assert not output.exists()
    assert [compute_file_sha256(file.local_path) for file in bundle.files] == checksums


@pytest.mark.parametrize("field", ["barcodes", "features/id"])
@pytest.mark.parametrize("value", [1, 1.0, True])
def test_tenx_h5_numeric_identity_is_not_stringified(
    tmp_path: Path, field: str, value: object
) -> None:
    path = create_synthetic_10x_h5(
        tmp_path / "source.h5",
        [("ENSG00000141510.7", "TP53", "Gene Expression")],
        ["AAACCCAAGT-1"],
        np.array([1]),
        np.array([0]),
        np.array([0, 1]),
    )
    with h5py.File(path, "r+") as handle:
        del handle[f"matrix/{field}"]
        handle.create_dataset(f"matrix/{field}", data=np.array([value]))
    bundle = ImportSourceSet(
        SourceFormat.TENX_H5, (ImportSourceFile("p", ImportFileRole.PRIMARY, path),)
    )
    with pytest.raises(ImporterSecurityError):
        TenxH5Importer().canonicalize(
            bundle, str(tmp_path / "out.h5ad"), Organism.HUMAN, GeneIdScheme.ENSEMBL
        )


# =========================================================================
# FIXTURE BUILDER HELPERS
# =========================================================================


def create_synthetic_10x_h5(
    file_path: Path,
    features: list[tuple[str, str, str]],  # (id, name, feature_type)
    barcodes: list[str],
    data: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    indptr: np.ndarray[Any, Any],
    *,
    custom_shape: tuple[int, int] | None = None,
) -> Path:
    """Create a synthetic 10x HDF5 filtered feature barcode matrix."""
    shape = custom_shape if custom_shape is not None else (len(features), len(barcodes))
    with h5py.File(file_path, "w") as h5:
        mat = h5.create_group("matrix")
        mat.create_dataset("data", data=data)
        mat.create_dataset("indices", data=indices)
        mat.create_dataset("indptr", data=indptr)
        mat.create_dataset("shape", data=np.array(shape, dtype=np.int32))
        mat.create_dataset(
            "barcodes",
            data=np.array([b.encode("utf-8") for b in barcodes], dtype=object),
        )

        feat_grp = mat.create_group("features")
        feat_grp.create_dataset(
            "id", data=np.array([f[0].encode("utf-8") for f in features], dtype=object)
        )
        feat_grp.create_dataset(
            "name", data=np.array([f[1].encode("utf-8") for f in features], dtype=object)
        )
        feat_grp.create_dataset(
            "feature_type",
            data=np.array([f[2].encode("utf-8") for f in features], dtype=object),
        )

    return file_path


def create_synthetic_10x_mtx(
    dir_path: Path,
    features: list[tuple[str, str, str]] | list[tuple[str, str]],
    barcodes: list[str],
    entries: list[tuple[int, int, int | float]],  # 1-based (row, col, value)
    *,
    gzip_matrix: bool = False,
    gzip_features: bool = False,
    gzip_barcodes: bool = False,
    legacy_genes: bool = False,
    header_override: str | None = None,
    dimensions_override: tuple[int, int, int] | None = None,
) -> tuple[Path, Path, Path]:
    """Create a synthetic 10x Matrix Market triplet (matrix, features, barcodes)."""
    dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Barcodes file
    bc_name = "barcodes.tsv.gz" if gzip_barcodes else "barcodes.tsv"
    bc_path = dir_path / bc_name
    bc_content = "".join(f"{bc}\n" for bc in barcodes).encode("utf-8")
    if gzip_barcodes:
        with gzip.open(bc_path, "wb") as gz:
            gz.write(bc_content)
    else:
        bc_path.write_bytes(bc_content)

    # 2. Features / Genes file
    if legacy_genes:
        feat_name = "genes.tsv.gz" if gzip_features else "genes.tsv"
        feat_lines = [f"{f[0]}\t{f[1]}\n" for f in features]
    else:
        feat_name = "features.tsv.gz" if gzip_features else "features.tsv"
        feat_lines = [
            f"{f[0]}\t{f[1]}\t{f[2] if len(f) > 2 else 'Gene Expression'}\n" for f in features
        ]
    feat_path = dir_path / feat_name
    feat_content = "".join(feat_lines).encode("utf-8")
    if gzip_features:
        with gzip.open(feat_path, "wb") as gz:
            gz.write(feat_content)
    else:
        feat_path.write_bytes(feat_content)

    # 3. Matrix Market file
    mtx_name = "matrix.mtx.gz" if gzip_matrix else "matrix.mtx"
    mtx_path = dir_path / mtx_name

    header = header_override or "%%MatrixMarket matrix coordinate integer general"
    n_rows, n_cols, n_entries = dimensions_override or (
        len(features),
        len(barcodes),
        len(entries),
    )

    mtx_lines = [
        header + "\n",
        "% Generated synthetic 10x matrix market test fixture\n",
        f"{n_rows} {n_cols} {n_entries}\n",
    ]
    for r, c, val in entries:
        mtx_lines.append(f"{r} {c} {val}\n")

    mtx_content = "".join(mtx_lines).encode("utf-8")
    if gzip_matrix:
        with gzip.open(mtx_path, "wb") as gz:
            gz.write(mtx_content)
    else:
        mtx_path.write_bytes(mtx_content)

    return mtx_path, feat_path, bc_path


# =========================================================================
# PROTOCOL CONFORMANCE
# =========================================================================


def test_tenx_importers_protocol_conformance() -> None:
    """TenxH5Importer and TenxMtxImporter satisfy DatasetImporter runtime checkable protocol."""
    h5_importer = TenxH5Importer()
    assert isinstance(h5_importer, DatasetImporter)

    mtx_importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    assert isinstance(mtx_importer, DatasetImporter)


# =========================================================================
# TENX_H5 DETECTION & STRUCTURAL MALFORMED TESTS
# =========================================================================


def test_h5_detect_positive_and_negative(tmp_path: Path) -> None:
    """TenxH5Importer detect() handles valid HDF5, invalid format, and plaintext."""
    p_valid = create_synthetic_10x_h5(
        tmp_path / "valid.h5",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["AAAC-1"],
        data=np.array([10], dtype=np.int32),
        indices=np.array([0], dtype=np.int32),
        indptr=np.array([0, 1], dtype=np.int32),
    )

    importer = TenxH5Importer()

    # Valid TENX_H5
    src_valid = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p_valid),),
    )
    assert importer.detect(src_valid) is True

    # Plain text renamed .h5
    p_txt = tmp_path / "fake.h5"
    p_txt.write_text("not an hdf5 container")
    src_txt = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("f2", ImportFileRole.PRIMARY, p_txt),),
    )
    assert importer.detect(src_txt) is False

    # Wrong SourceFormat (e.g. H5AD)
    src_wrong_fmt = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("f3", ImportFileRole.PRIMARY, p_valid),),
    )
    assert importer.detect(src_wrong_fmt) is False


def test_h5_missing_matrix_group_rejected(tmp_path: Path) -> None:
    """Generic HDF5 missing /matrix group is rejected as MISSING_COMPONENT."""
    p = tmp_path / "empty_hdf5.h5"
    with h5py.File(p, "w") as h5:
        h5.create_group("other_group")

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.MISSING_COMPONENT
    assert "/matrix" in issues[0].message


def test_h5_missing_internal_dataset_rejected(tmp_path: Path) -> None:
    """HDF5 missing required dataset (e.g. barcodes) is rejected as MISSING_COMPONENT."""
    p = tmp_path / "missing_dset.h5"
    with h5py.File(p, "w") as h5:
        mat = h5.create_group("matrix")
        mat.create_dataset("data", data=np.array([1, 2]))
        mat.create_dataset("indices", data=np.array([0, 1]))
        mat.create_dataset("indptr", data=np.array([0, 2]))
        mat.create_dataset("shape", data=np.array([2, 1]))
        # barcodes is omitted!
        feat = mat.create_group("features")
        feat.create_dataset("id", data=np.array([b"g1", b"g2"], dtype=object))
        feat.create_dataset("name", data=np.array([b"g1", b"g2"], dtype=object))
        feat.create_dataset(
            "feature_type", data=np.array([b"Gene Expression", b"Gene Expression"], dtype=object)
        )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.MISSING_COMPONENT
    assert "barcodes" in issues[0].message


def test_h5_dimension_mismatch_rejected(tmp_path: Path) -> None:
    """Shape declared in HDF5 not matching actual array lengths fails as DIMENSION_MISMATCH."""
    p = create_synthetic_10x_h5(
        tmp_path / "dim_mismatch.h5",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["AAAC-1"],
        data=np.array([10], dtype=np.int32),
        indices=np.array([0], dtype=np.int32),
        indptr=np.array([0, 1], dtype=np.int32),
        custom_shape=(100, 50),  # Declares 100x50, but features is 1 and barcodes is 1
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()
    issues = importer.validate(src)
    assert len(issues) >= 1
    assert issues[0].code == ScientificIssueCode.DIMENSION_MISMATCH


def test_h5_corrupt_sparse_indptr_rejected(tmp_path: Path) -> None:
    """Non-monotonic or invalid indptr fails as MALFORMED_INPUT."""
    p = create_synthetic_10x_h5(
        tmp_path / "bad_indptr.h5",
        features=[("ENSG1", "TP53", "Gene Expression"), ("ENSG2", "BRCA1", "Gene Expression")],
        barcodes=["c1", "c2"],
        data=np.array([10, 20], dtype=np.int32),
        indices=np.array([0, 1], dtype=np.int32),
        indptr=np.array([0, 2, 1], dtype=np.int32),  # Decreasing from 2 to 1!
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.MALFORMED_INPUT


def test_h5_duplicate_barcodes_rejected(tmp_path: Path) -> None:
    """Duplicate barcodes in HDF5 rejected as DUPLICATE_CELL_IDS."""
    p = create_synthetic_10x_h5(
        tmp_path / "dup_bc.h5",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["CELL_DUP", "CELL_DUP"],
        data=np.array([1, 2], dtype=np.int32),
        indices=np.array([0, 0], dtype=np.int32),
        indptr=np.array([0, 1, 2], dtype=np.int32),
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.DUPLICATE_CELL_IDS


def test_h5_duplicate_stable_feature_ids_rejected(tmp_path: Path) -> None:
    """Duplicate stable feature IDs in HDF5 rejected as AMBIGUOUS_GENE_IDENTIFIERS."""
    p = create_synthetic_10x_h5(
        tmp_path / "dup_feat.h5",
        features=[
            ("ENSG1", "TP53", "Gene Expression"),
            ("ENSG1", "TP53_DUP", "Gene Expression"),
        ],
        barcodes=["c1"],
        data=np.array([1, 2], dtype=np.int32),
        indices=np.array([0, 1], dtype=np.int32),
        indptr=np.array([0, 2], dtype=np.int32),
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS


# =========================================================================
# TENX_MTX DETECTION & STRUCTURAL MALFORMED TESTS
# =========================================================================


def test_mtx_detect_positive_and_negative(tmp_path: Path) -> None:
    """TenxMtxImporter detect() handles valid triplet, missing roles, and gzip."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "mtx_valid",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["c1"],
        entries=[(1, 1, 10)],
    )

    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)

    # Valid triplet
    src_valid = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    assert importer.detect(src_valid) is True

    # Missing BARCODES role
    src_incomplete = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
        ),
    )
    assert importer.detect(src_incomplete) is False


def test_mtx_bad_header_rejected(tmp_path: Path) -> None:
    """Matrix Market file with invalid header fails as MALFORMED_INPUT."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "mtx_bad_hdr",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["c1"],
        entries=[(1, 1, 10)],
        header_override="NOT_MATRIX_MARKET header",
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_g3",
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.MALFORMED_INPUT


def test_mtx_dimensions_mismatch_rejected(tmp_path: Path) -> None:
    """Matrix Market declaring dimensions unequal to features/barcodes fails DIMENSION_MISMATCH."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "mtx_dim_mis",
        features=[("ENSG1", "TP53", "Gene Expression"), ("ENSG2", "BRCA1", "Gene Expression")],
        barcodes=["c1", "c2"],
        entries=[(1, 1, 10)],
        dimensions_override=(10, 20, 1),  # Declares 10 rows x 20 cols, but actual is 2 x 2
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_g3",
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) >= 1
    assert issues[0].code == ScientificIssueCode.DIMENSION_MISMATCH


def test_mtx_truncated_data_rejected(tmp_path: Path) -> None:
    """Matrix Market file truncated before declaring n_entries fails as MALFORMED_INPUT."""
    dir_p = tmp_path / "truncated_mtx"
    dir_p.mkdir(parents=True)
    bc_p = dir_p / "barcodes.tsv"
    bc_p.write_text("c1\n")
    feat_p = dir_p / "features.tsv"
    feat_p.write_text("ENSG1\tTP53\tGene Expression\n")
    mtx_p = dir_p / "matrix.mtx"
    # Declares 5 entries but only provides 1!
    mtx_p.write_text("%%MatrixMarket matrix coordinate integer general\n1 1 5\n1 1 10\n")

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_g3",
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.MALFORMED_INPUT
    assert "Truncated" in issues[0].message


def test_mtx_coordinate_out_of_bounds_rejected(tmp_path: Path) -> None:
    """Matrix Market coordinate referencing row > n_rows fails as MALFORMED_INPUT."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "mtx_oob",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["c1"],
        entries=[(99, 1, 10)],  # Row 99 > 1
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_g3",
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.MALFORMED_INPUT
    assert "out of bounds" in issues[0].message.lower()


# =========================================================================
# MODALITY TESTS (PURE GENE EXPRESSION VS MIXED/OTHER)
# =========================================================================


def test_h5_modality_mixed_rejected(tmp_path: Path) -> None:
    """10x H5 with Gene Expression + Antibody Capture is rejected as UNSUPPORTED_MODALITY."""
    p = create_synthetic_10x_h5(
        tmp_path / "mixed_mod.h5",
        features=[
            ("ENSG1", "CD3E", "Gene Expression"),
            ("AB1", "CD3_antibody", "Antibody Capture"),
        ],
        barcodes=["c1"],
        data=np.array([10, 50], dtype=np.int32),
        indices=np.array([0, 1], dtype=np.int32),
        indptr=np.array([0, 2], dtype=np.int32),
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.UNSUPPORTED_MODALITY
    assert "Mixed modality" in issues[0].message


def test_h5_modality_antibody_only_rejected(tmp_path: Path) -> None:
    """10x H5 with pure Antibody Capture is rejected as UNSUPPORTED_MODALITY."""
    p = create_synthetic_10x_h5(
        tmp_path / "ab_only.h5",
        features=[
            ("AB1", "CD3_antibody", "Antibody Capture"),
            ("AB2", "CD4_antibody", "Antibody Capture"),
        ],
        barcodes=["c1"],
        data=np.array([10, 20], dtype=np.int32),
        indices=np.array([0, 1], dtype=np.int32),
        indptr=np.array([0, 2], dtype=np.int32),
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.UNSUPPORTED_MODALITY


def test_mtx_modality_mixed_rejected(tmp_path: Path) -> None:
    """10x MTX with Gene Expression + Antibody Capture is rejected as UNSUPPORTED_MODALITY."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "mtx_mixed_mod",
        features=[
            ("ENSG1", "CD3E", "Gene Expression"),
            ("AB1", "CD3_antibody", "Antibody Capture"),
        ],
        barcodes=["c1"],
        entries=[(1, 1, 10), (2, 1, 20)],
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_g3",
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.UNSUPPORTED_MODALITY


# =========================================================================
# MODERN FEATURES VS LEGACY GENES (PLAIN & GZIP)
# =========================================================================


def test_mtx_modern_features_tsv_gzip(tmp_path: Path) -> None:
    """3-column features.tsv.gz is parsed correctly."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "mtx_gz",
        features=[
            ("ENSG00000141510", "TP53", "Gene Expression"),
            ("ENSG00000139618", "BRCA2", "Gene Expression"),
        ],
        barcodes=["c1", "c2"],
        entries=[(1, 1, 5), (2, 2, 8)],
        gzip_matrix=True,
        gzip_features=True,
        gzip_barcodes=True,
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_g3",
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 0

    inspect_res = importer.inspect(src)
    assert inspect_res.status == InspectionStatus.VALID
    assert inspect_res.cellCountEstimate == 2
    assert inspect_res.geneCountEstimate == 2
    assert inspect_res.detectedOrganism == Organism.HUMAN


def test_mtx_legacy_genes_tsv_gzip(tmp_path: Path) -> None:
    """2-column legacy genes.tsv.gz is parsed correctly as Gene Expression."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "legacy_genes",
        features=[
            ("ENSG00000141510", "TP53"),
            ("ENSG00000139618", "BRCA2"),
        ],
        barcodes=["c1", "c2"],
        entries=[(1, 1, 1), (2, 2, 2)],
        legacy_genes=True,
        gzip_features=True,
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_g3",
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 0

    inspect_res = importer.inspect(src)
    assert inspect_res.status == InspectionStatus.VALID
    assert inspect_res.detectedOrganism == Organism.HUMAN


def test_duplicate_gene_symbols_allowed_with_unique_stable_ids(tmp_path: Path) -> None:
    """Duplicate gene symbols (e.g. MARCH1) with distinct Ensembl IDs are valid biology."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "dup_symbols",
        features=[
            ("ENSG00000141510", "MARCH1", "Gene Expression"),
            ("ENSG00000139618", "MARCH1", "Gene Expression"),
        ],
        barcodes=["c1"],
        entries=[(1, 1, 10), (2, 1, 20)],
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_g3",
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 0


# =========================================================================
# BARCODES TESTS
# =========================================================================


def test_barcodes_suffix_preservation(tmp_path: Path) -> None:
    """Exact barcode values including suffix (-1) are preserved in canonical obs_names."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "bc_suffix",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["AAACATACACCAGT-1", "GGTCATACACCAGT-2"],
        entries=[(1, 1, 5), (1, 2, 8)],
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_g3",
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    out_p = tmp_path / "canonical_bc.h5ad"
    importer.canonicalize(
        source_set=src,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )

    c_adata = ad.read_h5ad(out_p)
    assert list(c_adata.obs_names) == ["AAACATACACCAGT-1", "GGTCATACACCAGT-2"]


# =========================================================================
# MANDATORY ORIENTATION TRANSPOSE GOLDEN TESTS (SECTIONS 44 & 45)
# =========================================================================


def test_orientation_transpose_golden_h5(tmp_path: Path) -> None:
    """H5 Golden Orientation: 3 genes x 2 cells source -> canonical 2 cells x 3 genes.

    Source (features x barcodes):
            c0  c1
        g0 [10,  0]
        g1 [ 0, 20]
        g2 [30, 40]

    Canonical (cells x genes):
            g0  g1  g2
        c0 [10,  0, 30]
        c1 [ 0, 20, 40]
    """
    p = tmp_path / "orient_test.h5"
    features = [
        ("ENSG00000000001", "GENE_0", "Gene Expression"),
        ("ENSG00000000002", "GENE_1", "Gene Expression"),
        ("ENSG00000000003", "GENE_2", "Gene Expression"),
    ]
    barcodes = ["CELL_0", "CELL_1"]

    # 10x CSC format:
    # Column 0 (CELL_0): g0=10, g2=30 -> indices [0, 2], data [10, 30]
    # Column 1 (CELL_1): g1=20, g2=40 -> indices [1, 2], data [20, 40]
    data = np.array([10, 30, 20, 40], dtype=np.int32)
    indices = np.array([0, 2, 1, 2], dtype=np.int32)
    indptr = np.array([0, 2, 4], dtype=np.int32)

    create_synthetic_10x_h5(p, features, barcodes, data, indices, indptr)

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()

    out_p = tmp_path / "canonical_h5_orient.h5ad"
    res = importer.canonicalize(
        source_set=src,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    assert res.cellCount == 2
    assert res.geneCount == 3

    c_adata = ad.read_h5ad(out_p)
    assert c_adata.shape == (2, 3), f"Expected (2 cells, 3 genes), got {c_adata.shape}"
    assert list(c_adata.obs_names) == ["CELL_0", "CELL_1"]
    assert list(c_adata.var_names) == [
        "ENSG00000000001",
        "ENSG00000000002",
        "ENSG00000000003",
    ]

    expected_dense = np.array(
        [
            [10, 0, 30],
            [0, 20, 40],
        ]
    )
    actual_dense = c_adata.X.toarray()
    assert np.array_equal(
        actual_dense, expected_dense
    ), f"Transposed matrix values mismatch:\nExpected:\n{expected_dense}\nActual:\n{actual_dense}"


def test_orientation_transpose_golden_mtx(tmp_path: Path) -> None:
    """MTX Golden Orientation: 3 genes x 2 cells source -> canonical 2 cells x 3 genes.

    Source entries: (1-based row=feature, 1-based col=barcode, count)
    (1, 1, 10), (3, 1, 30), (2, 2, 20), (3, 2, 40)
    """
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "mtx_orient",
        features=[
            ("ENSG00000000001", "GENE_0", "Gene Expression"),
            ("ENSG00000000002", "GENE_1", "Gene Expression"),
            ("ENSG00000000003", "GENE_2", "Gene Expression"),
        ],
        barcodes=["CELL_0", "CELL_1"],
        entries=[
            (1, 1, 10),
            (3, 1, 30),
            (2, 2, 20),
            (3, 2, 40),
        ],
    )

    src = ImportSourceSet(
        source_format=SourceFormat.TENX_MTX,
        files=(
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_g3",
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)

    out_p = tmp_path / "canonical_mtx_orient.h5ad"
    res = importer.canonicalize(
        source_set=src,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    assert res.cellCount == 2
    assert res.geneCount == 3

    c_adata = ad.read_h5ad(out_p)
    assert c_adata.shape == (2, 3)

    expected_dense = np.array(
        [
            [10, 0, 30],
            [0, 20, 40],
        ]
    )
    assert np.array_equal(c_adata.X.toarray(), expected_dense)


# =========================================================================
# ORGANISM RESOLUTION TESTS
# =========================================================================


def test_10x_organism_human_and_mouse_candidates(tmp_path: Path) -> None:
    """100% human Ensembl IDs resolves HUMAN; 100% mouse Ensembl resolves MOUSE."""
    # Human H5
    p_h = create_synthetic_10x_h5(
        tmp_path / "human.h5",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["c1"],
        data=np.array([1], dtype=np.int32),
        indices=np.array([0], dtype=np.int32),
        indptr=np.array([0, 1], dtype=np.int32),
    )
    importer_h5 = TenxH5Importer()
    res_h = importer_h5.inspect(
        ImportSourceSet(
            SourceFormat.TENX_H5,
            (ImportSourceFile("f1", ImportFileRole.PRIMARY, p_h),),
            dataset_version_id="dsv_h_test",
        )
    )
    assert res_h.detectedOrganism == Organism.HUMAN
    assert res_h.detectedGeneIdScheme == GeneIdScheme.ENSEMBL

    # Mouse MTX
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "mouse_mtx",
        features=[("ENSMUSG00000059552", "Trp53", "Gene Expression")],
        barcodes=["c1"],
        entries=[(1, 1, 1)],
    )
    importer_mtx = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    res_m = importer_mtx.inspect(
        ImportSourceSet(
            SourceFormat.TENX_MTX,
            (
                ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
                ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
                ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
            ),
            dataset_version_id="dsv_m_test",
        )
    )
    assert res_m.detectedOrganism == Organism.MOUSE
    assert res_m.detectedGeneIdScheme == GeneIdScheme.ENSEMBL


def test_10x_organism_mixed_species_conflict(tmp_path: Path) -> None:
    """Mixed Human (ENSG) and Mouse (ENSMUSG) IDs is a conflict (INVALID)."""
    p = create_synthetic_10x_h5(
        tmp_path / "mixed_species.h5",
        features=[
            ("ENSG00000141510", "TP53", "Gene Expression"),
            ("ENSMUSG00000059552", "Trp53", "Gene Expression"),
        ],
        barcodes=["c1"],
        data=np.array([1, 2], dtype=np.int32),
        indices=np.array([0, 1], dtype=np.int32),
        indptr=np.array([0, 2], dtype=np.int32),
    )
    importer = TenxH5Importer()
    res = importer.inspect(
        ImportSourceSet(
            SourceFormat.TENX_H5,
            (ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
            dataset_version_id="dsv_mixed_test",
        )
    )
    assert res.status == InspectionStatus.INVALID
    assert any(
        iss.code == ScientificIssueCode.UNRESOLVED_SPECIES
        and iss.severity == ScientificIssueSeverity.ERROR
        for iss in res.issues
    )


def test_10x_organism_symbols_only_needs_mapping(tmp_path: Path) -> None:
    """Symbol-only feature IDs flags NEEDS_MAPPING."""
    p = create_synthetic_10x_h5(
        tmp_path / "symbols.h5",
        features=[
            ("TP53", "TP53", "Gene Expression"),
            ("BRCA1", "BRCA1", "Gene Expression"),
        ],
        barcodes=["c1"],
        data=np.array([1, 2], dtype=np.int32),
        indices=np.array([0, 1], dtype=np.int32),
        indptr=np.array([0, 2], dtype=np.int32),
    )
    importer = TenxH5Importer()
    res = importer.inspect(
        ImportSourceSet(
            SourceFormat.TENX_H5,
            (ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
            dataset_version_id="dsv_sym_test",
        )
    )
    assert res.status == InspectionStatus.NEEDS_MAPPING
    assert res.requiresMapping is True
    assert res.detectedOrganism is None
    assert res.detectedGeneIdScheme == GeneIdScheme.SYMBOL


def test_10x_ensembl_version_suffix_preserved(tmp_path: Path) -> None:
    """Ensembl IDs with version suffixes (e.g. ENSG00000141510.7) are preserved in canonical var."""
    p = create_synthetic_10x_h5(
        tmp_path / "versioned.h5",
        features=[("ENSG00000141510.15", "TP53", "Gene Expression")],
        barcodes=["c1"],
        data=np.array([5], dtype=np.int32),
        indices=np.array([0], dtype=np.int32),
        indptr=np.array([0, 1], dtype=np.int32),
    )
    importer = TenxH5Importer()
    out_p = tmp_path / "canonical_vers.h5ad"
    importer.canonicalize(
        source_set=ImportSourceSet(
            SourceFormat.TENX_H5, (ImportSourceFile("f1", ImportFileRole.PRIMARY, p),)
        ),
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    c_adata = ad.read_h5ad(out_p)
    assert "ENSG00000141510.15" in c_adata.var_names


# =========================================================================
# RAW COUNT AMBIGUITY & VALIDATION TESTS
# =========================================================================


def test_10x_count_ambiguous_near_integral_rejected(tmp_path: Path) -> None:
    """Near-integral floating values fail closed as AMBIGUOUS_COUNT_SEMANTICS."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "ambig_mtx",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["c1"],
        entries=[(1, 1, 10.0 + 1e-15)],  # Near-integral ambiguous float
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS


def test_10x_count_fractional_rejected(tmp_path: Path) -> None:
    """Continuous/normalized fractional values fail closed as NON_COUNT_INPUT."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "frac_mtx",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["c1"],
        entries=[(1, 1, 2.5)],
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.NON_COUNT_INPUT


def test_10x_count_negative_rejected(tmp_path: Path) -> None:
    """Negative values fail closed as NON_COUNT_INPUT."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "neg_mtx",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["c1"],
        entries=[(1, 1, -5)],
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.NON_COUNT_INPUT


# =========================================================================
# SLOT HYGIENE & ROUND-TRIP VERIFICATION (SECTIONS 49, 50, 51)
# =========================================================================


def test_10x_slot_hygiene_and_round_trip_h5(tmp_path: Path) -> None:
    """H5 Canonical output conforms strictly to Policy B and passes Step D validator."""
    p = create_synthetic_10x_h5(
        tmp_path / "hygiene.h5",
        features=[
            ("ENSG00000141510", "TP53", "Gene Expression"),
            ("ENSG00000139618", "BRCA2", "Gene Expression"),
        ],
        barcodes=["CELL_A", "CELL_B"],
        data=np.array([1, 2], dtype=np.int32),
        indices=np.array([0, 1], dtype=np.int32),
        indptr=np.array([0, 1, 2], dtype=np.int32),
    )
    src = ImportSourceSet(
        SourceFormat.TENX_H5,
        (ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()
    out_p = tmp_path / "canonical_hygiene_h5.h5ad"

    res = importer.canonicalize(
        source_set=src,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        reference_build="GRCh38",
    )

    # 1. Read back and validate canonical AnnData
    c_adata = ad.read_h5ad(out_p)
    validate_canonical_anndata(
        c_adata,
        expected_organism=Organism.HUMAN,
        expected_gene_id_scheme=GeneIdScheme.ENSEMBL,
    )

    # 2. Strict slot hygiene assertions
    assert sp.isspmatrix_csr(c_adata.X)
    assert len(c_adata.layers) == 0
    assert len(c_adata.obsm) == 0
    assert len(c_adata.varm) == 0
    assert len(c_adata.obsp) == 0
    assert len(c_adata.varp) == 0

    # 3. Canonical uns metadata
    assert set(c_adata.uns.keys()) == {"organism", "gene_id_scheme", "reference_build"}
    assert c_adata.uns["organism"] == "HUMAN"
    assert c_adata.uns["gene_id_scheme"] == "ENSEMBL"
    assert c_adata.uns["reference_build"] == "GRCh38"

    # 4. Checksum matches written artifact bytes
    actual_sha = compute_file_sha256(out_p)
    assert res.canonicalChecksum == actual_sha


def test_10x_slot_hygiene_and_round_trip_mtx(tmp_path: Path) -> None:
    """MTX Canonical output conforms strictly to Policy B and passes Step D validator."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "hygiene_mtx",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["CELL_A"],
        entries=[(1, 1, 42)],
        gzip_matrix=True,
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    out_p = tmp_path / "canonical_hygiene_mtx.h5ad"

    res = importer.canonicalize(
        source_set=src,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )

    c_adata = ad.read_h5ad(out_p)
    validate_canonical_anndata(
        c_adata,
        expected_organism=Organism.HUMAN,
        expected_gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    assert sp.isspmatrix_csr(c_adata.X)
    assert len(c_adata.layers) == 0
    actual_sha = compute_file_sha256(out_p)
    assert res.canonicalChecksum == actual_sha


# =========================================================================
# SOURCE IMMUTABILITY TESTS (SECTION 52)
# =========================================================================


def test_10x_source_immutability_h5(tmp_path: Path) -> None:
    """Source H5 file SHA-256 before and after canonicalization is strictly identical."""
    p = create_synthetic_10x_h5(
        tmp_path / "immut.h5",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["c1"],
        data=np.array([10], dtype=np.int32),
        indices=np.array([0], dtype=np.int32),
        indptr=np.array([0, 1], dtype=np.int32),
    )
    sha_before = compute_file_sha256(p)

    src = ImportSourceSet(
        SourceFormat.TENX_H5,
        (ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()
    out_p = tmp_path / "canonical_immut.h5ad"
    importer.canonicalize(
        source_set=src,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )

    sha_after = compute_file_sha256(p)
    assert sha_before == sha_after


def test_10x_source_immutability_mtx(tmp_path: Path) -> None:
    """All 3 source MTX member files SHA-256 values are identical before and after."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "immut_mtx",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["c1"],
        entries=[(1, 1, 10)],
    )
    shas_before = {
        "m": compute_file_sha256(mtx_p),
        "f": compute_file_sha256(feat_p),
        "b": compute_file_sha256(bc_p),
    }

    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    out_p = tmp_path / "canonical_immut_mtx.h5ad"
    importer.canonicalize(
        source_set=src,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )

    shas_after = {
        "m": compute_file_sha256(mtx_p),
        "f": compute_file_sha256(feat_p),
        "b": compute_file_sha256(bc_p),
    }
    assert shas_before == shas_after


def test_source_path_equals_output_path_rejected(tmp_path: Path) -> None:
    """Directly outputting to source file path is rejected as security violation."""
    p = create_synthetic_10x_h5(
        tmp_path / "same.h5",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["c1"],
        data=np.array([10], dtype=np.int32),
        indices=np.array([0], dtype=np.int32),
        indptr=np.array([0, 1], dtype=np.int32),
    )
    src = ImportSourceSet(
        SourceFormat.TENX_H5,
        (ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
    )
    importer = TenxH5Importer()
    with pytest.raises(ImporterSecurityError, match="Source file immutability violation"):
        importer.canonicalize(
            source_set=src,
            output_path=str(p),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
        )


# =========================================================================
# ADDITIONAL MTX MALFORMED & ROBUSTNESS TESTS
# =========================================================================


def test_mtx_duplicate_barcodes_rejected(tmp_path: Path) -> None:
    """Duplicate barcodes in MTX barcodes file fails as DUPLICATE_CELL_IDS."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "mtx_dup_bc",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["CELL_DUP", "CELL_DUP"],
        entries=[(1, 1, 1), (1, 2, 2)],
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.DUPLICATE_CELL_IDS


def test_mtx_blank_barcodes_rejected(tmp_path: Path) -> None:
    """Blank or whitespace line in barcodes file fails as MALFORMED_INPUT."""
    dir_p = tmp_path / "mtx_blank_bc"
    dir_p.mkdir(parents=True)
    bc_p = dir_p / "barcodes.tsv"
    bc_p.write_text("c1\n   \nc2\n")
    feat_p = dir_p / "features.tsv"
    feat_p.write_text("ENSG1\tTP53\tGene Expression\n")
    mtx_p = dir_p / "matrix.mtx"
    mtx_p.write_text("%%MatrixMarket matrix coordinate integer general\n1 2 1\n1 1 5\n")

    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.MALFORMED_INPUT


def test_mtx_duplicate_stable_feature_ids_rejected(tmp_path: Path) -> None:
    """Duplicate stable feature IDs in features file fails as AMBIGUOUS_GENE_IDENTIFIERS."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "mtx_dup_feat",
        features=[
            ("ENSG1", "TP53", "Gene Expression"),
            ("ENSG1", "TP53_DUP", "Gene Expression"),
        ],
        barcodes=["c1"],
        entries=[(1, 1, 1), (2, 1, 2)],
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS


def test_mtx_blank_feature_id_rejected(tmp_path: Path) -> None:
    """Empty stable feature ID in features file fails as MALFORMED_INPUT."""
    dir_p = tmp_path / "mtx_blank_feat"
    dir_p.mkdir(parents=True)
    bc_p = dir_p / "barcodes.tsv"
    bc_p.write_text("c1\n")
    feat_p = dir_p / "features.tsv"
    feat_p.write_text("\tTP53\tGene Expression\n")
    mtx_p = dir_p / "matrix.mtx"
    mtx_p.write_text("%%MatrixMarket matrix coordinate integer general\n1 1 1\n1 1 5\n")

    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.MALFORMED_INPUT


def test_mtx_modality_antibody_only_rejected(tmp_path: Path) -> None:
    """10x MTX with pure Antibody Capture is rejected as UNSUPPORTED_MODALITY."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "mtx_ab_only",
        features=[
            ("AB1", "CD3", "Antibody Capture"),
            ("AB2", "CD4", "Antibody Capture"),
        ],
        barcodes=["c1"],
        entries=[(1, 1, 10), (2, 1, 20)],
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.UNSUPPORTED_MODALITY


def test_mtx_corrupt_gzip_rejected(tmp_path: Path) -> None:
    """Corrupted gzip member in MTX triplet fails as MALFORMED_INPUT."""
    dir_p = tmp_path / "corrupt_gz"
    dir_p.mkdir(parents=True)
    bc_p = dir_p / "barcodes.tsv"
    bc_p.write_text("c1\n")
    feat_p = dir_p / "features.tsv"
    feat_p.write_text("ENSG1\tTP53\tGene Expression\n")
    # Corrupt gzip bytes (starts with magic bytes but has corrupt body)
    mtx_p = dir_p / "matrix.mtx.gz"
    mtx_p.write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xffcorrupted_data_not_valid_deflate")

    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.MALFORMED_INPUT


def test_integral_floats_accepted(tmp_path: Path) -> None:
    """Integral floating counts (e.g. 10.0, 20.0) are accepted as integer counts without error."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "integral_float",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["c1"],
        entries=[(1, 1, 10.0)],
        header_override="%%MatrixMarket matrix coordinate real general",
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    out_p = tmp_path / "canonical_float.h5ad"
    res = importer.canonicalize(
        source_set=src,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    assert res.cellCount == 1
    assert res.geneCount == 1
    c_adata = ad.read_h5ad(out_p)
    assert c_adata.X[0, 0] == 10


def test_zero_heavy_sparse_matrix_preserved(tmp_path: Path) -> None:
    """A zero-heavy sparse matrix (e.g. 1000 genes x 500 cells with 2 entries) remains sparse."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "zero_heavy",
        features=[(f"ENSG{i:07d}", f"G_{i}", "Gene Expression") for i in range(1000)],
        barcodes=[f"BC_{j}" for j in range(500)],
        entries=[(10, 20, 1), (900, 450, 5)],
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    out_p = tmp_path / "canonical_zero_heavy.h5ad"
    importer.canonicalize(
        source_set=src,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    c_adata = ad.read_h5ad(out_p)
    assert c_adata.shape == (500, 1000)
    assert sp.isspmatrix_csr(c_adata.X)
    assert c_adata.X.nnz == 2


# =========================================================================
# STEP F CORRECTION TESTS: CONFIG, DECOMPRESSION, SAMPLE AUTHORITY, IDENTITY
# =========================================================================


def test_tenx_config_fail_closed_validation() -> None:
    """TenxImporterConfig requires explicit, valid DecompressionLimits with zero defaults."""
    # 1. Missing decompression_limits raises TypeError
    with pytest.raises(TypeError):
        TenxImporterConfig()  # type: ignore[call-arg]

    # 2. None decompression_limits raises TypeError
    with pytest.raises(TypeError, match="must be an instance of DecompressionLimits"):
        TenxImporterConfig(decompression_limits=None)  # type: ignore[arg-type]

    # 3. Invalid non-DecompressionLimits type raises TypeError
    with pytest.raises(TypeError, match="must be an instance of DecompressionLimits"):
        TenxImporterConfig(decompression_limits="not_a_limit")  # type: ignore[arg-type]

    # 4. Invalid limits within DecompressionLimits fail closed via Step D validator
    with pytest.raises(ImporterSecurityError, match="must be an integer strictly greater than 0"):
        DecompressionLimits(max_uncompressed_bytes=0, max_compression_ratio=10.0)

    with pytest.raises(ImporterSecurityError, match="must be a float strictly greater than 0.0"):
        DecompressionLimits(max_uncompressed_bytes=1000, max_compression_ratio=0.0)

    # 5. TenxMtxImporter rejects non-TenxImporterConfig
    with pytest.raises(TypeError, match="config must be an instance of TenxImporterConfig"):
        TenxMtxImporter(None)  # type: ignore[arg-type]


def test_10x_mtx_decompression_limits_matrix_fails_resource_limit(tmp_path: Path) -> None:
    """matrix.mtx.gz exceeding max_uncompressed_bytes fails closed with RESOURCE_LIMIT."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "small_matrix_limit",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["BC1"],
        entries=[(1, 1, 5)],
        gzip_matrix=True,
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_test",
    )
    small_limits = DecompressionLimits(max_uncompressed_bytes=10, max_compression_ratio=200.0)
    importer = TenxMtxImporter(TenxImporterConfig(decompression_limits=small_limits))

    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.RESOURCE_LIMIT
    assert "Decompression limit exceeded" in issues[0].message


def test_10x_mtx_decompression_limits_features_fails_resource_limit(tmp_path: Path) -> None:
    """features.tsv.gz exceeding max_uncompressed_bytes fails closed with RESOURCE_LIMIT."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "small_feat_limit",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["BC1"],
        entries=[(1, 1, 5)],
        gzip_features=True,
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_test",
    )
    small_limits = DecompressionLimits(max_uncompressed_bytes=10, max_compression_ratio=200.0)
    importer = TenxMtxImporter(TenxImporterConfig(decompression_limits=small_limits))

    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.RESOURCE_LIMIT
    assert "Decompression limit exceeded" in issues[0].message


def test_10x_mtx_decompression_limits_legacy_genes_fails_resource_limit(tmp_path: Path) -> None:
    """legacy genes.tsv.gz exceeding max_uncompressed_bytes fails closed with RESOURCE_LIMIT."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "small_genes_limit",
        features=[("ENSG00000141510", "TP53")],
        barcodes=["BC1"],
        entries=[(1, 1, 5)],
        legacy_genes=True,
        gzip_features=True,
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_test",
    )
    small_limits = DecompressionLimits(max_uncompressed_bytes=10, max_compression_ratio=200.0)
    importer = TenxMtxImporter(TenxImporterConfig(decompression_limits=small_limits))

    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.RESOURCE_LIMIT
    assert "Decompression limit exceeded" in issues[0].message


def test_10x_mtx_decompression_limits_barcodes_fails_resource_limit(tmp_path: Path) -> None:
    """barcodes.tsv.gz exceeding max_uncompressed_bytes fails closed with RESOURCE_LIMIT."""
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "small_bc_limit",
        features=[("ENSG1", "TP53", "Gene Expression")],
        barcodes=["CELL_BARCODE_LONG_STRING_001"],
        entries=[(1, 1, 5)],
        gzip_barcodes=True,
    )
    src = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_test",
    )
    small_limits = DecompressionLimits(max_uncompressed_bytes=10, max_compression_ratio=200.0)
    importer = TenxMtxImporter(TenxImporterConfig(decompression_limits=small_limits))

    issues = importer.validate(src)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.RESOURCE_LIMIT
    assert "Decompression limit exceeded" in issues[0].message


def test_10x_canonicalize_no_sample_mapping_zero_obs_columns(tmp_path: Path) -> None:
    """Case A: When no sample mapping is supplied, canonical obs has shape (N, 0) and zero synthetic columns."""
    # 1. TenxMtxImporter
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "case_a_mtx",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["CELL_01", "CELL_02"],
        entries=[(1, 1, 10), (1, 2, 20)],
    )
    src_mtx = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_case_a",
    )
    importer_mtx = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    out_mtx = tmp_path / "canonical_case_a_mtx.h5ad"
    importer_mtx.canonicalize(
        source_set=src_mtx,
        output_path=str(out_mtx),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=None,
    )
    adata_mtx = ad.read_h5ad(out_mtx)
    assert adata_mtx.obs.shape == (2, 0)
    assert "sample_key" not in adata_mtx.obs.columns
    assert list(adata_mtx.obs_names) == ["CELL_01", "CELL_02"]

    # 2. TenxH5Importer
    p_h5 = create_synthetic_10x_h5(
        tmp_path / "case_a_h5.h5",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["CELL_01", "CELL_02"],
        data=np.array([10, 20], dtype=np.int32),
        indices=np.array([0, 0], dtype=np.int32),
        indptr=np.array([0, 1, 2], dtype=np.int32),
    )
    src_h5 = ImportSourceSet(
        SourceFormat.TENX_H5,
        (ImportSourceFile("h1", ImportFileRole.PRIMARY, p_h5),),
        dataset_version_id="dsv_case_a_h5",
    )
    importer_h5 = TenxH5Importer()
    out_h5 = tmp_path / "canonical_case_a_h5.h5ad"
    importer_h5.canonicalize(
        source_set=src_h5,
        output_path=str(out_h5),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=None,
    )
    adata_h5 = ad.read_h5ad(out_h5)
    assert adata_h5.obs.shape == (2, 0)
    assert "sample_key" not in adata_h5.obs.columns
    assert list(adata_h5.obs_names) == ["CELL_01", "CELL_02"]


def test_10x_sample_mapping_inapplicable_fails_closed(tmp_path: Path) -> None:
    """10x sources have no source sample dimension; barcode is not sample identity.
    Passing sampleMappings fails closed with MISSING_REQUIRED_METADATA.
    """
    # 1. TenxMtxImporter
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "inapp_mtx",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["CELL_01", "CELL_02"],
        entries=[(1, 1, 10), (1, 2, 20)],
    )
    src_mtx = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_inapp_mtx",
    )
    mapping = MetadataMappingSubmission(
        datasetVersionId="dsv_inapp_mtx",
        organism=Organism.HUMAN,
        geneIdScheme=GeneIdScheme.ENSEMBL,
        sampleMappings=[
            SampleMappingEntry(sourceValue="CELL_01", sampleKey="Donor_Treated"),
            SampleMappingEntry(sourceValue="CELL_02", sampleKey="Donor_Control"),
        ],
    )
    importer_mtx = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    out_mtx = tmp_path / "canonical_inapp_mtx.h5ad"
    with pytest.raises(ImporterSecurityError) as exc_info_mtx:
        importer_mtx.canonicalize(
            source_set=src_mtx,
            output_path=str(out_mtx),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
    assert exc_info_mtx.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA

    # 2. TenxH5Importer
    p_h5 = create_synthetic_10x_h5(
        tmp_path / "inapp_h5.h5",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["CELL_01", "CELL_02"],
        data=np.array([10, 20], dtype=np.int32),
        indices=np.array([0, 0], dtype=np.int32),
        indptr=np.array([0, 1, 2], dtype=np.int32),
    )
    src_h5 = ImportSourceSet(
        SourceFormat.TENX_H5,
        (ImportSourceFile("h1", ImportFileRole.PRIMARY, p_h5),),
        dataset_version_id="dsv_inapp_h5",
    )
    importer_h5 = TenxH5Importer()
    out_h5 = tmp_path / "canonical_inapp_h5.h5ad"
    with pytest.raises(ImporterSecurityError) as exc_info_h5:
        importer_h5.canonicalize(
            source_set=src_h5,
            output_path=str(out_h5),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
    assert exc_info_h5.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


def test_10x_single_mapping_not_global_fails_closed(tmp_path: Path) -> None:
    """A single SampleMappingEntry does NOT infer a library-level sample assignment.
    Plain 10x sources have no source sample dimension, so importer fails closed.
    """
    p_h5 = create_synthetic_10x_h5(
        tmp_path / "single_not_global.h5",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["CELL_01", "CELL_02"],
        data=np.array([10, 20], dtype=np.int32),
        indices=np.array([0, 0], dtype=np.int32),
        indptr=np.array([0, 1, 2], dtype=np.int32),
    )
    src_h5 = ImportSourceSet(
        SourceFormat.TENX_H5,
        (ImportSourceFile("h1", ImportFileRole.PRIMARY, p_h5),),
        dataset_version_id="dsv_single_test",
    )
    mapping = MetadataMappingSubmission(
        datasetVersionId="dsv_single_test",
        organism=Organism.HUMAN,
        geneIdScheme=GeneIdScheme.ENSEMBL,
        sampleMappings=[
            SampleMappingEntry(sourceValue="library_1", sampleKey="Patient_Alpha"),
        ],
    )
    importer = TenxH5Importer()
    out_h5 = tmp_path / "canonical_single_not_global.h5ad"
    with pytest.raises(ImporterSecurityError) as exc_info:
        importer.canonicalize(
            source_set=src_h5,
            output_path=str(out_h5),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
    assert exc_info.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


def test_10x_sample_column_inapplicable_fails_closed(tmp_path: Path) -> None:
    """10x sources do not have cell metadata columns; sampleColumn request fails closed."""
    p_h5 = create_synthetic_10x_h5(
        tmp_path / "sample_col_h5.h5",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["CELL_01", "CELL_02"],
        data=np.array([10, 20], dtype=np.int32),
        indices=np.array([0, 0], dtype=np.int32),
        indptr=np.array([0, 1, 2], dtype=np.int32),
    )
    src = ImportSourceSet(
        SourceFormat.TENX_H5,
        (ImportSourceFile("h1", ImportFileRole.PRIMARY, p_h5),),
        dataset_version_id="dsv_col_test",
    )
    mapping = MetadataMappingSubmission(
        datasetVersionId="dsv_col_test",
        organism=Organism.HUMAN,
        geneIdScheme=GeneIdScheme.ENSEMBL,
        sampleColumn="donor",
    )
    with pytest.raises(ImporterSecurityError) as exc_info:
        TenxH5Importer().canonicalize(
            source_set=src,
            output_path=str(tmp_path / "out_col.h5ad"),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
            mapping=mapping,
        )
    assert exc_info.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


def test_10x_mapping_without_cell_mappings_succeeds(tmp_path: Path) -> None:
    """MetadataMappingSubmission with confirmed organism and geneIdScheme but NO inapplicable
    cell mappings canonicalizes cleanly with shape (n_cells, 0) and no sample_key.
    """
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "clean_meta_mtx",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["CELL_01", "CELL_02"],
        entries=[(1, 1, 10), (1, 2, 20)],
    )
    src_mtx = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id="dsv_clean_meta",
    )
    mapping = MetadataMappingSubmission(
        datasetVersionId="dsv_clean_meta",
        organism=Organism.HUMAN,
        geneIdScheme=GeneIdScheme.ENSEMBL,
        referenceBuild="GRCh38",
    )
    importer = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    out_mtx = tmp_path / "canonical_clean_meta.h5ad"
    importer.canonicalize(
        source_set=src_mtx,
        output_path=str(out_mtx),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )
    adata = ad.read_h5ad(out_mtx)
    assert adata.obs.shape == (2, 0)
    assert "sample_key" not in adata.obs.columns
    assert list(adata.obs_names) == ["CELL_01", "CELL_02"]


def test_10x_inspect_missing_dataset_version_id_fails_controlled(tmp_path: Path) -> None:
    """inspect() for both 10x importers fails closed if dataset_version_id is missing or whitespace."""
    # 1. TenxH5
    p_h5 = create_synthetic_10x_h5(
        tmp_path / "id_auth.h5",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["c1"],
        data=np.array([1], dtype=np.int32),
        indices=np.array([0], dtype=np.int32),
        indptr=np.array([0, 1], dtype=np.int32),
    )
    importer_h5 = TenxH5Importer()
    src_h5_none = ImportSourceSet(
        SourceFormat.TENX_H5,
        (ImportSourceFile("f1", ImportFileRole.PRIMARY, p_h5),),
        dataset_version_id=None,
    )
    with pytest.raises(ImporterSecurityError) as exc_info:
        importer_h5.inspect(src_h5_none)
    assert exc_info.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA

    src_h5_empty = ImportSourceSet(
        SourceFormat.TENX_H5,
        (ImportSourceFile("f1", ImportFileRole.PRIMARY, p_h5),),
        dataset_version_id="   ",
    )
    with pytest.raises(ImporterSecurityError) as exc_info:
        importer_h5.inspect(src_h5_empty)
    assert exc_info.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA

    # 2. TenxMtx
    mtx_p, feat_p, bc_p = create_synthetic_10x_mtx(
        tmp_path / "id_auth_mtx",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["c1"],
        entries=[(1, 1, 1)],
    )
    importer_mtx = TenxMtxImporter(DEFAULT_TEST_CONFIG)
    src_mtx_none = ImportSourceSet(
        SourceFormat.TENX_MTX,
        (
            ImportSourceFile("m1", ImportFileRole.MATRIX, mtx_p),
            ImportSourceFile("f1", ImportFileRole.FEATURES, feat_p),
            ImportSourceFile("b1", ImportFileRole.BARCODES, bc_p),
        ),
        dataset_version_id=None,
    )
    with pytest.raises(ImporterSecurityError) as exc_info:
        importer_mtx.inspect(src_mtx_none)
    assert exc_info.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


def test_10x_inspect_dataset_version_id_distinct_from_file_id(tmp_path: Path) -> None:
    """inspect() reflects exact dataset_version_id and never falls back to file ID."""
    p_h5 = create_synthetic_10x_h5(
        tmp_path / "distinct_id.h5",
        features=[("ENSG00000141510", "TP53", "Gene Expression")],
        barcodes=["c1"],
        data=np.array([1], dtype=np.int32),
        indices=np.array([0], dtype=np.int32),
        indptr=np.array([0, 1], dtype=np.int32),
    )
    src = ImportSourceSet(
        SourceFormat.TENX_H5,
        (ImportSourceFile("dsf_primary_file_xyz", ImportFileRole.PRIMARY, p_h5),),
        dataset_version_id="dsv_trusted_version_abc",
    )
    res = TenxH5Importer().inspect(src)
    assert res.datasetVersionId == "dsv_trusted_version_abc"
    assert res.datasetVersionId != "dsf_primary_file_xyz"
