"""Unit and golden fixture tests for H5ADImporter (Phase 3 Step E).

Covers:
- Protocol conformance (satisfies DatasetImporter).
- detect() on valid H5AD, plain text, invalid format.
- Golden fixtures G1 to G10:
    G1: valid HUMAN Ensembl, raw counts in X.
    G2: valid MOUSE Ensembl, raw counts in X.
    G3: X normalized, layers['counts'] valid raw counts.
    G4: X normalized, raw.X valid and aligned.
    G5: normalized-only, no raw counts -> INVALID / NON_COUNT_INPUT.
    G6: duplicate obs_names -> INVALID / DUPLICATE_CELL_IDS.
    G7: stable Ensembl var_names + duplicate descriptive gene symbols -> VALID.
    G8: duplicate symbol-only var_names with no alternate stable ID -> INVALID / AMBIGUOUS_GENE_IDENTIFIERS.
    G9: unresolved organism -> NEEDS_MAPPING / UNRESOLVED_SPECIES.
    G10: conflicting human metadata vs mouse IDs -> INVALID / UNRESOLVED_SPECIES.
- adata.raw alignment tests (exact, superset subsetting, missing features).
- counts-layer tests (counts layer priority over transformed X, conflicting counts).
- Metadata & sample mapping tests.
- Slot hygiene tests (obsm, varm, uns, layers purged).
- Round-trip canonical verification and checksum matching.
- Source immutability test.
- Sparse preservation without densification.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
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
from cellweave_science.importers.canonical import validate_canonical_anndata
from cellweave_science.importers.checksum import compute_file_sha256
from cellweave_science.importers.errors import ImporterSecurityError
from cellweave_science.importers.h5ad import H5ADImporter
from cellweave_science.importers.sources import ImportSourceFile, ImportSourceSet


def test_h5ad_importer_protocol_conformance() -> None:
    """H5ADImporter conforms to DatasetImporter runtime checkable protocol."""
    importer = H5ADImporter()
    from cellweave_science.importers import DatasetImporter

    assert isinstance(importer, DatasetImporter)


def test_detect_valid_h5ad(tmp_path: Path) -> None:
    """detect() returns True for valid single PRIMARY H5AD container."""
    p = tmp_path / "dataset.h5ad"
    adata = ad.AnnData(X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32))
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_1", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    assert importer.detect(source_set) is True


def test_detect_rejects_non_hdf5(tmp_path: Path) -> None:
    """detect() returns False for plaintext file named .h5ad."""
    fake_h5ad = tmp_path / "fake.h5ad"
    fake_h5ad.write_text("plain text not hdf5", encoding="utf-8")

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_1", ImportFileRole.PRIMARY, fake_h5ad),),
    )
    importer = H5ADImporter()
    assert importer.detect(source_set) is False


def test_detect_rejects_wrong_source_format(tmp_path: Path) -> None:
    """detect() returns False if source_format is not H5AD."""
    p = tmp_path / "dataset.h5ad"
    adata = ad.AnnData(X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32))
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.TENX_H5,
        files=(ImportSourceFile("dsf_1", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    assert importer.detect(source_set) is False


# =========================================================================
# GOLDEN FIXTURE TESTS (G1 - G10)
# =========================================================================


def test_golden_g1_human_ensembl_counts_in_x(tmp_path: Path) -> None:
    """G1: Valid HUMAN Ensembl dataset with raw counts in X."""
    p = tmp_path / "g1.h5ad"
    x = sp.csr_matrix([[10, 0, 5], [0, 20, 1]], dtype=np.int32)
    obs = pd.DataFrame(index=["cell_1", "cell_2"])
    var = pd.DataFrame(index=["ENSG00000141510", "ENSG00000139618", "ENSG00000157764"])
    var["gene_symbol"] = ["TP53", "BRCA2", "BRAF"]
    adata = ad.AnnData(X=x, obs=obs, var=var, uns={"reference_build": "GRCh38"})
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_g1", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_g1",
    )
    importer = H5ADImporter()

    # 1. Validation
    issues = importer.validate(source_set)
    assert len(issues) == 0

    # 2. Inspection
    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.VALID
    assert inspect_res.detectedOrganism == Organism.HUMAN
    assert inspect_res.detectedGeneIdScheme == GeneIdScheme.ENSEMBL
    assert inspect_res.cellCountEstimate == 2
    assert inspect_res.geneCountEstimate == 3
    assert inspect_res.referenceBuildCandidate == "GRCh38"

    # 3. Canonicalize
    out_p = tmp_path / "canonical_g1.h5ad"
    canon_res = importer.canonicalize(
        source_set=source_set,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        reference_build="GRCh38",
    )
    assert canon_res.cellCount == 2
    assert canon_res.geneCount == 3
    assert canon_res.canonicalChecksum == compute_file_sha256(out_p)

    # 4. Canonical validation check
    c_adata = ad.read_h5ad(out_p)
    validate_canonical_anndata(
        c_adata, expected_organism=Organism.HUMAN, expected_gene_id_scheme=GeneIdScheme.ENSEMBL
    )


def test_golden_g2_mouse_ensembl_counts_in_x(tmp_path: Path) -> None:
    """G2: Valid MOUSE Ensembl dataset with raw counts in X."""
    p = tmp_path / "g2.h5ad"
    x = sp.csr_matrix([[5, 1], [0, 8]], dtype=np.int32)
    obs = pd.DataFrame(index=["cell_m1", "cell_m2"])
    var = pd.DataFrame(index=["ENSMUSG00000001234", "ENSMUSG00000005678"])
    adata = ad.AnnData(
        X=x, obs=obs, var=var, uns={"organism": "MOUSE", "reference_build": "GRCm39"}
    )
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_g2", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_g2",
    )
    importer = H5ADImporter()

    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.VALID
    assert inspect_res.detectedOrganism == Organism.MOUSE
    assert inspect_res.detectedGeneIdScheme == GeneIdScheme.ENSEMBL

    out_p = tmp_path / "canonical_g2.h5ad"
    canon_res = importer.canonicalize(
        source_set=source_set,
        output_path=str(out_p),
        organism=Organism.MOUSE,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        reference_build="GRCm39",
    )
    assert canon_res.organism == Organism.MOUSE


def test_golden_g3_x_normalized_counts_in_layer(tmp_path: Path) -> None:
    """G3: X is normalized float, but layers['counts'] contains valid raw counts."""
    p = tmp_path / "g3.h5ad"
    x_norm = np.array([[1.5, 0.0], [0.8, 2.3]], dtype=np.float32)
    raw_counts = sp.csr_matrix([[10, 0], [4, 15]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSG00000141510", "ENSG00000139618"])
    adata = ad.AnnData(X=x_norm, obs=obs, var=var, layers={"counts": raw_counts})
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_g3", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()

    # Validation should pass because layers['counts'] provides raw counts
    assert len(importer.validate(source_set)) == 0

    out_p = tmp_path / "canonical_g3.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    c_adata = ad.read_h5ad(out_p)
    # Policy B: canonical X must contain the raw counts from the layer
    assert (c_adata.X.toarray() == raw_counts.toarray()).all()
    assert "counts" not in c_adata.layers


def test_golden_g4_x_normalized_counts_in_raw(tmp_path: Path) -> None:
    """G4: X is normalized float, but raw.X contains valid aligned raw counts."""
    p = tmp_path / "g4.h5ad"
    x_norm = np.array([[1.5, 0.0], [0.8, 2.3]], dtype=np.float32)
    raw_counts = sp.csr_matrix([[10, 0], [4, 15]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSG00000141510", "ENSG00000139618"])

    adata_raw = ad.AnnData(X=raw_counts, obs=obs, var=var)
    adata = ad.AnnData(X=x_norm, obs=obs, var=var)
    adata.raw = adata_raw
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_g4", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    assert len(importer.validate(source_set)) == 0

    out_p = tmp_path / "canonical_g4.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    c_adata = ad.read_h5ad(out_p)
    assert (c_adata.X.toarray() == raw_counts.toarray()).all()


def test_golden_g5_normalized_only_rejected(tmp_path: Path) -> None:
    """G5: Normalized-only H5AD with no raw counts is rejected as INVALID / NON_COUNT_INPUT."""
    p = tmp_path / "g5.h5ad"
    x_norm = np.array([[1.5, 0.0], [0.8, 2.3]], dtype=np.float32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSG00000141510", "ENSG00000139618"])
    adata = ad.AnnData(X=x_norm, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_g5", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_g5",
    )
    importer = H5ADImporter()

    issues = importer.validate(source_set)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.NON_COUNT_INPUT

    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.INVALID


def test_golden_g6_duplicate_obs_names_rejected(tmp_path: Path) -> None:
    """G6: Duplicate obs_names is rejected as DUPLICATE_CELL_IDS without auto-renaming."""
    p = tmp_path / "g6.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    with pytest.warns(UserWarning):
        adata = ad.AnnData(
            X=x,
            obs=pd.DataFrame(index=["cell_dup", "cell_dup"]),
            var=pd.DataFrame(index=["ENSG00000141510", "ENSG00000139618"]),
        )
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_g6", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()

    issues = importer.validate(source_set)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.DUPLICATE_CELL_IDS


def test_golden_g7_stable_var_names_duplicate_symbols(tmp_path: Path) -> None:
    """G7: Unique Ensembl var_names with duplicate descriptive gene_symbol column is VALID."""
    p = tmp_path / "g7.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    # 2 distinct Ensembl IDs, but both have symbol 'MARCH1' (valid biology)
    var = pd.DataFrame(index=["ENSG00000141510", "ENSG00000139618"])
    var["gene_symbol"] = ["MARCH1", "MARCH1"]
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_g7", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    assert len(importer.validate(source_set)) == 0

    out_p = tmp_path / "canonical_g7.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    c_adata = ad.read_h5ad(out_p)
    assert list(c_adata.var_names) == ["ENSG00000141510", "ENSG00000139618"]
    assert list(c_adata.var["gene_symbol"]) == ["MARCH1", "MARCH1"]


def test_golden_g8_duplicate_symbol_only_var_names(tmp_path: Path) -> None:
    """G8: Duplicate symbol-only var_names with no alternate stable ID column is rejected."""
    p = tmp_path / "g8.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    with pytest.warns(UserWarning):
        adata = ad.AnnData(
            X=x,
            obs=obs,
            var=pd.DataFrame(index=["TP53", "TP53"]),  # Duplicate symbols, no Ensembl column
        )
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_g8", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()

    issues = importer.validate(source_set)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.AMBIGUOUS_GENE_IDENTIFIERS


def test_golden_g9_unresolved_organism(tmp_path: Path) -> None:
    """G9: Symbol-only gene IDs with no explicit metadata -> NEEDS_MAPPING / UNRESOLVED_SPECIES."""
    p = tmp_path / "g9.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["TP53", "EGFR"])  # Unique symbols, but species ambiguous
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_g9", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_g9",
    )
    importer = H5ADImporter()

    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.NEEDS_MAPPING
    assert inspect_res.requiresMapping is True
    assert inspect_res.detectedOrganism is None
    assert any(iss.code == ScientificIssueCode.UNRESOLVED_SPECIES for iss in inspect_res.issues)


def test_golden_g10_conflicting_organism_metadata(tmp_path: Path) -> None:
    """G10: Explicit HUMAN uns metadata conflicts with mouse ENSMUSG IDs -> INVALID."""
    p = tmp_path / "g10.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSMUSG00000001234", "ENSMUSG00000005678"])
    adata = ad.AnnData(X=x, obs=obs, var=var, uns={"organism": "HUMAN"})
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_g10", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_g10",
    )
    importer = H5ADImporter()

    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.INVALID
    assert any(iss.code == ScientificIssueCode.UNRESOLVED_SPECIES for iss in inspect_res.issues)


# =========================================================================
# adata.raw ALIGNMENT & LAYER TESTS
# =========================================================================


def test_raw_superset_feature_subsetting(tmp_path: Path) -> None:
    """adata.raw is a superset of current features: subsetting is performed deterministically."""
    p = tmp_path / "raw_superset.h5ad"
    raw_counts = sp.csr_matrix([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    raw_var = pd.DataFrame(index=["ENSG1", "ENSG2", "ENSG3"])

    adata_raw = ad.AnnData(X=raw_counts, obs=obs, var=raw_var)

    # Current adata only has ENSG3 and ENSG1 (subset and reordered)
    x_norm = np.array([[3.2, 1.1], [6.4, 4.8]], dtype=np.float32)
    cur_var = pd.DataFrame(index=["ENSG3", "ENSG1"])
    adata = ad.AnnData(X=x_norm, obs=obs, var=cur_var)
    adata.raw = adata_raw
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_raw", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()

    out_p = tmp_path / "canon_raw.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    c_adata = ad.read_h5ad(out_p)
    # Expected reordered subset: col 0 is ENSG3 (30, 60), col 1 is ENSG1 (10, 40)
    expected_x = np.array([[30, 10], [60, 40]])
    assert (c_adata.X.toarray() == expected_x).all()


def test_raw_missing_features_rejected(tmp_path: Path) -> None:
    """adata.raw missing features present in current AnnData is rejected as raw source."""
    p = tmp_path / "raw_missing.h5ad"
    raw_counts = sp.csr_matrix([[10], [40]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    raw_var = pd.DataFrame(index=["ENSG1"])
    adata_raw = ad.AnnData(X=raw_counts, obs=obs, var=raw_var)

    # Current has ENSG1 and ENSG2, but raw only has ENSG1
    x_norm = np.array([[1.1, 2.2], [4.3, 5.4]], dtype=np.float32)
    cur_var = pd.DataFrame(index=["ENSG1", "ENSG2"])
    adata = ad.AnnData(X=x_norm, obs=obs, var=cur_var)
    adata.raw = adata_raw
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_miss", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    # Since X is normalized and raw is missing features, validation fails
    issues = importer.validate(source_set)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.NON_COUNT_INPUT


def test_conflicting_raw_count_sources_rejected(tmp_path: Path) -> None:
    """X and layers['counts'] both contain valid counts but differ -> AMBIGUOUS_COUNT_SEMANTICS."""
    p = tmp_path / "conflicting_counts.h5ad"
    x_counts = sp.csr_matrix([[10, 20], [30, 40]], dtype=np.int32)
    layer_counts = sp.csr_matrix([[10, 25], [30, 40]], dtype=np.int32)  # Differs at [0, 1]
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSG1", "ENSG2"])

    adata = ad.AnnData(X=x_counts, obs=obs, var=var, layers={"counts": layer_counts})
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_conf", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    issues = importer.validate(source_set)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS


# =========================================================================
# SLOT HYGIENE & METADATA TESTS
# =========================================================================


def test_slot_hygiene_sanitization(tmp_path: Path) -> None:
    """Source analysis artifacts (X_umap, PCA, leiden, layers) are strictly purged in canonical output."""
    p = tmp_path / "dirty_source.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"], data={"cluster": ["1", "2"]})
    var = pd.DataFrame(index=["ENSG1", "ENSG2"])

    adata = ad.AnnData(
        X=x,
        obs=obs,
        var=var,
        obsm={
            "X_umap": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "X_pca": np.array([[1.0, 2.0], [3.0, 4.0]]),
        },
        uns={"neighbors": {"params": {}}, "leiden": ["1", "2"]},
        layers={"normalized": np.array([[0.1, 0.2], [0.3, 0.4]])},
    )
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_dirty", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()

    out_p = tmp_path / "clean_canonical.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    c_adata = ad.read_h5ad(out_p)

    # Analysis slots must be empty
    assert len(c_adata.obsm) == 0
    assert len(c_adata.varm) == 0
    assert len(c_adata.layers) == 0
    assert "neighbors" not in c_adata.uns
    assert "leiden" not in c_adata.uns


def test_metadata_mapping_application(tmp_path: Path) -> None:
    """Explicit MetadataMappingSubmission correctly populates sampleKey, condition, batch."""
    p = tmp_path / "meta_source.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(
        index=["c1", "c2"],
        data={
            "donor": ["patient_A", "patient_B"],
            "disease_state": ["control", "treated"],
            "run_batch": ["b1", "b2"],
        },
    )
    var = pd.DataFrame(index=["ENSG1", "ENSG2"])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_meta", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()

    mapping = MetadataMappingSubmission(
        datasetVersionId="dsv_test",
        organism=Organism.HUMAN,
        geneIdScheme=GeneIdScheme.ENSEMBL,
        sampleColumn="donor",
        sampleMappings=[
            SampleMappingEntry(sourceValue="patient_A", sampleKey="SMP_001"),
            SampleMappingEntry(sourceValue="patient_B", sampleKey="SMP_002"),
        ],
        conditionColumn="disease_state",
        batchColumn="run_batch",
    )

    out_p = tmp_path / "canonical_mapped.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        mapping=mapping,
    )
    c_adata = ad.read_h5ad(out_p)
    assert list(c_adata.obs["sample_key"]) == ["SMP_001", "SMP_002"]
    assert list(c_adata.obs["condition"]) == ["control", "treated"]
    assert list(c_adata.obs["batch"]) == ["b1", "b2"]


def test_source_immutability(tmp_path: Path) -> None:
    """Canonicalization does not mutate the source H5AD file."""
    p = tmp_path / "source_immut.h5ad"
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
    )
    adata.write_h5ad(p)

    sha_before = compute_file_sha256(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_immut", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    out_p = tmp_path / "canonical_out.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )

    sha_after = compute_file_sha256(p)
    assert sha_before == sha_after


def test_source_path_equals_output_path_rejected(tmp_path: Path) -> None:
    """Accidental source_path == output_path is strictly rejected."""
    p = tmp_path / "same.h5ad"
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
    )
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_same", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    with pytest.raises(ImporterSecurityError, match="Source file immutability violation"):
        importer.canonicalize(
            source_set=source_set,
            output_path=str(p),
            organism=Organism.HUMAN,
            gene_id_scheme=GeneIdScheme.ENSEMBL,
        )


def test_sparse_preservation(tmp_path: Path) -> None:
    """Canonicalization of a sparse source matrix remains CSR sparse without densification."""
    p = tmp_path / "sparse_large.h5ad"
    # 5,000 cells x 10,000 genes with low density
    sparse_x = sp.csr_matrix(
        ([1, 2, 3], ([0, 100, 4999], [0, 500, 9999])), shape=(5000, 10000), dtype=np.int32
    )
    obs = pd.DataFrame(index=[f"c_{i}" for i in range(5000)])
    var = pd.DataFrame(index=[f"ENSG{i:08d}" for i in range(10000)])
    adata = ad.AnnData(X=sparse_x, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_sp", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()

    out_p = tmp_path / "canonical_sparse.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    c_adata = ad.read_h5ad(out_p)
    assert sp.isspmatrix_csr(c_adata.X)
    assert c_adata.X.nnz == 3


# =========================================================================
# ORGANISM RESOLUTION & STEP D REUSE REGRESSION MATRIX
# =========================================================================


def test_organism_100_percent_human_ensembl(tmp_path: Path) -> None:
    """100% HUMAN Ensembl IDs with no metadata resolves to HUMAN."""
    p = tmp_path / "human_100.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSG00000141510", "ENSG00000139618"])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_h100", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_h100",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.VALID
    assert inspect_res.detectedOrganism == Organism.HUMAN
    assert inspect_res.detectedGeneIdScheme == GeneIdScheme.ENSEMBL


def test_organism_100_percent_mouse_ensembl(tmp_path: Path) -> None:
    """100% MOUSE Ensembl IDs with no metadata resolves to MOUSE."""
    p = tmp_path / "mouse_100.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSMUSG00000059552", "ENSMUSG00000001234"])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_m100", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_m100",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.VALID
    assert inspect_res.detectedOrganism == Organism.MOUSE
    assert inspect_res.detectedGeneIdScheme == GeneIdScheme.ENSEMBL


def test_organism_80_percent_human_20_percent_symbols_unresolved(tmp_path: Path) -> None:
    """80% HUMAN Ensembl + 20% symbols/random IDs DOES NOT resolve HUMAN (conservative Step D policy)."""
    p = tmp_path / "human_80.h5ad"
    # 4 Human Ensembl IDs (80%) + 1 gene symbol (20%)
    x = sp.csr_matrix([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(
        index=[
            "ENSG00000141510",
            "ENSG00000139618",
            "ENSG00000157764",
            "ENSG00000171862",
            "UNRESOLVED_SYMBOL",
        ]
    )
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_h80", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_h80",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    # Conservative policy requires 100% consistent Ensembl IDs with 0 non-Ensembl
    assert inspect_res.detectedOrganism is None
    assert inspect_res.status == InspectionStatus.NEEDS_MAPPING
    assert inspect_res.requiresMapping is True


def test_organism_80_percent_mouse_20_percent_symbols_unresolved(tmp_path: Path) -> None:
    """80% MOUSE Ensembl + 20% symbols/random IDs DOES NOT resolve MOUSE (conservative Step D policy)."""
    p = tmp_path / "mouse_80.h5ad"
    # 4 Mouse Ensembl IDs (80%) + 1 gene symbol (20%)
    x = sp.csr_matrix([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(
        index=[
            "ENSMUSG00000059552",
            "ENSMUSG00000001234",
            "ENSMUSG00000005678",
            "ENSMUSG00000009999",
            "ACTB",
        ]
    )
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_m80", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_m80",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    assert inspect_res.detectedOrganism is None
    assert inspect_res.status == InspectionStatus.NEEDS_MAPPING
    assert inspect_res.requiresMapping is True


def test_organism_mixed_human_mouse_ensembl_conflict(tmp_path: Path) -> None:
    """Mixture of Human (ENSG) and Mouse (ENSMUSG) IDs is a conflict and marks dataset INVALID."""
    p = tmp_path / "mixed_species.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSG00000141510", "ENSMUSG00000059552"])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_mixed", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_mixed",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.INVALID
    assert any(
        iss.code == ScientificIssueCode.UNRESOLVED_SPECIES
        and iss.severity == ScientificIssueSeverity.ERROR
        for iss in inspect_res.issues
    )


def test_organism_symbols_only_no_metadata_needs_mapping(tmp_path: Path) -> None:
    """Symbol-only dataset without explicit metadata flags NEEDS_MAPPING."""
    p = tmp_path / "symbols_only.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["TP53", "BRCA1"])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_sym", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_sym",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    assert inspect_res.detectedOrganism is None
    assert inspect_res.status == InspectionStatus.NEEDS_MAPPING
    assert inspect_res.requiresMapping is True


def test_organism_symbols_only_exact_human_metadata(tmp_path: Path) -> None:
    """Symbol-only dataset with exact 'Homo sapiens' metadata resolves to HUMAN."""
    p = tmp_path / "symbols_human.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["TP53", "BRCA1"])
    adata = ad.AnnData(X=x, obs=obs, var=var, uns={"organism": "Homo sapiens"})
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_sym_h", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_sym_h",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.VALID
    assert inspect_res.detectedOrganism == Organism.HUMAN


def test_organism_symbols_only_exact_mouse_metadata(tmp_path: Path) -> None:
    """Symbol-only dataset with exact 'Mus musculus' metadata resolves to MOUSE."""
    p = tmp_path / "symbols_mouse.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["Trp53", "Actb"])
    adata = ad.AnnData(X=x, obs=obs, var=var, uns={"organism": "Mus musculus"})
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_sym_m", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_sym_m",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.VALID
    assert inspect_res.detectedOrganism == Organism.MOUSE


def test_organism_explicit_human_conflicts_with_mouse_ensembl(tmp_path: Path) -> None:
    """Explicit HUMAN metadata with MOUSE Ensembl IDs is a conflict (INVALID)."""
    p = tmp_path / "conflict_h_m.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSMUSG00000059552", "ENSMUSG00000001234"])
    adata = ad.AnnData(X=x, obs=obs, var=var, uns={"organism": "HUMAN"})
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_conf_hm", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_conf_hm",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.INVALID
    assert any(
        iss.code == ScientificIssueCode.UNRESOLVED_SPECIES
        and iss.severity == ScientificIssueSeverity.ERROR
        for iss in inspect_res.issues
    )


def test_organism_explicit_mouse_conflicts_with_human_ensembl(tmp_path: Path) -> None:
    """Explicit MOUSE metadata with HUMAN Ensembl IDs is a conflict (INVALID)."""
    p = tmp_path / "conflict_m_h.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSG00000141510", "ENSG00000139618"])
    adata = ad.AnnData(X=x, obs=obs, var=var, uns={"organism": "mouse"})
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_conf_mh", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_conf_mh",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.INVALID
    assert any(
        iss.code == ScientificIssueCode.UNRESOLVED_SPECIES
        and iss.severity == ScientificIssueSeverity.ERROR
        for iss in inspect_res.issues
    )


def test_organism_explicit_unsupported_species_rejected(tmp_path: Path) -> None:
    """Explicit unsupported species in metadata raises UNSUPPORTED_SPECIES."""
    p = tmp_path / "unsupported_species.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=x, obs=obs, var=var, uns={"species": "Danio rerio"})
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_unsupp", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_unsupp",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.INVALID
    assert any(iss.code == ScientificIssueCode.UNSUPPORTED_SPECIES for iss in inspect_res.issues)


@pytest.mark.parametrize(
    "invalid_val",
    [
        "humanized mouse",
        "mouse-like",
        "Homo sapiens sample",
        "mixed human/mouse",
        "rat",
    ],
)
def test_organism_metadata_fuzzy_substring_values_rejected(
    tmp_path: Path, invalid_val: str
) -> None:
    """Values with substrings of human/mouse do NOT resolve silently (no fuzzy matching)."""
    p = tmp_path / f"fuzzy_{abs(hash(invalid_val))}.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=x, obs=obs, var=var, uns={"organism": invalid_val})
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_fuzz", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_fuzz",
    )
    importer = H5ADImporter()
    inspect_res = importer.inspect(source_set)
    assert inspect_res.status == InspectionStatus.INVALID
    assert any(iss.code == ScientificIssueCode.UNSUPPORTED_SPECIES for iss in inspect_res.issues)


def test_organism_metadata_taxonomy_ids(tmp_path: Path) -> None:
    """Exact taxonomy IDs '9606' and '10090' resolve cleanly without substring ambiguity."""
    p_h = tmp_path / "tax_9606.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["g1", "g2"])
    ad.AnnData(X=x, obs=obs, var=var, uns={"taxon_id": "9606"}).write_h5ad(p_h)

    importer = H5ADImporter()
    res_h = importer.inspect(
        ImportSourceSet(
            SourceFormat.H5AD,
            (ImportSourceFile("f1", ImportFileRole.PRIMARY, p_h),),
            dataset_version_id="dsv_tax_h",
        )
    )
    assert res_h.detectedOrganism == Organism.HUMAN

    p_m = tmp_path / "tax_10090.h5ad"
    ad.AnnData(X=x, obs=obs, var=var, uns={"taxon_id": 10090}).write_h5ad(p_m)
    res_m = importer.inspect(
        ImportSourceSet(
            SourceFormat.H5AD,
            (ImportSourceFile("f2", ImportFileRole.PRIMARY, p_m),),
            dataset_version_id="dsv_tax_m",
        )
    )
    assert res_m.detectedOrganism == Organism.MOUSE


# =========================================================================
# RAW COUNT AMBIGUITY REGRESSION TESTS
# =========================================================================


def test_ambiguous_floating_x_rejected(tmp_path: Path) -> None:
    """Near-integral floating X (within machine epsilon) produces AMBIGUOUS_COUNT_SEMANTICS."""
    p = tmp_path / "ambig_x.h5ad"
    # 10.0 + 1e-15 is classified as AMBIGUOUS by Step D classify_count_matrix
    ambig_mat = np.array([[10.0 + 1e-15, 20.0], [30.0, 40.0]], dtype=np.float64)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSG1", "ENSG2"])
    adata = ad.AnnData(X=ambig_mat, obs=obs, var=var)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_ambig_x", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    issues = importer.validate(source_set)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS


def test_ambiguous_counts_layer_rejected(tmp_path: Path) -> None:
    """Near-integral floating layers['counts'] produces AMBIGUOUS_COUNT_SEMANTICS."""
    p = tmp_path / "ambig_layer.h5ad"
    x_norm = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    ambig_mat = np.array([[10.0 + 1e-15, 20.0], [30.0, 40.0]], dtype=np.float64)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSG1", "ENSG2"])
    adata = ad.AnnData(X=x_norm, obs=obs, var=var, layers={"counts": ambig_mat})
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_ambig_l", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    issues = importer.validate(source_set)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS


def test_ambiguous_raw_x_rejected(tmp_path: Path) -> None:
    """Near-integral floating raw.X produces AMBIGUOUS_COUNT_SEMANTICS without silent bypass."""
    p = tmp_path / "ambig_raw.h5ad"
    x_norm = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    ambig_mat = np.array([[10.0 + 1e-15, 20.0], [30.0, 40.0]], dtype=np.float64)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSG1", "ENSG2"])

    raw_adata = ad.AnnData(X=ambig_mat, obs=obs, var=var)
    adata = ad.AnnData(X=x_norm, obs=obs, var=var)
    adata.raw = raw_adata
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_ambig_raw", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    issues = importer.validate(source_set)
    assert len(issues) == 1
    assert issues[0].code == ScientificIssueCode.AMBIGUOUS_COUNT_SEMANTICS


# =========================================================================
# CANONICAL UNS PROVENANCE AUDIT TEST
# =========================================================================


def test_canonical_uns_audit(tmp_path: Path) -> None:
    """Verifies that canonical uns contains ONLY approved scientific metadata.

    Invariants:
    1. adata.uns contains ONLY 'organism', 'gene_id_scheme', and 'reference_build'.
    2. 'cellweave' dictionary is NOT present in uns (source SHA authority belongs to persistence).
    3. Source uns analysis metadata is NOT copied wholesale.
    """
    p = tmp_path / "source_uns.h5ad"
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    obs = pd.DataFrame(index=["c1", "c2"])
    var = pd.DataFrame(index=["ENSG1", "ENSG2"])
    # Source has extensive analysis metadata
    source_uns = {
        "pca": {"variance": [1.0, 0.5]},
        "neighbors": {"connectivities": "dummy"},
        "source_checksum": "fake_checksum",
        "unapproved_key": 12345,
    }
    adata = ad.AnnData(X=x, obs=obs, var=var, uns=source_uns)
    adata.write_h5ad(p)

    source_set = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_uns", ImportFileRole.PRIMARY, p),),
    )
    importer = H5ADImporter()
    out_p = tmp_path / "canonical_clean_uns.h5ad"
    importer.canonicalize(
        source_set=source_set,
        output_path=str(out_p),
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        reference_build="GRCh38",
    )

    c_adata = ad.read_h5ad(out_p)
    # 1. Check exact allowed keys
    assert set(c_adata.uns.keys()) == {"organism", "gene_id_scheme", "reference_build"}
    assert c_adata.uns["organism"] == "HUMAN"
    assert c_adata.uns["gene_id_scheme"] == "ENSEMBL"
    assert c_adata.uns["reference_build"] == "GRCh38"

    # 2. Check that no unapproved or source keys survived
    assert "cellweave" not in c_adata.uns
    assert "source_checksum" not in c_adata.uns
    assert "pca" not in c_adata.uns
    assert "neighbors" not in c_adata.uns
    assert "unapproved_key" not in c_adata.uns


def test_h5ad_inspect_missing_dataset_version_id_fails_controlled(tmp_path: Path) -> None:
    """inspect() fails closed with MISSING_REQUIRED_METADATA if dataset_version_id is absent."""
    p = tmp_path / "valid.h5ad"
    ad.AnnData(
        X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["ENSG00000141510", "ENSG00000139618"]),
    ).write_h5ad(p)

    importer = H5ADImporter()

    # None dataset_version_id
    src_none = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
        dataset_version_id=None,
    )
    with pytest.raises(ImporterSecurityError) as exc_info:
        importer.inspect(src_none)
    assert exc_info.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA

    # Empty string dataset_version_id
    src_empty = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("f1", ImportFileRole.PRIMARY, p),),
        dataset_version_id="   ",
    )
    with pytest.raises(ImporterSecurityError) as exc_info:
        importer.inspect(src_empty)
    assert exc_info.value.issue_code == ScientificIssueCode.MISSING_REQUIRED_METADATA


def test_h5ad_inspect_dataset_version_id_distinct_from_file_id(tmp_path: Path) -> None:
    """inspect() reflects exact dataset_version_id and never falls back to file ID."""
    p = tmp_path / "valid.h5ad"
    ad.AnnData(
        X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["ENSG00000141510", "ENSG00000139618"]),
    ).write_h5ad(p)

    src = ImportSourceSet(
        source_format=SourceFormat.H5AD,
        files=(ImportSourceFile("dsf_file_identifier_999", ImportFileRole.PRIMARY, p),),
        dataset_version_id="dsv_version_identifier_111",
    )
    res = H5ADImporter().inspect(src)
    assert res.datasetVersionId == "dsv_version_identifier_111"
    assert res.datasetVersionId != "dsf_file_identifier_999"
