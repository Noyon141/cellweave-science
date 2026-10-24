"""Unit tests for Phase 3 Step D canonical AnnData builder and validator primitives."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellweave_science.contracts import GeneIdScheme, Organism
from cellweave_science.importers.canonical import (
    build_canonical_anndata,
    validate_canonical_anndata,
)
from cellweave_science.importers.errors import CanonicalValidationError, CountValidationError


@pytest.mark.parametrize(
    "slot,key",
    [
        *[
            ("uns", key)
            for key in [
                "source_checksum",
                "canonical_checksum",
                "organization_id",
                "project_id",
                "storage_key",
                "random_metadata",
                "neighbors",
                "leiden",
                "qc_metrics",
            ]
        ],
        *[
            ("obs", key)
            for key in [
                "cellAxisIndexColumn",
                "source_row_number",
                "raw_annotation",
                "qc_total_counts",
                "leiden",
            ]
        ],
        *[
            ("var", key)
            for key in [
                "featureAxisIndexColumn",
                "highly_variable",
                "mean",
                "dispersion",
                "qc_metric",
                "source_random_column",
            ]
        ],
    ],
)
def test_canonical_rejects_unauthorized_metadata_without_mutation(slot: str, key: str) -> None:
    adata = build_canonical_anndata(
        sp.csr_matrix([[1]]), obs_names=["CELL_A-1"], var_names=["ENSG00000141510.7"]
    )
    container = getattr(adata, slot)
    container[key] = "prohibited"
    before = adata.copy()
    with pytest.raises(CanonicalValidationError, match=f"unauthorized {slot}"):
        validate_canonical_anndata(adata)
    pd.testing.assert_frame_equal(adata.obs, before.obs)
    pd.testing.assert_frame_equal(adata.var, before.var)
    assert adata.uns == before.uns
    assert (adata.X != before.X).nnz == 0


def test_canonical_optional_metadata_and_exact_identities_do_not_mutate() -> None:
    obs = pd.DataFrame(
        {
            col: ["value"]
            for col in [
                "sample_key",
                "condition",
                "batch",
                "biological_replicate",
                "technical_replicate",
            ]
        },
        index=["AAACCCAAGT-1"],
    )
    var = pd.DataFrame(
        {"source_feature_id": ["probe_α.7"], "gene_symbol": ["TP53"]}, index=["ENSG00000141510.7"]
    )
    adata = build_canonical_anndata(
        sp.csr_matrix([[1]]),
        obs=obs,
        var=var,
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        reference_build="GRCh38",
    )
    before = adata.copy()
    validate_canonical_anndata(adata)
    assert list(adata.obs_names) == list(obs.index)
    assert list(adata.var_names) == list(var.index)
    pd.testing.assert_frame_equal(adata.obs, before.obs)
    pd.testing.assert_frame_equal(adata.var, before.var)
    assert adata.uns == before.uns


@pytest.mark.parametrize("value", [" source ", 1, 1.0, True, None])
def test_canonical_source_feature_identity_is_not_normalized(value: object) -> None:
    adata = build_canonical_anndata(sp.csr_matrix([[1]]), obs_names=["c1"], var_names=["g1"])
    adata.var["source_feature_id"] = [value]
    with pytest.raises(CountValidationError):
        validate_canonical_anndata(adata)


@pytest.mark.parametrize("axis", ["obs", "var"])
@pytest.mark.parametrize("value", [" invalid ", 1, 1.0, True, None])
def test_canonical_validator_rejects_invalid_index_without_rewriting(
    axis: str, value: object
) -> None:
    adata = build_canonical_anndata(sp.csr_matrix([[1]]), obs_names=["c1"], var_names=["g1"])
    frame = getattr(adata, axis)
    frame.index = pd.Index([value], dtype=object)
    with pytest.raises(CountValidationError):
        validate_canonical_anndata(adata)
    assert frame.index.tolist() == [value]


# =========================================================================
# POSITIVE MATRIX TESTS
# =========================================================================


def test_positive_canonical_human_ensembl() -> None:
    """Positive test: HUMAN + ENSEMBL canonical AnnData."""
    x = sp.csr_matrix([[10, 0, 5], [0, 20, 1]], dtype=np.int32)
    adata = build_canonical_anndata(
        x=x,
        obs_names=["cell_1", "cell_2"],
        var_names=["ENSG00000001", "ENSG00000002", "ENSG00000003"],
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        reference_build="GRCh38",
    )
    validate_canonical_anndata(
        adata,
        expected_organism=Organism.HUMAN,
        expected_gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    assert sp.isspmatrix_csr(adata.X)
    assert "counts" not in adata.layers
    assert adata.uns["reference_build"] == "GRCh38"


def test_positive_canonical_mouse_ensembl() -> None:
    """Positive test: MOUSE + ENSEMBL canonical AnnData."""
    x = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)
    adata = build_canonical_anndata(
        x=x,
        obs_names=["cell_m1", "cell_m2"],
        var_names=["ENSMUSG00000001", "ENSMUSG00000002"],
        organism=Organism.MOUSE,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        reference_build="GRCm39",
    )
    validate_canonical_anndata(
        adata,
        expected_organism=Organism.MOUSE,
        expected_gene_id_scheme=GeneIdScheme.ENSEMBL,
    )
    assert sp.isspmatrix_csr(adata.X)
    assert "counts" not in adata.layers
    assert adata.uns["reference_build"] == "GRCm39"


def test_positive_canonical_symbol_scheme() -> None:
    """Positive test: explicit SYMBOL gene scheme with unique stable canonical var_names."""
    x = sp.csr_matrix([[5, 0], [0, 10]], dtype=np.int32)
    adata = build_canonical_anndata(
        x=x,
        obs_names=["c1", "c2"],
        var_names=["TP53", "EGFR"],
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.SYMBOL,
        reference_build=None,
    )
    validate_canonical_anndata(
        adata,
        expected_organism=Organism.HUMAN,
        expected_gene_id_scheme=GeneIdScheme.SYMBOL,
    )
    assert sp.isspmatrix_csr(adata.X)
    assert "counts" not in adata.layers
    assert "reference_build" not in adata.uns


# =========================================================================
# COMPLETE NEGATIVE MATRIX TESTS (16 Conditions)
# =========================================================================


def test_negative_1_dense_x() -> None:
    """Condition 1: Rejection of dense X."""
    adata = ad.AnnData(
        X=np.array([[1, 2], [3, 4]]),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    with pytest.raises(CanonicalValidationError, match="must be a scipy.sparse.csr_matrix"):
        validate_canonical_anndata(adata)


def test_negative_2_fractional_non_count_x() -> None:
    """Condition 2: Rejection of fractional continuous/normalized X."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[1.5, 2.0], [3.0, 4.2]], dtype=np.float64),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    with pytest.raises(CountValidationError, match="non-integral continuous float values"):
        validate_canonical_anndata(adata)


def test_negative_3_negative_x() -> None:
    """Condition 3: Rejection of negative X."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[-1, 2], [3, 4]], dtype=np.int32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    with pytest.raises(CountValidationError, match="negative expression values"):
        validate_canonical_anndata(adata)


def test_negative_4_nan_x() -> None:
    """Condition 4: Rejection of NaN in X."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    with pytest.raises(CountValidationError, match="NaN or Infinite"):
        validate_canonical_anndata(adata)


def test_negative_5_inf_x() -> None:
    """Condition 5: Rejection of Inf in X."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[np.inf, 2.0], [3.0, 4.0]], dtype=np.float32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    with pytest.raises(CountValidationError, match="NaN or Infinite"):
        validate_canonical_anndata(adata)


def test_negative_6_duplicate_obs_names() -> None:
    """Condition 6: Rejection of duplicate obs_names."""
    with pytest.warns(UserWarning, match="Observation names are not unique"):
        adata = ad.AnnData(
            X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
            obs=pd.DataFrame(index=["dup_cell", "dup_cell"]),
            var=pd.DataFrame(index=["g1", "g2"]),
        )
    with pytest.raises(CountValidationError, match="Duplicate canonical obs_names"):
        validate_canonical_anndata(adata)


def test_negative_7_duplicate_var_names() -> None:
    """Condition 7: Rejection of duplicate var_names."""
    with pytest.warns(UserWarning, match="Variable names are not unique"):
        adata = ad.AnnData(
            X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
            obs=pd.DataFrame(index=["c1", "c2"]),
            var=pd.DataFrame(index=["dup_gene", "dup_gene"]),
        )
    with pytest.raises(CountValidationError, match="Duplicate canonical var_names"):
        validate_canonical_anndata(adata)


def test_negative_8_empty_blank_obs_identifier() -> None:
    """Condition 8: Rejection of empty/blank cell identifier."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
        obs=pd.DataFrame(index=["c1", "   "]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    with pytest.raises(CountValidationError, match="Empty or whitespace identifier"):
        validate_canonical_anndata(adata)


def test_negative_9_empty_blank_var_identifier() -> None:
    """Condition 9: Rejection of empty/blank gene identifier."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", ""]),
    )
    with pytest.raises(CountValidationError, match="Empty or whitespace identifier"):
        validate_canonical_anndata(adata)


def test_negative_10_zero_observations() -> None:
    """Condition 10: Rejection of 0 observations."""
    adata = ad.AnnData(
        X=sp.csr_matrix((0, 2), dtype=np.int32),
        obs=pd.DataFrame(index=[]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    with pytest.raises(CanonicalValidationError, match="must have >0 cells"):
        validate_canonical_anndata(adata)


def test_negative_11_zero_variables() -> None:
    """Condition 11: Rejection of 0 variables."""
    adata = ad.AnnData(
        X=sp.csr_matrix((2, 0), dtype=np.int32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=[]),
    )
    with pytest.raises(CanonicalValidationError, match="must have >0 genes"):
        validate_canonical_anndata(adata)


def test_negative_12_x_obs_dimension_mismatch() -> None:
    """Condition 12: Rejection of X / obs dimension mismatch."""
    with pytest.raises(
        CanonicalValidationError, match="Cell identifier count .* does not match matrix rows"
    ):
        build_canonical_anndata(
            x=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
            obs_names=["c1"],  # 1 cell for 2 rows
            var_names=["g1", "g2"],
        )


def test_negative_13_x_var_dimension_mismatch() -> None:
    """Condition 13: Rejection of X / var dimension mismatch."""
    with pytest.raises(
        CanonicalValidationError, match="Gene identifier count .* does not match matrix columns"
    ):
        build_canonical_anndata(
            x=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
            obs_names=["c1", "c2"],
            var_names=["g1"],  # 1 gene for 2 columns
        )


def test_negative_14_unsupported_organism() -> None:
    """Condition 14: Rejection of unsupported organism in uns."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
        uns={"organism": "DANIO_RERIO"},
    )
    with pytest.raises(CanonicalValidationError, match="Unsupported organism in uns"):
        validate_canonical_anndata(adata)


def test_negative_15_unsupported_gene_id_scheme() -> None:
    """Condition 15: Rejection of unsupported gene ID scheme in uns."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
        uns={"gene_id_scheme": "ENTREZ_GENE_ID"},
    )
    with pytest.raises(CanonicalValidationError, match="Unsupported gene ID scheme in uns"):
        validate_canonical_anndata(adata)


def test_negative_16_forbidden_counts_layer() -> None:
    """Condition 16: Rejection of premature layers['counts']."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
        layers={"counts": sp.csr_matrix([[1, 2], [3, 4]], dtype=np.int32)},
    )
    with pytest.raises(
        CanonicalValidationError, match="must store raw counts strictly in X, not in layers"
    ):
        validate_canonical_anndata(adata)
