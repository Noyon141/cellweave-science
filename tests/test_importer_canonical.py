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


def test_build_canonical_anndata_synthetic() -> None:
    """Build canonical AnnData with CSR X, Policy B slots, and controlled metadata."""
    x = np.array([[10, 0, 5], [0, 20, 1]], dtype=np.int32)
    obs_names = ["cell_A", "cell_B"]
    var_names = ["ENSG00000001", "ENSG00000002", "ENSG00000003"]

    adata = build_canonical_anndata(
        x=x,
        obs_names=obs_names,
        var_names=var_names,
        organism=Organism.HUMAN,
        gene_id_scheme=GeneIdScheme.ENSEMBL,
        reference_build="GRCh38",
    )

    # Invariants check
    assert adata.shape == (2, 3)
    assert sp.isspmatrix_csr(adata.X)
    assert (adata.X.toarray() == x).all()
    assert list(adata.obs_names) == obs_names
    assert list(adata.var_names) == var_names

    # Policy B: counts layer NOT duplicated
    assert "counts" not in adata.layers
    assert len(adata.obsm) == 0
    assert len(adata.varm) == 0

    # Metadata check
    assert adata.uns["organism"] == "HUMAN"
    assert adata.uns["gene_id_scheme"] == "ENSEMBL"
    assert adata.uns["reference_build"] == "GRCh38"

    # Conforms to canonical validator
    validate_canonical_anndata(
        adata,
        expected_organism=Organism.HUMAN,
        expected_gene_id_scheme=GeneIdScheme.ENSEMBL,
    )


def test_canonical_validator_rejects_dense_x() -> None:
    """Canonical validator rejects dense AnnData.X matrices."""
    adata = ad.AnnData(
        X=np.array([[1, 2], [3, 4]]),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    with pytest.raises(CanonicalValidationError, match="must be a scipy.sparse.csr_matrix"):
        validate_canonical_anndata(adata)


def test_canonical_validator_rejects_duplicate_counts_layer() -> None:
    """Canonical validator rejects premature layers['counts']."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 2], [3, 4]]),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
        layers={"counts": sp.csr_matrix([[1, 2], [3, 4]])},
    )
    with pytest.raises(CanonicalValidationError, match="must store raw counts strictly in X"):
        validate_canonical_anndata(adata)


def test_canonical_validator_rejects_non_counts() -> None:
    """Canonical validator rejects non-count matrices in X."""
    adata = ad.AnnData(
        X=sp.csr_matrix([[-1, 2], [3, 4]]),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    with pytest.raises(CountValidationError, match="negative expression values"):
        validate_canonical_anndata(adata)
