"""Canonical AnnData builder and validation primitives conforming to Phase 3 Policy B.

Invariants:
- Policy B: adata.X stores authoritative raw unnormalized counts in SciPy CSR sparse matrix.
- adata.layers["counts"] is NOT created in Phase 3 (left clean for Phase 6 normalization).
- Matrix orientation is strictly (n_obs, n_vars) = (cells, genes).
- obs_names and var_names are unique, non-empty string identifiers.
- Controlled slot hygiene: no arbitrary source obsm, varm, obsp, varp, uns, or layers.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellweave_science.contracts import GeneIdScheme, Organism
from cellweave_science.importers.errors import CanonicalValidationError
from cellweave_science.importers.validation import (
    ensure_csr_matrix,
    validate_count_matrix,
    validate_identifiers,
)


def build_canonical_anndata(
    x: np.ndarray[Any, Any] | sp.spmatrix,
    obs: pd.DataFrame | None = None,
    var: pd.DataFrame | None = None,
    *,
    obs_names: list[str] | None = None,
    var_names: list[str] | None = None,
    organism: Organism | None = None,
    gene_id_scheme: GeneIdScheme | None = None,
    reference_build: str | None = None,
) -> ad.AnnData:
    """Build a canonical AnnData object adhering strictly to Phase 3 invariants.

    Args:
        x: Raw expression count matrix in cells x genes orientation.
        obs: Optional cell metadata DataFrame.
        var: Optional feature metadata DataFrame.
        obs_names: Optional explicit cell identifiers (if not in obs.index).
        var_names: Optional explicit feature identifiers (if not in var.index).
        organism: Confirmed species.
        gene_id_scheme: Confirmed identifier scheme.
        reference_build: Optional reference build (e.g., GRCh38).

    Returns:
        Canonical AnnData instance with CSR sparse X and controlled metadata slots.

    Raises:
        CanonicalValidationError: If input dimensions, identifiers, or counts violate invariants.
    """
    # 1. Validate raw count eligibility
    validate_count_matrix(x)

    n_obs, n_vars = x.shape

    # 2. Cell identifiers (obs_names)
    if obs_names is not None:
        clean_obs_names = validate_identifiers(obs_names, context="cell IDs (obs_names)")
    elif obs is not None and not obs.empty:
        clean_obs_names = validate_identifiers(obs.index.tolist(), context="cell IDs (obs.index)")
    else:
        raise CanonicalValidationError("Cell identifiers (obs_names) must be provided")

    if len(clean_obs_names) != n_obs:
        raise CanonicalValidationError(
            f"Cell identifier count ({len(clean_obs_names)}) does not match matrix rows ({n_obs})"
        )

    # 3. Gene identifiers (var_names)
    if var_names is not None:
        clean_var_names = validate_identifiers(var_names, context="gene IDs (var_names)")
    elif var is not None and not var.empty:
        clean_var_names = validate_identifiers(var.index.tolist(), context="gene IDs (var.index)")
    else:
        raise CanonicalValidationError("Gene identifiers (var_names) must be provided")

    if len(clean_var_names) != n_vars:
        raise CanonicalValidationError(
            f"Gene identifier count ({len(clean_var_names)}) does not match matrix columns ({n_vars})"
        )

    # 4. Construct clean obs & var DataFrames
    clean_obs = obs.copy() if obs is not None else pd.DataFrame(index=clean_obs_names)
    clean_obs.index = pd.Index(clean_obs_names, name="cell_id")

    clean_var = var.copy() if var is not None else pd.DataFrame(index=clean_var_names)
    clean_var.index = pd.Index(clean_var_names, name="gene_id")

    # 5. Ensure sparse CSR representation
    csr_x = ensure_csr_matrix(x)

    # 6. Initialize AnnData with clean slots (no obsm, varm, uns, layers)
    adata = ad.AnnData(
        X=csr_x,
        obs=clean_obs,
        var=clean_var,
    )

    # Controlled uns metadata
    if organism is not None:
        adata.uns["organism"] = organism.value
    if gene_id_scheme is not None:
        adata.uns["gene_id_scheme"] = gene_id_scheme.value
    if reference_build is not None:
        adata.uns["reference_build"] = reference_build

    return adata


def validate_canonical_anndata(
    adata: ad.AnnData,
    *,
    expected_organism: Organism | None = None,
    expected_gene_id_scheme: GeneIdScheme | None = None,
) -> None:
    """Validate that an AnnData instance conforms strictly to Phase 3 canonical requirements.

    Checks:
    1. n_obs > 0 and n_vars > 0.
    2. adata.X shape matches (n_obs, n_vars).
    3. adata.X is a SciPy CSR sparse matrix.
    4. adata.X data passes raw count validation (non-negative, finite, integers).
    5. obs_names and var_names are unique and non-empty.
    6. Expected organism and gene ID scheme are present if specified and match supported enums.
    7. Slot hygiene: layers is empty (no premature counts layer), obsm is empty.

    Raises:
        CanonicalValidationError: If any canonical invariant is violated.
    """
    if adata.n_obs <= 0:
        raise CanonicalValidationError(f"Canonical AnnData must have >0 cells, found {adata.n_obs}")

    if adata.n_vars <= 0:
        raise CanonicalValidationError(
            f"Canonical AnnData must have >0 genes, found {adata.n_vars}"
        )

    if adata.X is None:
        raise CanonicalValidationError("Canonical AnnData X matrix cannot be None")

    if not sp.isspmatrix_csr(adata.X):
        raise CanonicalValidationError(
            f"Canonical AnnData X must be a scipy.sparse.csr_matrix, found {type(adata.X)}"
        )

    if adata.X.shape != (adata.n_obs, adata.n_vars):
        raise CanonicalValidationError(
            f"Matrix shape {adata.X.shape} does not match (n_obs, n_vars) = ({adata.n_obs}, {adata.n_vars})"
        )

    # Validate raw counts
    validate_count_matrix(adata.X)

    # Validate identifier uniqueness and sanity
    validate_identifiers(adata.obs_names.tolist(), context="canonical obs_names")
    validate_identifiers(adata.var_names.tolist(), context="canonical var_names")

    # Validate only: never drop or rewrite metadata at the publication boundary.
    allowed_uns = {"organism", "gene_id_scheme", "reference_build"}
    allowed_obs = {
        "sample_key",
        "condition",
        "batch",
        "biological_replicate",
        "technical_replicate",
    }
    allowed_var = {"source_feature_id", "gene_symbol"}
    for label, keys, allowed in (
        ("uns", adata.uns.keys(), allowed_uns),
        ("obs", adata.obs.columns, allowed_obs),
        ("var", adata.var.columns, allowed_var),
    ):
        if any(key not in allowed for key in keys):
            raise CanonicalValidationError(
                f"Canonical AnnData contains unauthorized {label} metadata"
            )
    # Existing format-neutral callers may omit source_feature_id; importer outputs preserve it.
    if "source_feature_id" in adata.var:
        validate_identifiers(
            adata.var["source_feature_id"].tolist(),
            context="canonical source_feature_id",
            require_unique=False,
        )

    # Policy B hygiene checks: no layers, obsm, varm, obsp, varp, or raw
    if "counts" in adata.layers:
        raise CanonicalValidationError(
            "Canonical AnnData in Phase 3 must store raw counts strictly in X, not in layers['counts']"
        )

    if len(adata.layers) > 0:
        raise CanonicalValidationError(
            f"Canonical AnnData in Phase 3 must not contain layers, found: {list(adata.layers.keys())}"
        )

    if len(adata.obsm) > 0:
        raise CanonicalValidationError(
            f"Canonical AnnData in Phase 3 must not contain obsm slots, found: {list(adata.obsm.keys())}"
        )

    if len(adata.varm) > 0:
        raise CanonicalValidationError(
            f"Canonical AnnData in Phase 3 must not contain varm slots, found: {list(adata.varm.keys())}"
        )

    if len(adata.obsp) > 0:
        raise CanonicalValidationError(
            f"Canonical AnnData in Phase 3 must not contain obsp slots, found: {list(adata.obsp.keys())}"
        )

    if len(adata.varp) > 0:
        raise CanonicalValidationError(
            f"Canonical AnnData in Phase 3 must not contain varp slots, found: {list(adata.varp.keys())}"
        )

    if adata.raw is not None:
        raise CanonicalValidationError("Canonical AnnData in Phase 3 must not contain adata.raw")

    # Organism check
    stored_org = adata.uns.get("organism")
    if stored_org is not None:
        supported_orgs = {e.value for e in Organism}
        if stored_org not in supported_orgs:
            raise CanonicalValidationError(
                f"Unsupported organism in uns['organism']: {stored_org!r} (supported: {supported_orgs})"
            )
        if expected_organism is not None and stored_org != expected_organism.value:
            raise CanonicalValidationError(
                f"Expected organism {expected_organism.value}, found {stored_org}"
            )
    elif expected_organism is not None:
        raise CanonicalValidationError(
            f"Expected organism {expected_organism.value}, but uns['organism'] is not set"
        )

    # Gene ID scheme check
    stored_scheme = adata.uns.get("gene_id_scheme")
    if stored_scheme is not None:
        supported_schemes = {e.value for e in GeneIdScheme}
        if stored_scheme not in supported_schemes:
            raise CanonicalValidationError(
                f"Unsupported gene ID scheme in uns['gene_id_scheme']: {stored_scheme!r} (supported: {supported_schemes})"
            )
        if expected_gene_id_scheme is not None and stored_scheme != expected_gene_id_scheme.value:
            raise CanonicalValidationError(
                f"Expected gene ID scheme {expected_gene_id_scheme.value}, found {stored_scheme}"
            )
    elif expected_gene_id_scheme is not None:
        raise CanonicalValidationError(
            f"Expected gene ID scheme {expected_gene_id_scheme.value}, but uns['gene_id_scheme'] is not set"
        )


__all__ = [
    "build_canonical_anndata",
    "validate_canonical_anndata",
]
