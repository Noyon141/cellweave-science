"""Unit tests for Phase 3 Step D identifier validation and species detection primitives."""

from __future__ import annotations

import pytest

from cellweave_science.contracts import Organism
from cellweave_science.importers.errors import CountValidationError
from cellweave_science.importers.validation import (
    detect_ensembl_organism_candidates,
    validate_identifiers,
)


def test_unique_cell_identifiers() -> None:
    """Valid unique cell IDs pass with whitespace trimmed."""
    raw_cells = ["cell_1", "cell_2 ", "cell_3"]
    cleaned = validate_identifiers(raw_cells, context="cell IDs")
    assert cleaned == ["cell_1", "cell_2", "cell_3"]


def test_duplicate_cell_identifiers_rejected() -> None:
    """Duplicate cell identifiers are rejected without silent auto-deduplication."""
    raw_cells = ["AAACCC1", "AAACCC2", "AAACCC1"]
    with pytest.raises(CountValidationError, match="Duplicate cell IDs found: 'AAACCC1'"):
        validate_identifiers(raw_cells, context="cell IDs")


def test_empty_or_whitespace_identifiers_rejected() -> None:
    """Empty or whitespace-only identifiers are rejected."""
    with pytest.raises(CountValidationError, match="Empty or whitespace identifier found"):
        validate_identifiers(["cell_1", "   ", "cell_2"])


def test_ensembl_human_candidates_detection() -> None:
    """Ensembl human gene ID candidates (ENSG...) detect HUMAN."""
    genes = [
        "ENSG00000141510",
        "ENSG00000139618.15",
        "ENSG00000157764",
    ]
    detected = detect_ensembl_organism_candidates(genes)
    assert detected == Organism.HUMAN


def test_ensembl_mouse_candidates_detection() -> None:
    """Ensembl mouse gene ID candidates (ENSMUSG...) detect MOUSE."""
    genes = [
        "ENSMUSG00000001234",
        "ENSMUSG00000005678.3",
        "ENSMUSG00000009999",
    ]
    detected = detect_ensembl_organism_candidates(genes)
    assert detected == Organism.MOUSE


def test_mixed_species_unresolved() -> None:
    """Mixed human and mouse identifiers return None (unresolved/ambiguous)."""
    mixed = [
        "ENSG00000141510",
        "ENSMUSG00000001234",
    ]
    detected = detect_ensembl_organism_candidates(mixed)
    assert detected is None


def test_gene_symbols_only_unresolved() -> None:
    """Gene symbols alone do not infer species automatically."""
    symbols = ["TP53", "EGFR", "CD4", "GAPDH"]
    detected = detect_ensembl_organism_candidates(symbols)
    assert detected is None
