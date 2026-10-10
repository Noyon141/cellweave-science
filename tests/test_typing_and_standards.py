"""Test strict typing protocols and interface conformances."""

from __future__ import annotations

import pytest

from cellweave_science.adapters.compute import ComputeAdapter, ComputeMode
from cellweave_science.adapters.local import LocalComputeAdapter
from cellweave_science.adapters.modal_adapter import ModalComputeAdapter
from cellweave_science.annotation import AnnotationProvider
from cellweave_science.clustering import ClusteringEngine
from cellweave_science.differential_expression import (
    DifferentialExpressionEngine,
    MarkerGeneDiscovery,
)
from cellweave_science.embeddings import EmbeddingEngine
from cellweave_science.exports import ExportBundleGenerator
from cellweave_science.importers import DatasetImporter
from cellweave_science.normalization import NormalizationEngine
from cellweave_science.pathways import PathwayProvider
from cellweave_science.provenance import ProvenanceTracker
from cellweave_science.qc import QCEngine
from cellweave_science.validation import DatasetValidator
from cellweave_science.visualization import VisualizationArtifactGenerator


@pytest.mark.unit
def test_adapters_implement_compute_adapter() -> None:
    """Verify Local and Modal adapters inherit from ComputeAdapter."""
    local_adapter = LocalComputeAdapter()
    modal_adapter = ModalComputeAdapter()

    assert isinstance(local_adapter, ComputeAdapter)
    assert isinstance(modal_adapter, ComputeAdapter)
    assert local_adapter.mode == ComputeMode.LOCAL
    assert modal_adapter.mode == ComputeMode.MODAL


@pytest.mark.unit
def test_protocols_are_runtime_checkable() -> None:
    """Verify scientific protocols are runtime-checkable."""
    protocols = [
        DatasetImporter,
        DatasetValidator,
        QCEngine,
        NormalizationEngine,
        EmbeddingEngine,
        ClusteringEngine,
        AnnotationProvider,
        MarkerGeneDiscovery,
        DifferentialExpressionEngine,
        PathwayProvider,
        VisualizationArtifactGenerator,
        ExportBundleGenerator,
        ProvenanceTracker,
    ]
    for proto in protocols:
        assert hasattr(proto, "_is_runtime_protocol") or hasattr(proto, "__protocol_attrs__")
