"""Test packaging structure, metadata, and subpackage presence."""

from __future__ import annotations

import importlib
from pathlib import Path
import pytest

import cellweave_science


@pytest.mark.unit
def test_package_metadata() -> None:
    """Verify package version and top-level attributes."""
    assert hasattr(cellweave_science, "__version__")
    assert cellweave_science.__version__ == "0.1.0"


@pytest.mark.unit
@pytest.mark.parametrize(
    "subpackage",
    [
        "importers",
        "validation",
        "qc",
        "normalization",
        "embeddings",
        "clustering",
        "annotation",
        "differential_expression",
        "pathways",
        "visualization",
        "exports",
        "provenance",
        "contracts",
        "seeds",
        "logging",
        "adapters",
    ],
)
def test_all_prescribed_subpackages_importable(subpackage: str) -> None:
    """Verify that every module prescribed by the system design is importable."""
    module = importlib.import_module(f"cellweave_science.{subpackage}")
    assert module is not None


@pytest.mark.unit
def test_py_typed_marker_exists() -> None:
    """Ensure PEP 561 py.typed marker is present in root package."""
    pkg_dir = Path(cellweave_science.__file__).resolve().parent
    py_typed = pkg_dir / "py.typed"
    assert py_typed.exists(), "py.typed marker missing from cellweave_science"


@pytest.mark.unit
def test_modal_app_and_benchmarks_importable() -> None:
    """Verify modal_app and benchmarks packages can be loaded."""
    import benchmarks
    import modal_app

    assert hasattr(modal_app, "healthcheck")
    assert hasattr(benchmarks, "BenchmarkRunner")
