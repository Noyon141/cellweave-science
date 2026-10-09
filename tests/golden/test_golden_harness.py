"""Test golden reference dataset manifest and test harness structure."""

from __future__ import annotations

import re
from typing import Any, Dict
import pytest


REQUIRED_GOLDEN_KEYS = [
    "small_smoke_dataset",
    "human_pbmc_dataset",
    "mouse_dataset",
    "multi_sample_dataset",
    "batch_effect_dataset",
    "known_treatment_dataset",
    "large_benchmark_dataset",
]


@pytest.mark.golden
def test_golden_manifest_contains_all_prescribed_datasets(golden_manifest: Dict[str, Any]) -> None:
    """Verify that golden_manifest.json contains all datasets required by System Design Section 102."""
    datasets = golden_manifest.get("datasets", [])
    present_keys = {ds["key"] for ds in datasets}

    for required_key in REQUIRED_GOLDEN_KEYS:
        assert required_key in present_keys, f"Missing required golden dataset: {required_key}"


@pytest.mark.golden
def test_golden_manifest_dataset_attributes(golden_manifest: Dict[str, Any]) -> None:
    """Verify metadata validity and checksum formatting of golden datasets."""
    sha256_pattern = re.compile(r"^[a-f0-9]{64}$")

    for ds in golden_manifest.get("datasets", []):
        assert "key" in ds
        assert ds["organism"] in ("human", "mouse")
        assert ds["expectedCells"] > 0
        assert ds["expectedGenes"] > 0
        assert sha256_pattern.match(ds["sha256"]), f"Invalid SHA-256 for dataset {ds['key']}"
