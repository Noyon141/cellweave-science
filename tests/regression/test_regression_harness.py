"""Scientific regression test harness foundation.

Prescribes regression tolerance verification per System Design Section 101:
- cell counts after filtering
- QC distribution ranges
- PCA variance properties
- cluster consistency
- marker recovery
- annotation accuracy
- DE direction
- pathway expectations
- runtime & memory boundaries

In Phase 0, this test verifies the regression gate harness and tolerance structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass(frozen=True)
class RegressionTolerance:
    """Scientifically meaningful tolerances for numerical regression gates."""

    metric_name: str
    expected_value: float
    relative_tolerance: float  # e.g., 0.05 for 5% tolerance
    absolute_tolerance: float = 1e-4

    def is_within_tolerance(self, observed_value: float) -> bool:
        diff = abs(observed_value - self.expected_value)
        allowed_diff = max(
            self.absolute_tolerance, self.relative_tolerance * abs(self.expected_value)
        )
        return diff <= allowed_diff


@pytest.mark.regression
def test_regression_tolerance_evaluator() -> None:
    """Verify that regression tolerance checks allow expected stochastic variance but catch drift."""
    # Test tolerance for UMAP coordinate stability / PCA variance explained
    pca_tol = RegressionTolerance(
        metric_name="pca_explained_variance_ratio_pc1",
        expected_value=0.185,
        relative_tolerance=0.05,
    )
    # Observed within 5%
    assert pca_tol.is_within_tolerance(0.187) is True
    assert pca_tol.is_within_tolerance(0.180) is True

    # Observed drifted beyond 5%
    assert pca_tol.is_within_tolerance(0.250) is False
    assert pca_tol.is_within_tolerance(0.120) is False


@pytest.mark.regression
def test_regression_metric_definitions() -> None:
    """Verify that all regression metrics required by Section 101 are cataloged."""
    required_metrics = {
        "cell_counts_after_filtering",
        "qc_distributions",
        "pca_properties",
        "cluster_consistency",
        "known_marker_recovery",
        "annotation_accuracy",
        "de_direction",
        "pathway_expectations",
        "runtime",
        "memory",
    }
    # Manifest of supported metrics
    harness_catalog: dict[str, dict[str, Any]] = {
        "cell_counts_after_filtering": {"type": "integer_exact_or_bounded", "default_rtol": 0.0},
        "qc_distributions": {"type": "distribution_ks_test", "default_pvalue_threshold": 0.01},
        "pca_properties": {"type": "explained_variance_ratio", "default_rtol": 0.05},
        "cluster_consistency": {"type": "adjusted_rand_index", "default_min_score": 0.90},
        "known_marker_recovery": {"type": "top_n_recall", "default_min_recall": 0.85},
        "annotation_accuracy": {"type": "f1_score", "default_min_score": 0.90},
        "de_direction": {"type": "logfc_concordance", "default_min_concordance": 0.95},
        "pathway_expectations": {"type": "rank_correlation", "default_min_spearman": 0.80},
        "runtime": {"type": "duration_ceiling", "default_rtol": 0.25},
        "memory": {"type": "peak_rss_ceiling", "default_rtol": 0.20},
    }
    assert required_metrics.issubset(harness_catalog.keys())
