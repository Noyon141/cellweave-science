"""Test benchmark harness and resource tracking functions."""

from __future__ import annotations

import pytest

from benchmarks.harness import (
    TIER_CELL_COUNTS,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkTier,
)


@pytest.mark.benchmark
def test_benchmark_tiers_defined() -> None:
    """Verify all five benchmark tiers defined in System Design Section 151 are present."""
    assert BenchmarkTier.TIER_10K.value == "10k"
    assert BenchmarkTier.TIER_50K.value == "50k"
    assert BenchmarkTier.TIER_100K.value == "100k"
    assert BenchmarkTier.TIER_500K.value == "500k"
    assert BenchmarkTier.TIER_1M.value == "1M"

    assert TIER_CELL_COUNTS[BenchmarkTier.TIER_10K] == 10_000
    assert TIER_CELL_COUNTS[BenchmarkTier.TIER_1M] == 1_000_000


@pytest.mark.benchmark
def test_benchmark_runner_profiles_routine() -> None:
    """Verify BenchmarkRunner executes and captures timings accurately."""
    def sample_routine() -> str:
        # Simulate quick work
        _ = sum(i * i for i in range(1000))
        return ""

    result = BenchmarkRunner.profile_routine(BenchmarkTier.TIER_10K, sample_routine)
    assert isinstance(result, BenchmarkResult)
    assert result.tier == BenchmarkTier.TIER_10K
    assert result.cell_count == 10_000
    assert result.duration_seconds >= 0.0
    assert result.cpu_time_seconds >= 0.0
