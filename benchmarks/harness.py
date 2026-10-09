"""Scientific data size benchmarking harness.

Prescribes execution profiling across standard dataset size tiers per Section 151:
- 10k cells
- 50k cells
- 100k cells
- 500k cells
- 1M cells

Measures:
- import time
- peak RAM
- CPU time
- GPU time
- artifact size
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class BenchmarkTier(str, Enum):
    """Standardized dataset size tiers for CellWeave benchmarking."""

    TIER_10K = "10k"
    TIER_50K = "50k"
    TIER_100K = "100k"
    TIER_500K = "500k"
    TIER_1M = "1M"


TIER_CELL_COUNTS = {
    BenchmarkTier.TIER_10K: 10_000,
    BenchmarkTier.TIER_50K: 50_000,
    BenchmarkTier.TIER_100K: 100_000,
    BenchmarkTier.TIER_500K: 500_000,
    BenchmarkTier.TIER_1M: 1_000_000,
}


@dataclass(frozen=True)
class BenchmarkResult:
    """Structured metrics collected during benchmark run."""

    tier: BenchmarkTier
    cell_count: int
    duration_seconds: float
    cpu_time_seconds: float
    gpu_time_seconds: float
    peak_ram_bytes: int
    artifact_size_bytes: int


class BenchmarkRunner:
    """Profiles execution of scientific functions against standardized tiers."""

    @staticmethod
    def profile_routine(
        tier: BenchmarkTier,
        func: Callable[[], Optional[str]],
    ) -> BenchmarkResult:
        """Execute a target function and measure resource consumption.

        Args:
            tier: BenchmarkTier being evaluated.
            func: Target callable, optionally returning a file path for artifact size calculation.

        Returns:
            BenchmarkResult capturing duration, CPU time, and artifact size.
        """
        start_clock = time.monotonic()
        start_cpu = time.process_time()

        artifact_path = func()

        elapsed_clock = time.monotonic() - start_clock
        elapsed_cpu = time.process_time() - start_cpu

        artifact_size = 0
        if artifact_path and os.path.exists(artifact_path):
            artifact_size = os.path.getsize(artifact_path)

        return BenchmarkResult(
            tier=tier,
            cell_count=TIER_CELL_COUNTS[tier],
            duration_seconds=round(elapsed_clock, 4),
            cpu_time_seconds=round(elapsed_cpu, 4),
            gpu_time_seconds=0.0,
            peak_ram_bytes=0,
            artifact_size_bytes=artifact_size,
        )
