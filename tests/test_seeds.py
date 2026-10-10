"""Test deterministic seed utilities and reproducibility."""

from __future__ import annotations

import random

import pytest

from cellweave_science.seeds.seed import (
    DeterministicSeedContext,
    set_deterministic_seeds,
)


@pytest.mark.unit
def test_seed_reproducibility() -> None:
    """Verify that repeated seed setting produces identical pseudo-random sequences."""
    set_deterministic_seeds(12345)
    sequence_a = [random.random() for _ in range(10)]

    set_deterministic_seeds(12345)
    sequence_b = [random.random() for _ in range(10)]

    assert sequence_a == sequence_b


@pytest.mark.unit
def test_different_seeds_produce_different_sequences() -> None:
    """Verify that distinct seeds produce distinct outputs."""
    set_deterministic_seeds(100)
    seq_1 = [random.random() for _ in range(5)]

    set_deterministic_seeds(200)
    seq_2 = [random.random() for _ in range(5)]

    assert seq_1 != seq_2


@pytest.mark.unit
def test_seed_context_manager() -> None:
    """Verify that DeterministicSeedContext executes deterministically within scope."""
    with DeterministicSeedContext(999):
        val1 = random.random()

    with DeterministicSeedContext(999):
        val2 = random.random()

    assert val1 == val2
