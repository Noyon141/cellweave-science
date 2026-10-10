"""Deterministic seed management for CellWeave scientific pipelines.

Ensures repeatable scientific outputs across stochastic routines (e.g., UMAP,
Leiden clustering, Scrublet doublet simulation, PCA randomized solvers)
by strictly enforcing deterministic seeds in Python, NumPy, and PyTorch (if present).
"""

from __future__ import annotations

import os
import random
from types import TracebackType

DEFAULT_SEED: int = 42


def set_deterministic_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds across all standard runtime libraries.

    Args:
        seed: Positive integer seed value to enforce determinism.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


class DeterministicSeedContext:
    """Context manager to guarantee deterministic execution within a scoped block."""

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed
        self._python_state: tuple[object, ...] | None = None

    def __enter__(self) -> DeterministicSeedContext:
        self._python_state = random.getstate()
        set_deterministic_seeds(self.seed)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._python_state is not None:
            random.setstate(self._python_state)
