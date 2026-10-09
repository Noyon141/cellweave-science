"""Compute adapter package for cellweave-science."""

from cellweave_science.adapters.compute import ComputeAdapter, ComputeMode
from cellweave_science.adapters.local import LocalComputeAdapter
from cellweave_science.adapters.modal_adapter import (
    MODAL_RESOURCE_CONFIGS,
    ModalComputeAdapter,
)

__all__ = [
    "MODAL_RESOURCE_CONFIGS",
    "ComputeAdapter",
    "ComputeMode",
    "LocalComputeAdapter",
    "ModalComputeAdapter",
]
