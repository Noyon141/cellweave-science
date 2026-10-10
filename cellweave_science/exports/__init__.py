"""Export bundle and artifact package protocols.

Prescribes packaging of reproducible scientific exports and bundles
(processed .h5ad, CSV tables, reports, Reproducibility Bundle) per System Design Section 84.
Implementations belong to Phase 9.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExportBundleGenerator(Protocol):
    """Protocol for generating exportable research bundles."""

    def build_reproducibility_bundle(
        self,
        run_manifest_path: str,
        artifact_manifest_paths: list[str],
        output_bundle_path: str,
    ) -> dict[str, Any]:
        """Compile complete reproducibility zip bundle including run manifests and code versions."""
        ...


__all__ = ["ExportBundleGenerator"]
