"""Modal application deployment foundation for CellWeave scientific engine.

Defines the container image, resource profile mappings, and placeholder
healthcheck job deployed to dev/staging environments for Phase 0 exit criteria.
"""

from __future__ import annotations

import time
from typing import Any

try:
    import modal

    _MODAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    modal = None  # type: ignore[assignment]
    _MODAL_AVAILABLE = False


APP_NAME = "cellweave-science"

# Base scientific container image specification for Modal
if _MODAL_AVAILABLE and modal is not None:
    science_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "build-essential")
        .pip_install(
            "pydantic>=2.6.0,<3.0.0",
            "numpy>=1.24.0,<2.0.0",
            "scipy>=1.11.0,<1.14.0",
            "pandas>=2.1.0,<3.0.0",
            "anndata>=0.10.0,<0.12.0",
            "scanpy>=1.9.0,<1.11.0",
            "structlog>=24.1.0,<25.0.0",
        )
    )
    app = modal.App(name=APP_NAME, image=science_image)
else:  # pragma: no cover
    science_image = None  # type: ignore[assignment]
    app = None  # type: ignore[assignment]


def healthcheck() -> dict[str, Any]:
    """Phase 0 placeholder healthcheck function.

    Verifies that the scientific execution runtime is responsive and healthy.
    """
    return {
        "status": "healthy",
        "service": APP_NAME,
        "phase": "0-foundations",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# Register healthcheck with Modal App if Modal is present
if _MODAL_AVAILABLE and app is not None:
    healthcheck = app.function(cpu=1.0, memory=1024)(healthcheck)  # type: ignore[assignment]
