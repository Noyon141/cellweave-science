"""Narrow typing boundary for Linux-only stdlib resource accounting/enforcement."""

from __future__ import annotations

import importlib
import sys
from typing import Protocol, cast


class Usage(Protocol):
    ru_utime: float
    ru_stime: float
    ru_maxrss: int


class LinuxResource(Protocol):
    RLIMIT_CPU: int
    RUSAGE_CHILDREN: int

    def setrlimit(self, resource: int, limits: tuple[int, int]) -> None: ...
    def getrusage(self, who: int) -> Usage: ...


def load_linux_resource() -> LinuxResource:
    platform_name: str = sys.platform
    if platform_name != "linux":
        raise RuntimeError("Scientific subprocess resource enforcement requires Linux")
    # Standard-library Linux API; Windows type stubs intentionally omit these names.
    return cast(LinuxResource, importlib.import_module("resource"))
