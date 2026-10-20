"""Exceptions for scientific dataset import and canonicalization."""

from __future__ import annotations

from typing import Any

from cellweave_science.contracts import ScientificIssueCode


class ImporterSecurityError(RuntimeError):
    """Raised when an untrusted or malformed file/path violates importer security bounds."""

    def __init__(
        self,
        message: str,
        *,
        issue_code: ScientificIssueCode = ScientificIssueCode.SECURITY_REJECTED,
    ) -> None:
        super().__init__(message)
        self.issue_code = issue_code


class DecompressionLimitError(ImporterSecurityError):
    """Raised when decompression exceeds configured safety limits (ratio, uncompressed bytes)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, issue_code=ScientificIssueCode.RESOURCE_LIMIT)


class MalformedArchiveError(ImporterSecurityError):
    """Raised when compressed streams or containers are corrupted or unreadable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, issue_code=ScientificIssueCode.MALFORMED_INPUT)


class CountValidationError(ValueError):
    """Raised when raw count matrix fails numerical invariants (NaN, infinite, negative, normalized)."""

    def __init__(
        self, message: str, *, issue_code: ScientificIssueCode = ScientificIssueCode.NON_COUNT_INPUT
    ) -> None:
        super().__init__(message)
        self.issue_code = issue_code


class CanonicalValidationError(ValueError):
    """Raised when a candidate AnnData fails canonical Phase 3 invariants."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


__all__ = [
    "ImporterSecurityError",
    "DecompressionLimitError",
    "MalformedArchiveError",
    "CountValidationError",
    "CanonicalValidationError",
]
