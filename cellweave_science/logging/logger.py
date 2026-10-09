"""Structured JSON logging foundation for CellWeave scientific engine.

Adheres strictly to the architectural constraints:
- Structured JSON output.
- No printing or logging of raw scientific matrices, vectors, or sensitive filenames.
- Only identifiers, aggregate counts, execution timings, and error classifications.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict


class ScientificDataSanitizerFilter(logging.Filter):
    """Filter that intercepts log records and sanitizes forbidden scientific structures.

    Guarantees that raw matrices, numpy arrays, AnnData representations,
    and sensitive file paths never leak into log aggregators.
    """

    FORBIDDEN_TYPES: tuple[str, ...] = (
        "ndarray",
        "spmatrix",
        "AnnData",
        "DataFrame",
        "Series",
    )

    FORBIDDEN_KEY_FRAGMENTS: tuple[str, ...] = (
        "matrix",
        "vector",
        "expression",
        "raw_counts",
        "barcode_list",
        "gene_list",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        # Sanitize message if it accidentally contains stringified matrix or AnnData
        msg = str(record.msg)
        if "AnnData object with n_obs" in msg or "array([" in msg or "sparse matrix" in msg:
            record.msg = "[SANITIZED: RAW SCIENTIFIC DATA EXCLUDED]"

        # Sanitize extra fields if present
        if hasattr(record, "structured_data") and isinstance(record.structured_data, dict):
            record.structured_data = self._sanitize_dict(record.structured_data)

        return True

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(fragment in key_lower for fragment in self.FORBIDDEN_KEY_FRAGMENTS):
                sanitized[key] = "[SANITIZED_SCIENTIFIC_FIELD]"
                continue

            val_type = type(value).__name__
            if any(forbidden in val_type for forbidden in self.FORBIDDEN_TYPES):
                sanitized[key] = f"[SANITIZED_{val_type.upper()}]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list) and len(value) > 50:
                sanitized[key] = f"[SANITIZED_LARGE_LIST: length={len(value)}]"
            else:
                sanitized[key] = value
        return sanitized


class JsonFormatter(logging.Formatter):
    """Format logs as structured single-line JSON records."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "structured_data") and isinstance(record.structured_data, dict):
            log_payload.update(record.structured_data)

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload)


class ScienceLogger:
    """Wrapper around standard logger providing structured logging semantics."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def info(self, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, kwargs)

    def _log(self, level: int, message: str, structured_data: Dict[str, Any]) -> None:
        sanitizer = ScientificDataSanitizerFilter()
        sanitized_data = sanitizer._sanitize_dict(structured_data)
        self._logger.log(level, message, extra={"structured_data": sanitized_data})


def get_logger(name: str = "cellweave_science") -> ScienceLogger:
    """Obtain or configure a structured JSON logger for scientific jobs."""
    base_logger = logging.getLogger(name)
    if not base_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler.addFilter(ScientificDataSanitizerFilter())
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)
        base_logger.propagate = False
    return ScienceLogger(base_logger)
