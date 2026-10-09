"""Test structured JSON logging and scientific data sanitization."""

from __future__ import annotations

import io
import json
import logging
import pytest

from cellweave_science.logging.logger import (
    JsonFormatter,
    ScientificDataSanitizerFilter,
    get_logger,
)


@pytest.mark.unit
def test_json_formatter_outputs_valid_json() -> None:
    """Verify that JsonFormatter generates parseable JSON strings."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test log message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Test log message"
    assert "timestamp" in parsed


@pytest.mark.unit
def test_sanitizer_blocks_forbidden_keys() -> None:
    """Verify that sensitive scientific dictionary keys are replaced."""
    sanitizer = ScientificDataSanitizerFilter()
    data = {
        "cell_count": 1000,
        "expression_matrix": [1.0, 2.0, 3.0],
        "raw_counts": [[1, 2], [3, 4]],
        "run_id": "run-123",
    }
    sanitized = sanitizer._sanitize_dict(data)

    assert sanitized["cell_count"] == 1000
    assert sanitized["run_id"] == "run-123"
    assert sanitized["expression_matrix"] == "[SANITIZED_SCIENTIFIC_FIELD]"
    assert sanitized["raw_counts"] == "[SANITIZED_SCIENTIFIC_FIELD]"


@pytest.mark.unit
def test_sanitizer_blocks_anndata_string_representations() -> None:
    """Verify that stringified AnnData representation messages are sanitized."""
    sanitizer = ScientificDataSanitizerFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="AnnData object with n_obs × n_vars = 1000 × 20000",
        args=(),
        exc_info=None,
    )
    sanitizer.filter(record)
    assert record.msg == "[SANITIZED: RAW SCIENTIFIC DATA EXCLUDED]"
