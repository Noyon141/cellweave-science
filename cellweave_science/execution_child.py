"""Private Linux child entrypoint. Receives capabilities on stdin, never in argv/logs."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from cellweave_contracts import PreparedExecution
from cellweave_science.linux_resource import load_linux_resource


def main() -> None:
    resource = load_linux_resource()
    raw = sys.stdin.buffer.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Execution metadata limit exceeded")
    request = PreparedExecution.model_validate_json(raw, strict=True)
    budget = request.envelope.ceilings
    # CPU seconds are a technical execution unit, never a commercial credit.
    cpu_seconds = math.ceil(budget.maxComputeUnits)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    from cellweave_science.execution import execute_prepared

    result = execute_prepared(request)
    Path(sys.argv[1]).write_text(result.model_dump_json(exclude_none=True), encoding="utf-8")


if __name__ == "__main__":
    main()
