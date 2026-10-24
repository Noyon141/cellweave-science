"""Private native-Modal dispatch, identity reconciliation and bounded Phase 3 execution.

Deploy explicitly; importing this module never applies infrastructure. Configuration
is mandatory. Provider-side coordination is not the business database: web persists
accepted provider IDs and execution projections. No database or R2 master secrets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
from fastapi.responses import JSONResponse

from cellweave_contracts import PreparedExecution

ENVIRONMENT = os.environ["CELLWEAVE_ENVIRONMENT"]
if ENVIRONMENT not in {"development", "staging", "production"}:
    raise ValueError("Explicit Modal environment required")
PROFILE = json.loads(os.environ["CELLWEAVE_MODAL_PROFILE_JSON"])
if (
    PROFILE.get("resourceClass") not in {"CPU_SMALL", "CPU_MEMORY", "CPU_HIGH_MEMORY"}
    or not isinstance(PROFILE.get("memoryMiB"), int)
    or isinstance(PROFILE.get("memoryMiB"), bool)
    or PROFILE["memoryMiB"] <= 0
    or not isinstance(PROFILE.get("cpu"), int | float)
    or isinstance(PROFILE["cpu"], bool)
    or not math.isfinite(PROFILE["cpu"])
    or PROFILE["cpu"] <= 0
    or not isinstance(PROFILE.get("maxRuntimeSeconds"), int)
    or isinstance(PROFILE.get("maxRuntimeSeconds"), bool)
    or not 0 < PROFILE["maxRuntimeSeconds"] <= 3300
):
    raise ValueError("Invalid configured Modal resource profile")

root = Path(__file__).resolve().parents[1]
wheel = "cellweave_contracts-0.1.0-py3-none-any.whl"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.13.1",
        "pandas==2.2.3",
        "anndata==0.10.9",
        "pydantic==2.6.4",
        "structlog==24.4.0",
        "fastapi==0.115.12",
        "modal==0.77.0",
    )
    .add_local_file(root / "vendor" / wheel, "/opt/contracts/" + wheel, copy=True)
    .run_commands("pip install --no-deps /opt/contracts/" + wheel)
    .add_local_python_source("modal_app", "cellweave_science")
    .env(
        {
            "CELLWEAVE_ENVIRONMENT": ENVIRONMENT,
            "CELLWEAVE_MODAL_PROFILE_JSON": json.dumps(PROFILE),
            "MODAL_APP_NAME": os.environ.get("MODAL_APP_NAME", "cellweave-science"),
        }
    )
)
app = modal.App(os.environ.get("MODAL_APP_NAME", "cellweave-science"), image=image)
# Must be explicitly created by the operator in the corresponding Modal environment.
registry = modal.Dict.from_name("cellweave-phase4-submissions-" + ENVIRONMENT)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def status(record: dict[str, Any], job_id: str, state: str | None = None) -> dict[str, Any]:
    value = {
        "jobId": job_id,
        "status": state or record.get("state", "UNKNOWN"),
        "startedAt": record["acceptedAt"],
    }
    if record.get("providerJobId"):
        value["providerJobId"] = record["providerJobId"]
    return value


@app.function(
    cpu=PROFILE["cpu"],
    memory=PROFILE["memoryMiB"],
    timeout=PROFILE["maxRuntimeSeconds"],
    retries=0,
    max_inputs=1,
    include_source=False,
)
def execute_worker(payload: dict[str, Any]) -> dict[str, Any]:
    from cellweave_science.linux_resource import load_linux_resource

    resource = load_linux_resource()

    request = PreparedExecution.model_validate_json(json.dumps(payload), strict=True)
    budget = request.envelope.ceilings
    if (
        budget.resourceClass.value != PROFILE["resourceClass"]
        or budget.maxMemoryClass.value != PROFILE["resourceClass"]
        or budget.maxRuntimeSeconds > PROFILE["maxRuntimeSeconds"]
    ):
        raise ValueError("Unsupported execution resource profile")
    with tempfile.TemporaryDirectory(prefix="cellweave-job-") as scratch:
        output = Path(scratch) / "result.json"
        # Native platform identity stays in the wrapper. The science child gets no control-plane secrets.
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "PYTHONPATH"}
        }
        environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.monotonic()
        with subprocess.Popen(
            [sys.executable, "-m", "cellweave_science.execution_child", str(output)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
        ) as child:
            assert child.stdin is not None
            child.stdin.write(request.model_dump_json(exclude_unset=True).encode())
            child.stdin.close()
            memory_integral = 0.0
            previous = time.monotonic()
            peak = 0
            while child.poll() is None:
                now = time.monotonic()
                if now - started >= budget.maxRuntimeSeconds:
                    child.kill()
                    child.wait()
                    raise ValueError("RESOURCE_CEILING")
                try:
                    lines = Path(f"/proc/{child.pid}/status").read_text().splitlines()
                    rss = next(
                        int(line.split()[1]) * 1024 for line in lines if line.startswith("VmRSS:")
                    )
                    peak = max(peak, rss)
                    memory_integral += rss / (1024 * 1024) * (now - previous)
                    if rss > PROFILE["memoryMiB"] * 1024 * 1024:
                        child.kill()
                        child.wait()
                        raise ValueError("OUT_OF_MEMORY_CEILING")
                except (FileNotFoundError, StopIteration):
                    pass
                previous = now
                time.sleep(0.05)
            if child.returncode != 0 or not output.is_file():
                raise ValueError("SCIENTIFIC_EXECUTION_FAILED")
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        return {
            "executionResult": json.loads(output.read_text()),
            "metrics": {
                "cpuSeconds": cpu,
                "memoryMegabyteSeconds": memory_integral,
                "peakMemoryBytes": max(peak, after.ru_maxrss * 1024),
            },
        }


@app.function(timeout=60, retries=0, include_source=False)
def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    prepared = PreparedExecution.model_validate_json(json.dumps(request["execution"]), strict=True)
    envelope = prepared.envelope
    if envelope.environment.value != ENVIRONMENT:
        raise ValueError("Execution environment mismatch")
    if (
        envelope.ceilings.resourceClass.value != PROFILE["resourceClass"]
        or envelope.ceilings.maxMemoryClass.value != PROFILE["resourceClass"]
        or envelope.ceilings.maxRuntimeSeconds > PROFILE["maxRuntimeSeconds"]
    ):
        raise ValueError("Unsupported execution resource profile")
    job_id = envelope.submissionId
    if request["jobId"] != job_id:
        raise ValueError("Logical submission identity mismatch")
    if datetime.fromisoformat(envelope.deadlineAt.replace("Z", "+00:00")) <= datetime.now(
        timezone.utc
    ):
        raise ValueError("Execution admission deadline exceeded")
    semantic = {
        "envelope": envelope.model_dump(mode="json"),
        "sourceChecksum": prepared.sourceChecksum,
        "files": sorted(
            [
                {k: v for k, v in file.model_dump(mode="json").items() if k != "downloadUrl"}
                for file in prepared.files
            ],
            key=lambda file: file["fileName"],
        ),
    }
    fingerprint = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    record = {"acceptedAt": utc_now(), "fingerprint": fingerprint, "state": "UNKNOWN"}
    claimed = registry.put(job_id, record, skip_if_exists=True)
    if not claimed:
        existing = registry[job_id]
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("Submission conflict")
        return dict(existing)
    # Independent, persistent cancel tombstone closes the pre-spawn race.
    if registry.get("cancel:" + job_id, False):
        record["state"] = "CANCELLED"
        registry[job_id] = record
        return record
    call = execute_worker.spawn(prepared.model_dump(mode="json", exclude_unset=True))
    record.update({"providerJobId": call.object_id, "state": "PENDING"})
    registry[job_id] = record
    return record


@app.function(timeout=60, include_source=False)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def api(payload: dict[str, Any]) -> Any:
    try:
        operation = payload.get("operation")
        if operation == "submit":
            request = payload["request"]
            # Dispatcher is an independent Modal execution; worker HTTP loss does not own its lifetime.
            record = dispatch.spawn(request).get(timeout=40)
            if not record.get("providerJobId"):
                return JSONResponse({"error": "RECONCILIATION_REQUIRED"}, status_code=503)
            return status(record, request["jobId"])
        job_id = payload["jobId"]
        if operation == "cancel":
            registry["cancel:" + job_id] = True
        record = registry.get(job_id)
        if not record:
            if operation == "cancel":
                return {"jobId": job_id, "status": "CANCELLED", "startedAt": utc_now()}
            return JSONResponse({"error": "RECONCILIATION_REQUIRED"}, status_code=503)
        provider_id = record.get("providerJobId")
        if payload.get("providerJobId") and payload["providerJobId"] != provider_id:
            return JSONResponse({"error": "IDENTITY_CONFLICT"}, status_code=409)
        if not provider_id:
            return status(
                record, job_id, "CANCELLED" if record.get("state") == "CANCELLED" else "UNKNOWN"
            )
        call = modal.FunctionCall.from_id(provider_id)
        try:
            result = call.get(timeout=0)
            record["state"] = "COMPLETED"
            registry[job_id] = record
            if operation == "result":
                return {
                    "jobId": job_id,
                    "providerJobId": provider_id,
                    "status": "COMPLETED",
                    **result,
                }
            return status(record, job_id)
        except modal.exception.FunctionTimeoutError:
            record["state"] = "TIMEOUT"
            registry[job_id] = record
            return status(record, job_id)
        except modal.exception.TimeoutError:
            if operation == "cancel":
                call.cancel()
                # The native cancel RPC acknowledges cancellation; unknown RPC errors never confirm it.
                record["state"] = "CANCELLED"
                registry[job_id] = record
            elif record.get("state") != "CANCELLED":
                record["state"] = "RUNNING"
            return status(record, job_id)
        except ValueError:
            if record.get("state") != "CANCELLED":
                record["state"] = "FAILED"
                registry[job_id] = record
            return status(record, job_id)
        except Exception:
            # Transport/provider uncertainty is not proof of terminal scientific failure.
            return JSONResponse({"error": "RECONCILIATION_REQUIRED"}, status_code=503)
    except (KeyError, TypeError, ValueError):
        return JSONResponse({"error": "INVALID_EXECUTION"}, status_code=422)
    except Exception:
        return JSONResponse({"error": "PROVIDER_UNAVAILABLE"}, status_code=503)
