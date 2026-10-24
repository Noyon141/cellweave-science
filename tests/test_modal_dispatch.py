"""Provider coordination tests use fake native handles; no cloud requests or deployment."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_execution import prepared


class Registry:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.cancel_on_claim = False

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        if self.cancel_on_claim:
            self.values["cancel:" + key] = True
        return True


def setup(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Registry, list[str]]:
    monkeypatch.setenv("CELLWEAVE_ENVIRONMENT", "development")
    monkeypatch.setenv(
        "CELLWEAVE_MODAL_PROFILE_JSON",
        json.dumps(
            {"resourceClass": "CPU_SMALL", "cpu": 1, "memoryMiB": 1024, "maxRuntimeSeconds": 300}
        ),
    )
    module = importlib.import_module("modal_app.phase4")
    registry = Registry()
    calls: list[str] = []

    def spawn(payload: dict[str, Any]) -> SimpleNamespace:
        calls.append(payload["envelope"]["submissionId"])
        return SimpleNamespace(object_id="fc-native-test-return-value")

    monkeypatch.setattr(module, "registry", registry)
    monkeypatch.setattr(module, "execute_worker", SimpleNamespace(spawn=spawn))
    return module, registry, calls


def test_submit_response_loss_resolves_same_external_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, calls = setup(monkeypatch)
    execution, _ = prepared(tmp_path)
    request = {
        "jobId": execution.envelope.submissionId,
        "execution": execution.model_dump(mode="json", exclude_unset=True),
    }
    first = module.dispatch.local(request)
    # A retry may receive refreshed capabilities, but cannot change scientific input identity.
    request["execution"]["files"][0]["downloadUrl"] = (
        "https://storage.example/refreshed?ephemeral=new"
    )
    second = module.dispatch.local(request)
    assert first["providerJobId"] == second["providerJobId"] == "fc-native-test-return-value"
    assert len(calls) == 1


@pytest.mark.parametrize("during_claim", [False, True])
def test_cancel_before_or_during_claim_prevents_expensive_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during_claim: bool
) -> None:
    module, registry, calls = setup(monkeypatch)
    execution, _ = prepared(tmp_path)
    if during_claim:
        registry.cancel_on_claim = True
    else:
        registry["cancel:" + execution.envelope.submissionId] = True
    response = module.dispatch.local(
        {
            "jobId": execution.envelope.submissionId,
            "execution": execution.model_dump(mode="json", exclude_unset=True),
        }
    )
    assert response["state"] == "CANCELLED"
    assert "providerJobId" not in response
    assert calls == []


def test_submission_conflict_and_expired_admission_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, calls = setup(monkeypatch)
    execution, _ = prepared(tmp_path)
    request = {
        "jobId": execution.envelope.submissionId,
        "execution": execution.model_dump(mode="json", exclude_unset=True),
    }
    module.dispatch.local(request)
    request["execution"]["sourceChecksum"] = "b" * 64
    with pytest.raises(ValueError, match="conflict"):
        module.dispatch.local(request)
    request["execution"]["envelope"]["deadlineAt"] = "2000-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="deadline"):
        module.dispatch.local(request)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "failure, expected",
    [(OSError("transport unavailable"), 503), (ValueError("SCIENTIFIC_EXECUTION_FAILED"), 200)],
)
def test_provider_uncertainty_is_not_terminal_failure(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, expected: int
) -> None:
    module, registry, _ = setup(monkeypatch)
    registry["job"] = {
        "acceptedAt": "2026-01-01T00:00:00Z",
        "providerJobId": "native-id",
        "state": "RUNNING",
    }

    def get(*, timeout: int) -> None:
        assert timeout == 0
        raise failure

    monkeypatch.setattr(module.modal.FunctionCall, "from_id", lambda _: SimpleNamespace(get=get))
    response = module.api.local(
        {"operation": "status", "jobId": "job", "providerJobId": "native-id"}
    )
    if expected == 503:
        assert response.status_code == 503
        assert registry["job"]["state"] == "RUNNING"
    else:
        assert response["status"] == "FAILED"


def test_provider_identity_conflict_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    module, registry, _ = setup(monkeypatch)
    registry["job"] = {
        "acceptedAt": "2026-01-01T00:00:00Z",
        "providerJobId": "native-id",
        "state": "RUNNING",
    }
    response = module.api.local(
        {"operation": "result", "jobId": "job", "providerJobId": "different-id"}
    )
    assert response.status_code == 409
