from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gui import api


def test_app_lifespan_initializes_db(monkeypatch):
    init_db_calls = []

    def _init_db() -> None:
        init_db_calls.append(True)

    monkeypatch.setattr(api.store, "init_db", _init_db)

    with TestClient(api.app):
        pass

    assert len(init_db_calls) == 1


@pytest.mark.anyio
async def test_run_detail_returns_stored_report(monkeypatch):
    report = {"workflow_id": "wf-1", "summary": {"improved": True, "root_causes": ["x"]}}

    monkeypatch.setattr(api.store, "get_report", lambda workflow_id: report)
    monkeypatch.setattr(api.store, "get_live_traces", lambda workflow_id: [{"tool": "llm_call"}])

    def _unexpected_temporal_call(limit: int):
        raise AssertionError("_list_temporal_runs should not be called when report exists")

    monkeypatch.setattr(api, "_list_temporal_runs", _unexpected_temporal_call)

    result = await api.run_detail("wf-1")

    assert result == report


@pytest.mark.anyio
async def test_run_detail_builds_partial_response_from_temporal(monkeypatch):
    monkeypatch.setattr(api.store, "get_report", lambda workflow_id: None)
    monkeypatch.setattr(api.store, "get_live_traces", lambda workflow_id: [{"tool": "fetch_config"}])

    async def _fake_temporal_runs(limit: int):
        return [{"workflow_id": "wf-1", "scenario": "default", "status": "RUNNING"}]

    monkeypatch.setattr(api, "_list_temporal_runs", _fake_temporal_runs)

    result = await api.run_detail("wf-1")

    assert result["workflow_id"] == "wf-1"
    assert result["scenario"] == "default"
    assert result["status"] == "RUNNING"
    assert result["agent_trace"] == [{"tool": "fetch_config"}]
    assert result["summary"] == {"improved": None, "root_causes": []}


@pytest.mark.anyio
async def test_run_detail_handles_magicmock_temporal_result(monkeypatch):
    monkeypatch.setattr(api.store, "get_report", lambda workflow_id: None)
    monkeypatch.setattr(api.store, "get_live_traces", lambda workflow_id: [])

    mock_runs = MagicMock()
    mock_runs.__iter__.return_value = iter([{"workflow_id": "wf-1", "status": "RUNNING"}])
    monkeypatch.setattr(api, "_list_temporal_runs", MagicMock(return_value=mock_runs))

    result = await api.run_detail("wf-1")

    assert result["workflow_id"] == "wf-1"
    assert result["status"] == "RUNNING"
    assert result["summary"] == {"improved": None, "root_causes": []}


@pytest.mark.anyio
async def test_run_detail_raises_404_when_workflow_missing(monkeypatch):
    monkeypatch.setattr(api.store, "get_report", lambda workflow_id: None)
    monkeypatch.setattr(api.store, "get_live_traces", lambda workflow_id: [])

    async def _fake_temporal_runs(limit: int):
        return [{"workflow_id": "wf-2", "status": "RUNNING"}]

    monkeypatch.setattr(api, "_list_temporal_runs", _fake_temporal_runs)

    with pytest.raises(HTTPException) as exc:
        await api.run_detail("wf-1")

    assert exc.value.status_code == 404


@pytest.mark.anyio
async def test_list_runs_merges_temporal_and_stored(monkeypatch):
    stored_runs = [
        {
            "workflow_id": "wf-1",
            "scenario": "default",
            "status": "COMPLETED",
            "generated_at": "2026-02-07T10:00:00+00:00",
            "improved": 1,
            "root_cause": "GPU memory pressure",
            "report_path": "./reports/latency_report.json",
        },
        {
            "workflow_id": "wf-2",
            "scenario": "oom_pressure",
            "status": "FAILED",
            "generated_at": "2026-02-07T08:00:00+00:00",
            "improved": 0,
            "root_cause": "unknown",
            "report_path": "./reports/latency_report_2.json",
        },
    ]
    temporal_runs = [
        {
            "workflow_id": "wf-1",
            "scenario": "default",
            "status": "RUNNING",
            "start_time": "2026-02-07T09:00:00+00:00",
            "end_time": None,
        },
        {
            "workflow_id": "wf-3",
            "scenario": "tokenizer_cpu_bottleneck",
            "status": "RUNNING",
            "start_time": "2026-02-07T07:00:00+00:00",
            "end_time": None,
        },
    ]

    monkeypatch.setattr(api.store, "list_runs", lambda limit: stored_runs)

    async def _fake_temporal_runs(limit: int):
        return temporal_runs

    monkeypatch.setattr(api, "_list_temporal_runs", _fake_temporal_runs)

    response = await api.list_runs(limit=10)
    runs_by_id = {row["workflow_id"]: row for row in response["runs"]}

    assert set(runs_by_id) == {"wf-1", "wf-2", "wf-3"}
    assert runs_by_id["wf-1"]["status"] == "COMPLETED"
    assert runs_by_id["wf-1"]["improved"] == 1
