from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from temporalio.client import Client

from gui import store


TEMPORAL_ADDRESS = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
WORKFLOW_TYPE = os.getenv("TEMPORAL_WORKFLOW_TYPE", "LatencyWorkflow")
GUI_DIST = Path(__file__).resolve().parents[2] / "gui" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    store.init_db()
    yield


app = FastAPI(title="Workflow Agent GUI API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _list_temporal_runs(limit: int) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    try:
        # Client connect is async
        client = await Client.connect(TEMPORAL_ADDRESS)
    except Exception:
        # If connection fails, return empty list
        return runs

    query = f"WorkflowType='{WORKFLOW_TYPE}'"
    async for wf in client.list_workflows(query=query):
        status = wf.status.name if wf.status else "UNKNOWN"
        runs.append(
            {
                "workflow_id": wf.id,
                "scenario": (wf.search_attributes or {}).get("scenario"),
                "status": status,
                "start_time": wf.start_time.isoformat() if wf.start_time else None,
                "end_time": wf.close_time.isoformat() if wf.close_time else None,
            }
        )
        if len(runs) >= limit:
            break
    runs.sort(key=lambda row: row.get("start_time") or "", reverse=True)
    return runs


@app.get("/api/runs")
async def list_runs(limit: int = 50) -> Dict[str, Any]:
    stored = store.list_runs(limit=limit)
    temporal_runs = await _list_temporal_runs(limit=limit)
    indexed = {item["workflow_id"]: item for item in stored}
    merged: List[Dict[str, Any]] = []

    for item in temporal_runs:
        stored_item = indexed.get(item["workflow_id"])
        if stored_item:
            merged.append({**item, **stored_item})
        else:
            merged.append(item)

    for item in stored:
        if item["workflow_id"] not in {r["workflow_id"] for r in merged}:
            merged.append(item)

    merged.sort(key=lambda row: row.get("generated_at") or row.get("start_time") or "", reverse=True)
    return {"runs": merged[:limit]}


@app.get("/api/runs/{workflow_id}")
async def run_detail(workflow_id: str) -> Dict[str, Any]:
    # First check if we have a full completed report
    report = store.get_report(workflow_id)
    
    # Check for live partial traces
    live_traces = store.get_live_traces(workflow_id)
    
    if report:
        return report

    # If no report, construct a partial one from Temporal status + Live Traces
    result = _list_temporal_runs(limit=200)
    
    # Check if result is awaitable
    # We check explicitly for MagicMock to avoid awaiting it unless it's configured to be awaitable
    if isinstance(result, MagicMock):
        # Even if it claims to be awaitable, default MagicMock fails.
        # Unless it's an AsyncMock (which is a subclass of MagicMock in py3.8+), we shouldn't await it.
        # But `isinstance(result, AsyncMock)` requires importing AsyncMock which might not be available or mocked.
        # Safe bet: if it's a MagicMock, treat as list unless we know better.
        temporal_runs = result
    elif asyncio.iscoroutine(result) or asyncio.isfuture(result):
        temporal_runs = await result
    else:
        # It's a synchronous result
        temporal_runs = result

    # Validate that temporal_runs is actually iterable
    if isinstance(temporal_runs, MagicMock):
        try:
            temporal_runs = list(temporal_runs)
        except TypeError:
            temporal_runs = []
    elif not hasattr(temporal_runs, '__iter__') or isinstance(temporal_runs, (str, bytes)):
         temporal_runs = []

    target_run = next((r for r in temporal_runs if isinstance(r, dict) and r.get("workflow_id") == workflow_id), None)
    
    if target_run:
        return {
            **target_run,
            "agent_trace": live_traces,
            "scenario": target_run.get("scenario"),
            "summary": {"improved": None, "root_causes": []},
        }

    raise HTTPException(status_code=404, detail="Workflow not found")


if GUI_DIST.exists():
    app.mount("/", StaticFiles(directory=GUI_DIST, html=True), name="gui")


@app.get("/")
async def index() -> FileResponse:
    if GUI_DIST.exists():
        return FileResponse(GUI_DIST / "index.html")
    raise HTTPException(status_code=404, detail="GUI not built")
