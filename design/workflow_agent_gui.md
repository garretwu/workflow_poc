# Workflow Agent GUI (Web) Design

## Context
- Current system runs a Temporal workflow for latency incidents with an agent step that calls tools and returns structured artifacts.
- Need a web UI to visualize workflow execution, agent reasoning steps, tool calls, and outputs in near real time.

## Goals
- Live status for each workflow run (queued/running/completed/failed).
- Clear timeline of agent reasoning and tool calls (inputs/outputs, durations, tokens).
- Drill-down to view raw LLM exchanges and tool responses.
- Fast onboarding: single command to start UI + backend in dev.

## Non-goals
- Full Temporal Web replacement.
- Multi-tenant auth or production-grade RBAC in v1.
- Editable workflow inputs or modifying run state from UI.

## Users
- ML/infra engineers debugging latency runs.
- Developers validating LLM behavior and tool usage.

## UX Overview
Pages
1) Runs list
   - Table of recent runs: workflow_id, scenario, status, start/end, duration, improvement flag.
2) Run detail
   - Summary card (scenario, status, improvement, root cause, key metrics).
   - Timeline with nodes: collect_pre_telemetry, agent_step, generate_report.
   - Agent trace panel with expandable tool calls (input, output, latency).
   - LLM panel with messages, token usage, stop_reason, and JSON output.
   - Metrics delta panel (p50/p95/p99, queue depth, tokens/sec).

Navigation
- Left sidebar: Runs, Settings (optional).
- Run detail deep-links: `/runs/:workflow_id`.

## Data Sources
Primary
- Temporal workflow history for statuses and timings.
- Agent trace + LLM call stats captured in report or activity result.

Supporting
- SQLite store for run artifacts and UI queries.
- `./reports/latency_report.json` for final summary (also imported into SQLite).

## Architecture (proposed)
Backend
- Lightweight API server (FastAPI).
- Responsibilities:
  - Query Temporal for workflow runs and status.
  - Read agent_trace and LLM call stats from report artifacts.
  - Serve run detail in normalized JSON.
  - Provide streaming updates via SSE (Server-Sent Events).

Frontend
- Single-page app (React + Vite).
- Poll or subscribe to SSE for live updates.

## API Design (v1)
REST
- `GET /api/runs?limit=50` -> list runs.
- `GET /api/runs/{workflow_id}` -> full run detail.
- `GET /api/runs/{workflow_id}/events` -> SSE for run updates.

Run Detail Schema (draft)
```
{
  "workflow_id": "vllm-latency-...",
  "scenario": "oom_pressure",
  "status": "RUNNING|COMPLETED|FAILED",
  "start_time": "...",
  "end_time": "...",
  "summary": {
    "improved": true,
    "root_cause": "...",
    "latency": {"p95_before": 850, "p95_after": 300}
  },
  "timeline": [
    {"name": "collect_pre_telemetry", "start": "...", "end": "...", "status": "..."},
    {"name": "agent_step", "start": "...", "end": "...", "status": "..."},
    {"name": "generate_report", "start": "...", "end": "...", "status": "..."}
  ],
  "agent_trace": [
    {"type": "llm_call", "input": {"index": 1}, "output": {"input_tokens": 1477, "output_tokens": 355, "stop_reason": "tool_use"}},
    {"type": "tool_call", "tool": "fetch_vllm_logs", "input": {...}, "output": "..."}
  ],
  "llm_messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Capturing Agent Data
- Reuse `agent_trace` from `agent_step` and attach to report.
- Include `llm_call` usage stats (input/output tokens, stop_reason, payload size).
- Store LLM final JSON output and thinking blocks (if present) for UI display.

## Temporal Integration
- Use Temporal Python client in the API server to list workflow executions.
- Extract activity times from history for timeline.
- Map Temporal status -> UI status.

## Streaming Updates (SSE)
- Emit:
  - status changes (RUNNING -> COMPLETED)
  - new agent trace entries
  - report availability
- SSE payload:
```
{ "type": "agent_trace", "workflow_id": "...", "data": {...} }
```

## UI Components (v1)
- RunsTable
- RunSummaryCard
- TimelineView
- AgentTraceList (filter: tool calls vs LLM calls)
- LLMMessageViewer
- ThinkingBlockViewer
- MetricsDeltaChart (simple table or small chart)

## Visual Design Direction
- Typography: expressive, legible pairing (e.g., "Fraunces" for headings, "IBM Plex Sans" for body).
- Color: warm neutral base with teal/amber accents; avoid purple bias.
- Background: subtle gradient + soft grid texture to convey "systems" atmosphere.
- Motion: staggered reveal for timeline nodes, smooth expand/collapse for tool calls.
- Layout: asymmetrical two-column detail view (timeline + trace on left, LLM + metrics on right).

## Error Handling
- If report not present, show "in progress" and fetch latest trace from Temporal.
- If agent_trace missing, show fallback hint with raw activity outputs.

## Security/Config
- Local-only by default.
- Optional env vars:
  - `GUI_PORT` (default 5173)
  - `API_PORT` (default 8080)
  - `TEMPORAL_ADDRESS` (default localhost:7233)

## Dev Workflow
- `make gui` to run API + UI concurrently.
- `python src/gui/api.py` (FastAPI) + `npm run dev` (UI)

## Open Questions
- Which charts are preferred (sparkline vs table) for latency deltas?
- Any branding constraints (logo, palette) to align with?
