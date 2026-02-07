from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


DB_PATH = os.getenv("GUI_DB_PATH", "./reports/workflows.db")


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                workflow_id TEXT PRIMARY KEY,
                scenario TEXT,
                status TEXT,
                generated_at TEXT,
                improved INTEGER,
                root_cause TEXT,
                report_path TEXT,
                report_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT,
                trace_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def append_live_trace(workflow_id: str, trace_item: Dict[str, Any]) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO live_traces (workflow_id, trace_json) VALUES (?, ?)",
            (workflow_id, json.dumps(trace_item)),
        )
        conn.commit()


def get_live_traces(workflow_id: str) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT trace_json FROM live_traces WHERE workflow_id = ? ORDER BY id ASC",
            (workflow_id,),
        ).fetchall()
    return [json.loads(row["trace_json"]) for row in rows]


def upsert_report(report: Dict[str, Any]) -> None:
    workflow_id = report.get("workflow_id")
    if not workflow_id:
        return
    init_db()
    summary = report.get("summary", {})
    improved = 1 if summary.get("improved") else 0
    root_causes = summary.get("root_causes") or []
    root_cause = root_causes[0] if root_causes else None
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                workflow_id,
                scenario,
                status,
                generated_at,
                improved,
                root_cause,
                report_path,
                report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_id,
                report.get("scenario"),
                report.get("status"),
                report.get("generated_at"),
                improved,
                root_cause,
                report.get("report_path"),
                json.dumps(report),
            ),
        )
        conn.commit()


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT workflow_id, scenario, status, generated_at, improved, root_cause, report_path
            FROM runs
            ORDER BY generated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_report(workflow_id: str) -> Dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT report_json FROM runs WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
    if not row:
        return None
    return json.loads(row["report_json"])
