from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from temporalio import activity

def _load_snapshot(phase: str) -> Dict[str, Any]:
    file_name = "telemetry_pre.json" if phase == "pre" else "telemetry_post.json"
    mock_dir = Path(__file__).resolve().parents[1] / "mock_data"
    with (mock_dir / file_name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_scenario(snapshot: Dict[str, Any], scenario: str, phase: str) -> Dict[str, Any]:
    metrics = snapshot["metrics"]
    logs = snapshot["logs"]

    if scenario == "oom_pressure":
        if phase == "pre":
            metrics["gpu_mem_util"] = min(0.99, metrics["gpu_mem_util"] + 0.07)
            metrics["latency_ms"]["p95"] += 250
            metrics["latency_ms"]["p99"] += 600
            metrics["tokens_per_sec"] -= 2200
            metrics["queue_depth"] += 25
            logs.append("ERROR cudaMalloc retry, nearing OOM")
            logs.append("WARN allocator fragmentation high")
        else:
            metrics["gpu_mem_util"] = max(0.88, metrics["gpu_mem_util"])
            metrics["latency_ms"]["p95"] += 60
            metrics["latency_ms"]["p99"] += 120
            metrics["tokens_per_sec"] -= 400
            logs.append("INFO memory headroom restored after config changes")
    elif scenario == "tokenizer_cpu_bottleneck":
        if phase == "pre":
            metrics["cpu_util"] = min(0.98, metrics["cpu_util"] + 0.33)
            metrics["gpu_util"] = max(0.45, metrics["gpu_util"] - 0.30)
            metrics["latency_ms"]["p95"] += 220
            metrics["latency_ms"]["p99"] += 500
            metrics["tokens_per_sec"] -= 3000
            logs.append("WARN tokenizer thread saturation detected")
            logs.append("INFO GPU underutilized during prefill")
        else:
            metrics["cpu_util"] = max(0.70, metrics["cpu_util"])
            metrics["gpu_util"] = min(0.80, metrics["gpu_util"] + 0.10)
            metrics["latency_ms"]["p99"] += 120
            metrics["tokens_per_sec"] -= 600
            logs.append("INFO tokenizer pool enabled")
    else:
        if phase == "pre":
            metrics["latency_ms"]["p99"] += 300
            metrics["queue_depth"] += 15
            logs.append("WARN kv cache pressure detected")
            logs.append("WARN batch tokens near limit")
        else:
            logs.append("INFO kv cache pressure relieved")
            logs.append("INFO batch tokens reduced")

    return snapshot


@activity.defn
async def collect_telemetry(
    scenario: str, phase: str, current_config: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    snapshot = _load_snapshot(phase)
    snapshot["scenario"] = scenario
    snapshot["phase"] = phase
    snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
    snapshot = _apply_scenario(snapshot, scenario, phase)

    if current_config:
        snapshot["config"] = current_config

    return snapshot
