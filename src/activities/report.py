from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from temporalio import activity


def _percent_drop(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return round((before - after) / before, 3)


@activity.defn
async def validate_improvement(
    pre: Dict[str, Any], post: Dict[str, Any]
) -> Dict[str, Any]:
    pre_metrics = pre["metrics"]
    post_metrics = post["metrics"]

    p95_drop = _percent_drop(pre_metrics["latency_ms"]["p95"], post_metrics["latency_ms"]["p95"])
    p99_drop = _percent_drop(pre_metrics["latency_ms"]["p99"], post_metrics["latency_ms"]["p99"])
    queue_drop = _percent_drop(pre_metrics["queue_depth"], post_metrics["queue_depth"])
    tps_gain = None
    if pre_metrics["tokens_per_sec"] != 0:
        tps_gain = round(
            (post_metrics["tokens_per_sec"] - pre_metrics["tokens_per_sec"])
            / pre_metrics["tokens_per_sec"],
            3,
        )

    improved = False
    if p95_drop is not None and p99_drop is not None:
        improved = p95_drop >= 0.2 and p99_drop >= 0.2
        if queue_drop is not None:
            improved = improved and queue_drop >= 0.3

    return {
        "improved": improved,
        "deltas": {
            "p95_latency_drop": p95_drop,
            "p99_latency_drop": p99_drop,
            "queue_depth_drop": queue_drop,
            "tokens_per_sec_gain": tps_gain,
        },
        "notes": "Improvement threshold: >=20% latency reduction and >=30% queue drop.",
    }


@activity.defn
async def generate_report(
    scenario: str,
    pre: Dict[str, Any],
    post: Dict[str, Any],
    diagnosis: Dict[str, Any],
    fix_plan: Dict[str, Any],
    validation: Dict[str, Any],
    output_path: str,
    agent_trace: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    applied_config = post.get("config")
    if not applied_config:
        applied_config = dict(pre.get("config", {}))
        applied_config.update(fix_plan.get("config_updates", {}))
    post_metrics = post.get("metrics") or pre["metrics"]
    report = {
        "scenario": scenario,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "issue": f"p99 latency spike to {pre['metrics']['latency_ms']['p99']}ms",
            "root_causes": [cause["label"] for cause in diagnosis["root_causes"]],
            "resolution": "Applied mock configuration tuning in simulation.",
            "improved": validation["improved"],
        },
        "metrics": {
            "pre": pre["metrics"],
            "post": post_metrics,
            "deltas": validation["deltas"],
        },
        "evidence": {
            "logs": pre["logs"],
            "diagnosis_notes": diagnosis["notes"],
        },
        "recommendations": fix_plan["recommended_changes"],
        "rationale": fix_plan["rationale"],
        "applied_config": applied_config,
        "agent_trace": agent_trace or [],
        "timeline": [
            "collect_pre_telemetry",
            "agent_step",
            "generate_report",
        ],
        "report_path": output_path,
        "mocked": True,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report
