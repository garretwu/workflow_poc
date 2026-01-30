from __future__ import annotations

from typing import Any, Dict, List

from temporalio import activity


def _log_contains(logs: List[str], needle: str) -> bool:
    lowered = " ".join(logs).lower()
    return needle.lower() in lowered


@activity.defn
async def analyze_latency(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    metrics = telemetry["metrics"]
    logs = telemetry["logs"]
    root_causes: List[Dict[str, Any]] = []

    if (
        metrics["gpu_mem_util"] >= 0.95
        or _log_contains(logs, "oom")
        or _log_contains(logs, "cudamalloc")
    ):
        root_causes.append(
            {
                "id": "oom_pressure",
                "label": "GPU memory pressure and allocator churn",
                "evidence": [
                    f"gpu_mem_util={metrics['gpu_mem_util']}",
                    "log hint: OOM/allocator warnings",
                ],
            }
        )

    if metrics["cpu_util"] >= 0.90 and metrics["gpu_util"] <= 0.60:
        root_causes.append(
            {
                "id": "tokenizer_cpu_bottleneck",
                "label": "Tokenizer CPU bottleneck starving GPU",
                "evidence": [
                    f"cpu_util={metrics['cpu_util']}",
                    f"gpu_util={metrics['gpu_util']}",
                ],
            }
        )

    if metrics["queue_depth"] >= 60 and metrics["max_num_batched_tokens"] >= 8192:
        root_causes.append(
            {
                "id": "oversized_batch",
                "label": "Oversized batch inflating queue latency",
                "evidence": [
                    f"queue_depth={metrics['queue_depth']}",
                    f"max_num_batched_tokens={metrics['max_num_batched_tokens']}",
                ],
            }
        )

    if _log_contains(logs, "kv cache") or metrics["gpu_mem_util"] >= 0.88:
        root_causes.append(
            {
                "id": "kv_cache_pressure",
                "label": "KV cache pressure during prefill",
                "evidence": [
                    f"gpu_mem_util={metrics['gpu_mem_util']}",
                    "log hint: kv cache pressure/eviction",
                ],
            }
        )

    if not root_causes:
        root_causes.append(
            {
                "id": "unknown",
                "label": "Insufficient evidence for a specific cause",
                "evidence": ["metrics within expected thresholds"],
            }
        )

    symptoms = {
        "latency_ms": metrics["latency_ms"],
        "tokens_per_sec": metrics["tokens_per_sec"],
        "queue_depth": metrics["queue_depth"],
    }

    return {
        "root_causes": root_causes,
        "symptoms": symptoms,
        "notes": "Heuristic diagnosis derived from mocked telemetry and logs.",
    }
