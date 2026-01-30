from __future__ import annotations

from typing import Any, Dict, List

from temporalio import activity


def _apply_change(
    updates: Dict[str, Any],
    recommendations: List[str],
    rationale: List[str],
    current_config: Dict[str, Any],
    key: str,
    value: Any,
    flag: str,
    reason: str,
) -> None:
    if current_config.get(key) == value:
        return
    updates[key] = value
    if isinstance(value, bool):
        recommendations.append(flag if value else f"disable:{flag}")
    elif isinstance(value, (int, float)):
        recommendations.append(f"{flag}={value}")
    else:
        recommendations.append(f"{flag}={value}")
    rationale.append(reason)


@activity.defn
async def propose_fix(telemetry: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
    current_config = telemetry["config"]
    updates: Dict[str, Any] = {}
    recommendations: List[str] = []
    rationale: List[str] = []

    for cause in diagnosis["root_causes"]:
        cause_id = cause["id"]
        if cause_id == "oom_pressure":
            _apply_change(
                updates,
                recommendations,
                rationale,
                current_config,
                "gpu_memory_utilization",
                0.85,
                "--gpu-memory-utilization",
                "Lower headroom to avoid allocator churn.",
            )
            _apply_change(
                updates,
                recommendations,
                rationale,
                current_config,
                "max_model_len",
                3072,
                "--max-model-len",
                "Reduce KV cache pressure from long sequences.",
            )
            _apply_change(
                updates,
                recommendations,
                rationale,
                current_config,
                "max_num_batched_tokens",
                4096,
                "--max-num-batched-tokens",
                "Cap batch tokens to avoid OOM spikes.",
            )
            _apply_change(
                updates,
                recommendations,
                rationale,
                current_config,
                "enable_paged_kv_cache",
                True,
                "--enable-paged-kv-cache",
                "Paged KV reduces memory fragmentation.",
            )
        elif cause_id == "tokenizer_cpu_bottleneck":
            _apply_change(
                updates,
                recommendations,
                rationale,
                current_config,
                "tokenizer_pool_size",
                4,
                "--tokenizer-pool-size",
                "Parallelize tokenization across CPU cores.",
            )
            _apply_change(
                updates,
                recommendations,
                rationale,
                current_config,
                "async_output_processing",
                True,
                "--async-output-processing",
                "Reduce CPU stalls in output handling.",
            )
        elif cause_id == "oversized_batch":
            _apply_change(
                updates,
                recommendations,
                rationale,
                current_config,
                "max_num_batched_tokens",
                4096,
                "--max-num-batched-tokens",
                "Lower batch tokens to reduce queue latency.",
            )
            _apply_change(
                updates,
                recommendations,
                rationale,
                current_config,
                "max_num_seqs",
                48,
                "--max-num-seqs",
                "Limit concurrent sequences to stabilize p99.",
            )
        elif cause_id == "kv_cache_pressure":
            _apply_change(
                updates,
                recommendations,
                rationale,
                current_config,
                "enable_paged_kv_cache",
                True,
                "--enable-paged-kv-cache",
                "Paged KV cache reduces eviction churn.",
            )
            _apply_change(
                updates,
                recommendations,
                rationale,
                current_config,
                "max_model_len",
                3072,
                "--max-model-len",
                "Shorter context keeps KV cache within budget.",
            )

    if not updates:
        recommendations.append("no_change")
        rationale.append("No actionable config change from current signals.")

    return {
        "recommended_changes": recommendations,
        "config_updates": updates,
        "rationale": rationale,
    }


@activity.defn
async def apply_fix_simulation(
    current_config: Dict[str, Any], fix_plan: Dict[str, Any]
) -> Dict[str, Any]:
    updated = dict(current_config)
    updated.update(fix_plan.get("config_updates", {}))
    applied_changes = []
    for key, value in fix_plan.get("config_updates", {}).items():
        applied_changes.append(f"{key}: {current_config.get(key)} -> {value}")

    return {
        "updated_config": updated,
        "applied_changes": applied_changes,
        "notes": "Configuration updates applied in mock simulation only.",
    }
