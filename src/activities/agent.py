from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from langchain import agents
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import BaseTool, Tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr
from temporalio import activity

from activities.telemetry import _apply_scenario, _load_snapshot
from gui import store


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, BaseMessage):
        payload: Dict[str, Any] = {
            "type": value.type,
            "content": _json_safe(value.content),
        }
        name = getattr(value, "name", None)
        if name is not None:
            payload["name"] = name
        tool_call_id = getattr(value, "tool_call_id", None)
        if tool_call_id is not None:
            payload["tool_call_id"] = tool_call_id
        additional_kwargs = getattr(value, "additional_kwargs", None)
        if additional_kwargs:
            payload["additional_kwargs"] = _json_safe(additional_kwargs)
        return payload
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)


class SQLiteCallbackHandler(BaseCallbackHandler):
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        store.append_live_trace(
            self.workflow_id,
            {
                "tool": serialized.get("name"),
                "input": _json_safe(_parse_tool_input(input_str)),
                "log": "tool_call",
                "output": None,
            },
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        # We don't easily know WHICH tool ended here without tracking IDs, 
        # but for this POC we can just log the result.
        # Ideally we'd link it to the last tool start.
        # Using name from kwargs if available? kwargs usually has 'name'.
        name = str(kwargs.get("name", "unknown_tool"))
        store.append_live_trace(
            self.workflow_id,
            {
                "tool": name,
                "input": None,
                "log": "tool_result",
                "output": _json_safe(output),
            },
        )


class MinimaxChatModel(BaseChatModel):
    model: str
    api_key: str
    base_url: str
    workflow_id: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_s: int = 60

    _client: httpx.Client = PrivateAttr()
    _last_payload: Dict[str, Any] | None = PrivateAttr(default=None)
    _last_response: Dict[str, Any] | None = PrivateAttr(default=None)
    _call_stats: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.Client(timeout=self.timeout_s)

    @property
    def _llm_type(self) -> str:
        return "minimax"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url}

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.bind(tools=tools, **kwargs)

    def get_call_stats(self) -> List[Dict[str, Any]]:
        return list(self._call_stats)

    def _build_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        use_anthropic = "anthropic" in self.base_url
        system_messages = [
            message.content for message in messages if isinstance(message, SystemMessage)
        ]
        payload_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            if use_anthropic and isinstance(message, AIMessage) and message.tool_calls:
                blocks: List[Dict[str, Any]] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                for tool_call in message.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.get("id"),
                            "name": tool_call.get("name"),
                            "input": tool_call.get("args") or {},
                        }
                    )
                payload_messages.append({"role": "assistant", "content": blocks})
                continue
            if use_anthropic and isinstance(message, ToolMessage):
                payload_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": message.content,
                            }
                        ],
                    }
                )
            else:
                payload_messages.append(
                    {
                        "role": self._role_for_message(message),
                        "content": message.content,
                    }
                )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if use_anthropic and system_messages:
            payload["system"] = "\n".join(system_messages)
        if stop:
            payload["stop"] = stop
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = self._convert_tools(tools)
        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            payload["tool_choice"] = self._convert_tool_choice(tool_choice)
        extra = os.getenv("MINIMAX_EXTRA_BODY")
        if extra:
            try:
                payload.update(json.loads(extra))
            except json.JSONDecodeError:
                pass
        return payload

    def _call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = os.getenv("MINIMAX_CHAT_COMPLETIONS_URL")
        if not endpoint:
            if "anthropic" in self.base_url:
                endpoint = f"{self.base_url}/v1/messages"
            else:
                endpoint = f"{self.base_url}/v1/chat/completions"
        self._last_payload = payload
        if os.getenv("MINIMAX_TRACE") == "1":
            print("MINIMAX TRACE payload:", json.dumps(payload, indent=2))
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if "anthropic" in self.base_url:
            headers["x-api-key"] = self.api_key
            headers.setdefault("anthropic-version", os.getenv("ANTHROPIC_VERSION", "2023-06-01"))
        response = self._client.post(endpoint, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            if os.getenv("MINIMAX_TRACE") == "1":
                try:
                    print("MINIMAX TRACE error response:", response.text)
                except Exception:
                    pass
            raise
        data = response.json()
        self._last_response = data
        if os.getenv("MINIMAX_TRACE") == "1":
            print("MINIMAX TRACE response:", json.dumps(data, indent=2))
        thinking_text = None
        output_text = None
        content_blocks = data.get("content") if isinstance(data, dict) else None
        if isinstance(content_blocks, list):
            thinking_parts = [
                block.get("thinking")
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "thinking"
            ]
            thinking_text = "\n".join([part for part in thinking_parts if part]) or None
            text_parts = [
                block.get("text")
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            output_text = "".join([part for part in text_parts if part]) or None
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        stat = {
            "model": data.get("model", self.model) if isinstance(data, dict) else self.model,
            "endpoint": endpoint,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "stop_reason": data.get("stop_reason"),
            "message_count": len(payload.get("messages", [])),
            "payload_chars": len(json.dumps(payload)),
            "output_chars": len(output_text) if output_text else None,
            "thinking": thinking_text,
        }
        self._call_stats.append(stat)

        if self.workflow_id:
            store.append_live_trace(
                self.workflow_id,
                {
                    "tool": "llm_call",
                    "input": {"index": len(self._call_stats)},
                    "log": "llm_usage",
                    "output": stat,
                },
            )

        return data

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, **kwargs)
        data = self._call(payload)
        content = self._extract_content(data)
        tool_calls = self._extract_tool_calls(data)
        message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)], llm_output=data)

    def _extract_content(self, data: Dict[str, Any]) -> str:
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and message.get("content"):
                    return str(message["content"])
                if choice.get("text"):
                    return str(choice["text"])
        if "content" in data:
            content = data["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("text")
                ]
                if text_parts:
                    return "".join(text_parts)
                return ""
        raise ValueError("Unsupported MINIMAX response format")

    def _extract_tool_calls(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []
        content = data.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"tool_use", "tool_call"}:
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "args": block.get("input") or block.get("args") or {},
                        }
                    )
        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            for call in data["tool_calls"]:
                if isinstance(call, dict):
                    tool_calls.append(call)
        return tool_calls

    def _convert_tools(self, tools: Any) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict):
                if tool.get("type") == "function" and "function" in tool:
                    fn = tool["function"]
                    converted.append(
                        {
                            "name": fn.get("name"),
                            "description": fn.get("description", ""),
                            "input_schema": fn.get("parameters", {}),
                        }
                    )
                elif "name" in tool and "input_schema" in tool:
                    converted.append(tool)
                continue

            name = getattr(tool, "name", None)
            description = getattr(tool, "description", "")
            schema = None
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None:
                if hasattr(args_schema, "model_json_schema"):
                    schema = args_schema.model_json_schema()
                elif hasattr(args_schema, "schema"):
                    schema = args_schema.schema()
            if schema is None:
                schema = {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                }
            converted.append(
                {
                    "name": name,
                    "description": description or "",
                    "input_schema": schema,
                }
            )
        return converted

    def _convert_tool_choice(self, tool_choice: Any) -> Any:
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function" and "function" in tool_choice:
                return {"type": "tool", "name": tool_choice["function"].get("name")}
        return tool_choice

    @staticmethod
    def _role_for_message(message: BaseMessage) -> str:
        role = message.type
        if role == "human":
            return "user"
        if role == "ai":
            return "assistant"
        return role


class MultiArgTool(Tool):
    def _to_args_and_kwargs(
        self, tool_input: str | dict | list | tuple, tool_call_id: str | None
    ) -> tuple[tuple, dict]:
        if isinstance(tool_input, (list, tuple)):
            if len(tool_input) == 1:
                return (tool_input[0],), {}
            return (list(tool_input),), {}

        args, kwargs = BaseTool._to_args_and_kwargs(self, tool_input, tool_call_id)
        if len(args) + len(kwargs) == 1:
            if args:
                return args, {}
            return (list(kwargs.values())[0],), {}
        if kwargs and not args:
            return (kwargs,), {}
        if args and not kwargs:
            return (list(args),), {}
        if args and kwargs:
            merged = {"_args": list(args), **kwargs}
            return (merged,), {}
        return (), {}


def _load_mock_lines(file_name: str) -> List[str]:
    mock_dir = Path(__file__).resolve().parents[1] / "mock_data"
    path = mock_dir / file_name
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _parse_tool_input(raw: str) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        if "input" in raw and len(raw) == 1:
            return _parse_tool_input(raw.get("input"))
        return raw
    if not isinstance(raw, str):
        return {"query": raw}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"query": raw}


def _collect_telemetry_sync(
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


def _log_contains(logs: List[str], needle: str) -> bool:
    lowered = " ".join(logs).lower()
    return needle.lower() in lowered


def _analyze_latency_sync(telemetry: Dict[str, Any]) -> Dict[str, Any]:
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


def _propose_fix_sync(telemetry: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
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


def _apply_fix_simulation_sync(
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


def _percent_drop(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return round((before - after) / before, 3)


def _validate_improvement_sync(pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
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


def _ensure_agent_result(
    pre: Dict[str, Any],
    scenario: str,
    agent_output: Dict[str, Any],
    agent_trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    diagnosis = agent_output.get("diagnosis") or _analyze_latency_sync(pre)
    fix_plan = agent_output.get("fix_plan") or _propose_fix_sync(pre, diagnosis)
    fix_result = agent_output.get("fix_result")
    if not fix_result or "updated_config" not in fix_result:
        fix_result = _apply_fix_simulation_sync(pre["config"], fix_plan)
    post_telemetry = agent_output.get("post_telemetry")
    if not post_telemetry or "config" not in post_telemetry:
        post_telemetry = _collect_telemetry_sync(scenario, "post", fix_result["updated_config"])
    validation = agent_output.get("validation") or _validate_improvement_sync(pre, post_telemetry)
    trace = agent_output.get("agent_trace")
    if isinstance(trace, list):
        agent_trace.extend(trace)

    return {
        "diagnosis": diagnosis,
        "fix_plan": fix_plan,
        "fix_result": fix_result,
        "post_telemetry": post_telemetry,
        "validation": validation,
        "agent_trace": agent_trace,
    }


def _build_agent_prompt(scenario: str, pre: Dict[str, Any]) -> str:
    pre_json = json.dumps(pre, indent=2, sort_keys=True)
    return (
        "You are a latency incident fixer for vLLM. Use tools to gather evidence from logs, "
        "metrics, config, and system state before deciding on a fix.\n"
        "Return ONLY valid JSON with the following schema. Do not wrap in markdown. "
        "Keep it concise and valid JSON only.\n"
        "{\n"
        "  \"diagnosis\": {\n"
        "    \"root_causes\": [{\"id\": str, \"label\": str, \"evidence\": [str]}],\n"
        "    \"symptoms\": {\"latency_ms\": dict, \"tokens_per_sec\": number, \"queue_depth\": number},\n"
        "    \"notes\": str\n"
        "  },\n"
        "  \"fix_plan\": {\n"
        "    \"recommended_changes\": [str],\n"
        "    \"config_updates\": dict,\n"
        "    \"rationale\": [str]\n"
        "  },\n"
        "  \"fix_result\": {\n"
        "    \"updated_config\": dict,\n"
        "    \"applied_changes\": [str],\n"
        "    \"notes\": str\n"
        "  },\n"
        "  \"post_telemetry\": dict,\n"
        "  \"validation\": {\n"
        "    \"improved\": bool,\n"
        "    \"deltas\": dict,\n"
        "    \"notes\": str\n"
        "  },\n"
        "  \"agent_trace\": [dict]\n"
        "}\n"
        "After tool use, converge to a single primary root cause (1 item in root_causes) "
        "and provide a high-confidence fix plan. Limit evidence to at most 3 items.\n"
        "You may set fix_result, post_telemetry, validation, and agent_trace to null to keep the output short.\n"
        "You must call tools to fetch logs/metrics/config, apply a fix simulation, "
        "collect post telemetry, and validate improvement.\n"
        "Input:\n"
        f"scenario={scenario}\n"
        f"pre_telemetry={pre_json}\n"
    )


def _extract_json_payload(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise


def _repair_json_with_llm(text: str) -> Dict[str, Any] | None:
    try:
        llm = _build_minimax_model()
    except Exception:
        return None
    prompt = (
        "Fix the following content into valid JSON only. "
        "Return ONLY JSON, no markdown.\n\n"
        f"CONTENT:\n{text}\n"
    )
    try:
        response = llm.invoke(prompt)
        return _extract_json_payload(response.content)
    except Exception:
        return None


def _build_tools(scenario: str, pre: Dict[str, Any]) -> List[Tool]:
    def fetch_vllm_logs(raw: str) -> str:
        params = _parse_tool_input(raw)
        window = params.get("window", "15m")
        level = params.get("level", "info")
        limit = int(params.get("limit", 200))
        topics = params.get("topics") or []
        if isinstance(topics, str):
            topics = [topics]

        files: List[str] = []
        if scenario == "oom_pressure":
            files.append("vllm_cuda_oom_issue_26863.log")
        elif scenario == "tokenizer_cpu_bottleneck":
            files.append("vllm_issue_33369_logs.txt")
        else:
            files.append("vllm_issue_33369_logs.txt")

        if "network" in topics or "nccl" in topics:
            files.extend(["nccl_net_socket_truncated.log", "nccl_net_ib_errors.log"])

        lines: List[str] = []
        for file_name in files:
            lines.extend(_load_mock_lines(file_name))

        lines = lines[:limit]
        errors = [line for line in lines if "error" in line.lower() or "warn" in line.lower()]
        parsed = []
        for line in lines:
            lowered = line.lower()
            level_guess = "info"
            for level_name in ("error", "warn", "warning", "debug", "info"):
                if level_name in lowered:
                    level_guess = "warn" if level_name == "warning" else level_name
                    break
            parsed.append({"level": level_guess, "message": line})

        return json.dumps(
            {
                "source": "mock_data",
                "window": window,
                "level": level,
                "lines": lines,
                "parsed": parsed,
                "errors": errors,
            },
            indent=2,
        )

    def fetch_system_logs(raw: str) -> str:
        params = _parse_tool_input(raw)
        window = params.get("window", "30m")
        sources = params.get("sources") or ["journal"]
        if isinstance(sources, str):
            sources = [sources]
        limit = int(params.get("limit", 200))
        topics = params.get("topics") or []
        if isinstance(topics, str):
            topics = [topics]

        files: List[str] = []
        if scenario == "oom_pressure":
            files.append("linux_oom_killer.log")
        elif scenario == "tokenizer_cpu_bottleneck":
            files.append("linux_cpu_soft_lockup.log")
        else:
            files.extend(["nvidia_xid_errors.log", "linux_oom_killer.log"])

        if "gpu" in topics:
            files.append("nvidia_xid_errors.log")
        if "ecc" in topics:
            files.append("cuda_ecc_uncorrectable_error.log")

        lines: List[str] = []
        for file_name in files:
            lines.extend(_load_mock_lines(file_name))

        lines = lines[:limit]
        errors = [line for line in lines if "error" in line.lower() or "oom" in line.lower()]

        return json.dumps(
            {
                "source": "mock_data",
                "window": window,
                "sources": sources,
                "lines": lines,
                "errors": errors,
            },
            indent=2,
        )

    def fetch_prometheus_metrics(raw: str) -> str:
        params = _parse_tool_input(raw)
        endpoint = params.get("endpoint", "http://localhost:8000/metrics")
        lines = _load_mock_lines("vllm_metrics_prometheus.txt")
        samples = []
        for line in lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            metric_part = parts[0]
            value = parts[1]
            name = metric_part
            labels = {}
            if "{" in metric_part and metric_part.endswith("}"):
                name, label_blob = metric_part.split("{", 1)
                label_blob = label_blob.rstrip("}")
                for item in label_blob.split(","):
                    if not item:
                        continue
                    key, val = item.split("=", 1)
                    labels[key] = val.strip('"')
            samples.append({"name": name, "labels": labels, "value": value})

        return json.dumps(
            {
                "endpoint": endpoint,
                "raw_text": "\n".join(lines),
                "samples": samples,
            },
            indent=2,
        )

    def fetch_config(raw: str) -> str:
        _ = _parse_tool_input(raw)
        return json.dumps(
            {
                "config": pre["config"],
                "runtime": {
                    "batch_size": pre["metrics"]["batch_size"],
                    "max_num_batched_tokens": pre["metrics"]["max_num_batched_tokens"],
                    "max_num_seqs": pre["config"]["max_num_seqs"],
                },
                "version": "mocked",
            },
            indent=2,
        )

    def fetch_runtime_state(raw: str) -> str:
        _ = _parse_tool_input(raw)
        queue_depth = pre["metrics"]["queue_depth"]
        return json.dumps(
            {
                "queue_depth": queue_depth,
                "running": max(1, pre["config"]["max_num_seqs"] - queue_depth // 2),
                "waiting": max(0, queue_depth - 10),
                "scheduler_backlog_ms": queue_depth * 12,
                "worker_health": {
                    "healthy": queue_depth < 80,
                    "stuck": queue_depth >= 80,
                },
            },
            indent=2,
        )

    def fetch_gpu_stats(raw: str) -> str:
        _ = _parse_tool_input(raw)
        return json.dumps(
            {
                "gpus": [
                    {
                        "id": 0,
                        "util": pre["metrics"]["gpu_util"],
                        "mem_util": pre["metrics"]["gpu_mem_util"],
                        "mem_used_gb": round(pre["metrics"]["gpu_mem_util"] * 24, 2),
                        "mem_total_gb": 24,
                        "power_w": 285,
                        "temp_c": 74,
                    }
                ]
            },
            indent=2,
        )

    def analyze_metrics(raw: str) -> str:
        params = _parse_tool_input(raw)
        _ = params.get("prometheus_samples")
        metrics = pre["metrics"]
        hotspot = "balanced"
        if metrics["gpu_mem_util"] >= 0.92:
            hotspot = "gpu_memory"
        elif metrics["cpu_util"] >= 0.9:
            hotspot = "cpu"
        elif metrics["queue_depth"] >= 60:
            hotspot = "queue"
        latency_breakdown = {
            "prefill_ms": int(metrics["latency_ms"]["p95"] * 0.55),
            "decode_ms": int(metrics["latency_ms"]["p95"] * 0.35),
            "queue_ms": int(metrics["latency_ms"]["p95"] * 0.10),
        }
        return json.dumps(
            {
                "latency_breakdown": latency_breakdown,
                "derived_metrics": {
                    "p95_latency_ms": metrics["latency_ms"]["p95"],
                    "p99_latency_ms": metrics["latency_ms"]["p99"],
                    "queue_depth": metrics["queue_depth"],
                    "gpu_util": metrics["gpu_util"],
                    "gpu_mem_util": metrics["gpu_mem_util"],
                    "cpu_util": metrics["cpu_util"],
                },
                "hotspot": hotspot,
            },
            indent=2,
        )

    def apply_fix_simulation(raw: str) -> str:
        params = _parse_tool_input(raw)
        current_config = params.get("current_config") or pre["config"]
        fix_plan = params.get("fix_plan") or {}
        result = _apply_fix_simulation_sync(current_config, fix_plan)
        return json.dumps(result, indent=2)

    def collect_post_telemetry(raw: str) -> str:
        params = _parse_tool_input(raw)
        updated_config = params.get("updated_config") or pre["config"]
        result = _collect_telemetry_sync(scenario, "post", updated_config)
        return json.dumps(result, indent=2)

    def validate_improvement(raw: str) -> str:
        params = _parse_tool_input(raw)
        pre_telemetry = params.get("pre_telemetry") or pre
        post_telemetry = params.get("post_telemetry") or pre
        result = _validate_improvement_sync(pre_telemetry, post_telemetry)
        return json.dumps(result, indent=2)

    return [
        MultiArgTool(
            name="fetch_vllm_logs",
            func=fetch_vllm_logs,
            description=(
                "Fetch vLLM/service logs. Input JSON: "
                "{\"scenario\": str, \"window\": str, \"limit\": int, \"level\": str, \"topics\": [str]}. "
                "Use topics like 'nccl' or 'network' to include NCCL-related logs."
            ),
        ),
        MultiArgTool(
            name="fetch_system_logs",
            func=fetch_system_logs,
            description=(
                "Fetch system/kernel logs. Input JSON: "
                "{\"window\": str, \"sources\": [str], \"limit\": int, \"topics\": [str]}. "
                "Use topics like 'gpu' or 'ecc' to include driver/ECC lines."
            ),
        ),
        MultiArgTool(
            name="fetch_prometheus_metrics",
            func=fetch_prometheus_metrics,
            description="Fetch Prometheus /metrics text. Input JSON: {\"endpoint\": str}.",
        ),
        MultiArgTool(
            name="fetch_config",
            func=fetch_config,
            description="Fetch current model/runtime config. Input JSON: {}.",
        ),
        MultiArgTool(
            name="fetch_runtime_state",
            func=fetch_runtime_state,
            description="Fetch runtime queue/backlog state. Input JSON: {}.",
        ),
        MultiArgTool(
            name="fetch_gpu_stats",
            func=fetch_gpu_stats,
            description="Fetch GPU utilization and memory stats. Input JSON: {}.",
        ),
        MultiArgTool(
            name="analyze_metrics",
            func=analyze_metrics,
            description=(
                "Derive latency breakdown and hotspot. Input JSON: "
                "{\"pre_telemetry\": dict, \"prometheus_samples\": list}."
            ),
        ),
        MultiArgTool(
            name="apply_fix_simulation",
            func=apply_fix_simulation,
            description=(
                "Apply config changes in simulation. Input JSON: "
                "{\"current_config\": dict, \"fix_plan\": dict}."
            ),
        ),
        MultiArgTool(
            name="collect_post_telemetry",
            func=collect_post_telemetry,
            description=(
                "Collect post-change telemetry. Input JSON: "
                "{\"scenario\": str, \"updated_config\": dict}."
            ),
        ),
        MultiArgTool(
            name="validate_improvement",
            func=validate_improvement,
            description=(
                "Validate improvement. Input JSON: "
                "{\"pre_telemetry\": dict, \"post_telemetry\": dict}."
            ),
        ),
    ]


def _build_minimax_model(workflow_id: str | None = None) -> MinimaxChatModel:
    api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or ""
    if not api_key:
        raise RuntimeError("MINIMAX API key not configured (ANTHROPIC_AUTH_TOKEN missing).")
    base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv(
        "MINIMAX_BASE_URL", "https://api.minimax.chat"
    )
    model = os.getenv("MINIMAX_MODEL", "minimax-m2.1")

    return MinimaxChatModel(
        model=model,
        api_key=api_key,
        base_url=base_url,
        workflow_id=workflow_id,
        temperature=float(os.getenv("MINIMAX_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("MINIMAX_MAX_TOKENS", "4096")),
        timeout_s=int(os.getenv("MINIMAX_TIMEOUT_S", "60")),
    )


def _build_agent_executor(scenario: str, pre: Dict[str, Any], workflow_id: str | None = None) -> tuple[Any, MinimaxChatModel]:
    llm = _build_minimax_model(workflow_id)
    tools = _build_tools(scenario, pre)
    agent_executor = agents.create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a vLLM latency incident fixer. Use tools when needed and respond with JSON only."
        ),
    )
    return agent_executor, llm


def _format_agent_trace_from_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                trace.append(
                    {
                        "tool": tool_call.get("name"),
                        "input": tool_call.get("args"),
                        "log": "tool_call",
                        "output": None,
                    }
                )
        if isinstance(message, ToolMessage):
            trace.append(
                {
                    "tool": message.name or message.tool_call_id,
                    "input": None,
                    "log": "tool_result",
                    "output": message.content,
                }
            )
    return trace


@activity.defn
async def agent_step(scenario: str, pre: Dict[str, Any], workflow_id: str | None = None) -> Dict[str, Any]:
    agent_executor, llm = _build_agent_executor(scenario, pre, workflow_id)
    prompt = _build_agent_prompt(scenario, pre)
    
    callbacks = [SQLiteCallbackHandler(workflow_id)] if workflow_id else []
    
    try:
        raw_result = await asyncio.to_thread(
            agent_executor.invoke,
            {"messages": [{"role": "user", "content": prompt}]},
            config={"callbacks": callbacks}
        )
    except Exception as exc:
        fallback_trace = [
            {
                "tool": "agent",
                "input": None,
                "log": "LLM execution failed; fallback used.",
                "output": str(exc),
            }
        ]
        return _ensure_agent_result(pre, scenario, {}, fallback_trace)

    agent_trace: List[Dict[str, Any]] = []
    for idx, stat in enumerate(llm.get_call_stats(), start=1):
        agent_trace.append(
            {
                "tool": "llm_call",
                "input": {"index": idx},
                "log": "llm_usage",
                "output": stat,
            }
        )
    messages = raw_result.get("messages") if isinstance(raw_result, dict) else None
    if isinstance(messages, list):
        agent_trace.extend(_format_agent_trace_from_messages(messages))

    output_text = None
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                output_text = message.content
                break
            if isinstance(message, dict) and message.get("role") == "assistant":
                output_text = message.get("content")
                break
    if not output_text:
        output_text = "{}"

    try:
        agent_output = _extract_json_payload(output_text)
    except json.JSONDecodeError:
        repaired = _repair_json_with_llm(output_text)
        if repaired is not None:
            agent_output = repaired
            agent_trace.append(
                {
                    "tool": "agent",
                    "input": None,
                    "log": "Repaired invalid JSON via secondary LLM call.",
                    "output": None,
                }
            )
        else:
            agent_output = {
                "diagnosis": None,
                "fix_plan": None,
                "fix_result": None,
                "post_telemetry": None,
                "validation": None,
                "agent_trace": [
                    {
                        "tool": "agent",
                        "input": None,
                        "log": "Invalid JSON from LLM; fallback used.",
                        "output": output_text,
                    }
                ],
            }

    return _ensure_agent_result(pre, scenario, agent_output, agent_trace)
