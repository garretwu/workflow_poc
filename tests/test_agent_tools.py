import json

from langchain_core.messages import ToolMessage

from activities import agent as agent_activity


def _get_tool(tools, name):
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"tool {name} not found")


def test_fetch_vllm_logs_includes_oom_line():
    pre = agent_activity._collect_telemetry_sync("oom_pressure", "pre")
    tools = agent_activity._build_tools("oom_pressure", pre)
    tool = _get_tool(tools, "fetch_vllm_logs")
    payload = json.dumps({"window": "15m", "limit": 200})

    result = json.loads(tool.run(payload))

    assert any("CUDA out of memory" in line for line in result["lines"])


def test_fetch_system_logs_includes_oom_killer():
    pre = agent_activity._collect_telemetry_sync("oom_pressure", "pre")
    tools = agent_activity._build_tools("oom_pressure", pre)
    tool = _get_tool(tools, "fetch_system_logs")
    payload = json.dumps({"window": "30m", "limit": 200})

    result = json.loads(tool.run(payload))

    assert any("Out of memory" in line for line in result["lines"])


def test_fetch_system_logs_includes_ecc_topic():
    pre = agent_activity._collect_telemetry_sync("default", "pre")
    tools = agent_activity._build_tools("default", pre)
    tool = _get_tool(tools, "fetch_system_logs")
    payload = json.dumps({"topics": ["ecc"], "limit": 200})

    result = json.loads(tool.run(payload))

    assert any("uncorrectable ECC error" in line for line in result["lines"])


def test_fetch_prometheus_metrics_parses_samples():
    pre = agent_activity._collect_telemetry_sync("default", "pre")
    tools = agent_activity._build_tools("default", pre)
    tool = _get_tool(tools, "fetch_prometheus_metrics")

    result = json.loads(tool.run(json.dumps({"endpoint": "http://localhost:8000/metrics"})))

    assert result["samples"]
    assert any(sample["name"].startswith("vllm:iteration_tokens_total") for sample in result["samples"])
    assert "iteration_tokens_total" in result["raw_text"]


def test_apply_fix_simulation_updates_config():
    pre = agent_activity._collect_telemetry_sync("default", "pre")
    tools = agent_activity._build_tools("default", pre)
    tool = _get_tool(tools, "apply_fix_simulation")
    payload = {
        "current_config": pre["config"],
        "fix_plan": {"config_updates": {"max_num_batched_tokens": 4096}},
    }

    result = json.loads(tool.run(json.dumps(payload)))

    assert result["updated_config"]["max_num_batched_tokens"] == 4096
    assert any("max_num_batched_tokens" in change for change in result["applied_changes"])


def test_collect_post_telemetry_uses_updated_config():
    pre = agent_activity._collect_telemetry_sync("default", "pre")
    tools = agent_activity._build_tools("default", pre)
    tool = _get_tool(tools, "collect_post_telemetry")
    payload = {"updated_config": {**pre["config"], "max_model_len": 2048}}

    result = json.loads(tool.run(json.dumps(payload)))

    assert result["config"]["max_model_len"] == 2048


def test_validate_improvement_detects_improvement():
    pre = agent_activity._collect_telemetry_sync("oom_pressure", "pre")
    post = agent_activity._collect_telemetry_sync("oom_pressure", "post", pre["config"])
    tools = agent_activity._build_tools("oom_pressure", pre)
    tool = _get_tool(tools, "validate_improvement")
    payload = {"pre_telemetry": pre, "post_telemetry": post}

    result = json.loads(tool.run(json.dumps(payload)))

    assert result["improved"] is True
    assert "p95_latency_drop" in result["deltas"]


def test_sqlite_callback_handler_serializes_tool_message(monkeypatch):
    captured = []

    def _append_live_trace(workflow_id, trace_item):
        json.dumps(trace_item)
        captured.append((workflow_id, trace_item))

    monkeypatch.setattr(agent_activity.store, "append_live_trace", _append_live_trace)

    handler = agent_activity.SQLiteCallbackHandler("wf-123")
    output = ToolMessage(content="ok", tool_call_id="call-1", name="fetch_config")

    handler.on_tool_end(output, name="fetch_config")

    assert captured
    assert captured[0][0] == "wf-123"
    assert captured[0][1]["output"]["type"] == "tool"
    assert captured[0][1]["output"]["content"] == "ok"
