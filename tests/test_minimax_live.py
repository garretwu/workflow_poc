import json
import os
import re
from pathlib import Path

import pytest

from activities.agent import MinimaxChatModel


def _extract_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group(0))


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_AUTH_TOKEN") or not os.getenv("ANTHROPIC_BASE_URL"),
    reason="ANTHROPIC_AUTH_TOKEN and ANTHROPIC_BASE_URL must be set for live Minimax test",
)
def test_minimax_live_connection():
    llm = MinimaxChatModel(
        model=os.getenv("MINIMAX_MODEL", "minimax-m2.1"),
        api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
        base_url=os.environ["ANTHROPIC_BASE_URL"],
        temperature=0.0,
        max_tokens=128,
        timeout_s=30,
    )
    print(f"Minimax model: {llm.model}")
    print(f"Minimax base_url: {llm.base_url}")

    response = llm.invoke("Reply with the single word: ok")
    print(f"Minimax response content: {response.content}")
    if llm._last_payload:
        print("Minimax payload:", llm._last_payload)
    if llm._last_response:
        print("Minimax raw response:", llm._last_response)

    assert "ok" in response.content.lower()


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_AUTH_TOKEN") or not os.getenv("ANTHROPIC_BASE_URL"),
    reason="ANTHROPIC_AUTH_TOKEN and ANTHROPIC_BASE_URL must be set for live Minimax test",
)
def test_minimax_live_analysis_with_mock_data():
    llm = MinimaxChatModel(
        model=os.getenv("MINIMAX_MODEL", "minimax-m2.1"),
        api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
        base_url=os.environ["ANTHROPIC_BASE_URL"],
        temperature=0.2,
        max_tokens=1024,
        timeout_s=60,
    )

    telemetry_path = Path(__file__).resolve().parents[1] / "src" / "mock_data" / "telemetry_pre.json"
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry = {
        "metrics": {
            "latency_ms": telemetry["metrics"]["latency_ms"],
            "tokens_per_sec": telemetry["metrics"]["tokens_per_sec"],
            "queue_depth": telemetry["metrics"]["queue_depth"],
            "gpu_mem_util": telemetry["metrics"]["gpu_mem_util"],
            "cpu_util": telemetry["metrics"]["cpu_util"],
            "max_num_batched_tokens": telemetry["metrics"]["max_num_batched_tokens"],
        },
        "config": {
            "max_model_len": telemetry["config"]["max_model_len"],
            "max_num_batched_tokens": telemetry["config"]["max_num_batched_tokens"],
            "enable_paged_kv_cache": telemetry["config"]["enable_paged_kv_cache"],
            "gpu_memory_utilization": telemetry["config"]["gpu_memory_utilization"],
        },
        "logs": [
            "WARN scheduler queue depth exceeded 60",
            "WARN kv cache eviction rate high",
            "ERROR cudaMalloc retry, nearing OOM",
        ],
    }

    prompt = (
        "You are a latency incident fixer for vLLM. Analyze the telemetry and logs below. "
        "Return ONLY JSON with keys: root_causes (list of {id,label,evidence}), "
        "symptoms (latency_ms,tokens_per_sec,queue_depth), notes. "
        "No markdown fences.\n\n"
        f"telemetry={json.dumps(telemetry, separators=(",", ":"))}\n"
    )
    print("Minimax analysis prompt:", prompt)

    response = llm.invoke(prompt)
    print("Minimax analysis response:", response.content)
    if llm._last_payload:
        print("Minimax payload:", llm._last_payload)
    if llm._last_response:
        print("Minimax raw response:", llm._last_response)

    data = _extract_json(response.content)
    assert isinstance(data.get("root_causes"), list)
    assert "symptoms" in data
