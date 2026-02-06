import os

from activities import agent as agent_activity


def test_minimax_model_default_and_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimax.chat")
    monkeypatch.delenv("MINIMAX_MODEL", raising=False)

    llm = agent_activity._build_minimax_model()

    assert llm.model == "minimax-m2.1"
    assert llm.base_url == os.environ["ANTHROPIC_BASE_URL"]
