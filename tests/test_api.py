"""Unit tests for the Phase 5 FastAPI wrapper.

`AgentLoop.run()` is mocked so these tests exercise the HTTP layer (request
validation, response shape, error mapping) without spawning a real MCP
server subprocess or calling the Anthropic API.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from microbiome_agent.agent.trace import RunTrace
from microbiome_agent.api.app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_rejects_empty_question():
    resp = client.post("/analyze", json={"question": ""})
    assert resp.status_code == 422


def test_analyze_success_with_trace():
    fake_trace = RunTrace(model="claude-opus-4-5", question="q")
    fake_trace.turns = []  # no turns needed for this shape check

    async def _fake_run(self, question, server_params):
        self.last_trace = fake_trace
        return "3 taxa were significantly different."

    with patch("microbiome_agent.api.app.AgentLoop.run", new=_fake_run):
        resp = client.post("/analyze", json={"question": "Analyse the example dataset."})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "3 taxa were significantly different."
    assert body["trace"]["turns"] == 0
    assert body["trace"]["cost_usd"] == 0.0


def test_analyze_maps_runtime_error_to_422():
    async def _fake_run(self, question, server_params):
        raise RuntimeError("Agent exceeded the maximum of 20 turns without producing a final answer.")

    with patch("microbiome_agent.api.app.AgentLoop.run", new=_fake_run):
        resp = client.post("/analyze", json={"question": "Loop forever."})

    assert resp.status_code == 422
    assert "exceeded" in resp.json()["detail"]


def test_analyze_maps_unexpected_error_to_500():
    async def _fake_run(self, question, server_params):
        raise ValueError("boom")

    with patch("microbiome_agent.api.app.AgentLoop.run", new=_fake_run):
        resp = client.post("/analyze", json={"question": "Trigger a bug."})

    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]
