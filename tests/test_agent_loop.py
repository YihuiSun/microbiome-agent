"""Unit tests for the Phase 3 agent loop.

All external calls (Anthropic API, MCP server subprocess) are mocked so the
tests run instantly with no API key and no running server. The tests exercise
the loop's logic, not the underlying services.

Test cases
----------
1. Model answers on the first turn without calling any tool.
2. Model calls one tool, gets the result, then produces a final answer.
3. Model chains two tool calls before finishing (multi-turn).
4. Loop raises RuntimeError when max_turns is exceeded.
5. An MCP tool error is stringified with a ``[tool error]`` prefix and sent
   back to the model so it can handle it gracefully.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from microbiome_agent.agent.loop import (
    AgentLoop,
    _content_list_to_str,
    _extract_text,
    _mcp_content_to_str,
    _mcp_tool_to_anthropic,
)


# --------------------------------------------------------------------------- #
# Fake Anthropic / MCP objects                                                 #
# --------------------------------------------------------------------------- #

def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(
    name: str, input_: dict, id_: str = "call-1"
) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


def _end_turn(text: str) -> SimpleNamespace:
    """Fake Anthropic response that ends the conversation."""
    return SimpleNamespace(stop_reason="end_turn", content=[_text_block(text)])


def _tool_use_response(
    name: str, input_: dict, id_: str = "call-1"
) -> SimpleNamespace:
    """Fake Anthropic response that asks for one tool call."""
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[_tool_use_block(name, input_, id_)],
    )


def _mcp_ok(text: str) -> SimpleNamespace:
    return SimpleNamespace(isError=False, content=[SimpleNamespace(text=text)])


def _mcp_err(text: str) -> SimpleNamespace:
    return SimpleNamespace(isError=True, content=[SimpleNamespace(text=text)])


def _mcp_tool(name: str, description: str = "", schema: dict | None = None):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def _make_session(tool_names: list[str], call_results: dict[str, str]) -> AsyncMock:
    """Build a fake MCP ClientSession."""
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(
        return_value=SimpleNamespace(tools=[_mcp_tool(n) for n in tool_names])
    )

    async def _call_tool(name: str, args: dict):
        text = call_results.get(name, '{"ok": true}')
        return _mcp_ok(text)

    session.call_tool = AsyncMock(side_effect=_call_tool)
    return session


def run(coro):
    """Run a coroutine synchronously (safe outside an existing event loop)."""
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Helper function tests                                                         #
# --------------------------------------------------------------------------- #

class TestHelpers:
    def test_extract_text_single(self):
        assert _extract_text([_text_block("hello")]) == "hello"

    def test_extract_text_multiple(self):
        assert _extract_text([_text_block("a"), _text_block("b")]) == "a\nb"

    def test_extract_text_skips_tool_use_blocks(self):
        content = [_tool_use_block("foo", {}), _text_block("final")]
        assert _extract_text(content) == "final"

    def test_extract_text_empty(self):
        assert _extract_text([]) == ""

    def test_mcp_content_to_str_success(self):
        result = _mcp_ok('{"dataset_id": "ds-1"}')
        assert _mcp_content_to_str(result) == '{"dataset_id": "ds-1"}'

    def test_mcp_content_to_str_error_prefixed(self):
        result = _mcp_err("bad path")
        out = _mcp_content_to_str(result)
        assert out.startswith("[tool error]")
        assert "bad path" in out

    def test_mcp_tool_to_anthropic_shape(self):
        tool = _mcp_tool("run_differential_abundance", "Finds DA taxa.",
                         {"type": "object", "properties": {"dataset_id": {}}})
        converted = _mcp_tool_to_anthropic(tool)
        assert converted["name"] == "run_differential_abundance"
        assert converted["description"] == "Finds DA taxa."
        assert "properties" in converted["input_schema"]

    def test_mcp_tool_to_anthropic_empty_description(self):
        tool = _mcp_tool("ping", description="")
        converted = _mcp_tool_to_anthropic(tool)
        assert converted["description"] == ""

    def test_content_list_to_str_fallback(self):
        """Items without a .text attribute are str()-ed."""
        items = [SimpleNamespace(value=42)]  # no .text
        result = _content_list_to_str(items)
        assert "42" in result


# --------------------------------------------------------------------------- #
# Agent loop tests                                                              #
# --------------------------------------------------------------------------- #

class TestAgentLoop:
    def _agent(self, max_turns: int = 10) -> AgentLoop:
        return AgentLoop(model="claude-opus-4-5", max_tokens=512, max_turns=max_turns)

    # ------------------------------------------------------------------ #
    # 1. Single turn, no tool calls                                        #
    # ------------------------------------------------------------------ #
    def test_end_turn_without_tool_calls(self):
        """Model answers immediately — loop exits after one model call."""
        agent = self._agent()
        session = _make_session([], {})

        with patch.object(
            agent._anthropic.messages, "create",
            return_value=_end_turn("No tools needed."),
        ):
            result = run(agent._loop("What is 2+2?", [], session))

        assert result == "No tools needed."
        session.call_tool.assert_not_awaited()

    # ------------------------------------------------------------------ #
    # 2. One tool call then final answer                                   #
    # ------------------------------------------------------------------ #
    def test_one_tool_call_then_answer(self):
        """Model calls one tool, reads the result, then answers."""
        agent = self._agent()
        session = _make_session(
            ["load_example_dataset"],
            {"load_example_dataset": '{"dataset_id": "ds-1", "n_samples": 24}'},
        )

        responses = [
            _tool_use_response("load_example_dataset", {}),
            _end_turn("Dataset loaded: 24 samples."),
        ]
        idx = 0

        def _create(**kwargs):
            nonlocal idx
            r = responses[idx]; idx += 1; return r

        with patch.object(agent._anthropic.messages, "create", side_effect=_create):
            result = run(agent._loop("Load the example dataset.", [], session))

        assert "24" in result
        session.call_tool.assert_awaited_once_with("load_example_dataset", {})

    # ------------------------------------------------------------------ #
    # 3. Two chained tool calls                                            #
    # ------------------------------------------------------------------ #
    def test_chained_tool_calls(self):
        """Model chains two tool calls before producing a final answer."""
        agent = self._agent()
        session = _make_session(
            ["load_example_dataset", "run_differential_abundance"],
            {
                "load_example_dataset": '{"dataset_id": "ds-1"}',
                "run_differential_abundance": (
                    '{"analysis_id": "diff-1", "n_significant": 3}'
                ),
            },
        )

        responses = [
            _tool_use_response("load_example_dataset", {}, id_="c1"),
            _tool_use_response(
                "run_differential_abundance",
                {"dataset_id": "ds-1", "group_column": "study_condition"},
                id_="c2",
            ),
            _end_turn("Found 3 significantly different taxa."),
        ]
        idx = 0

        def _create(**kwargs):
            nonlocal idx
            r = responses[idx]; idx += 1; return r

        with patch.object(agent._anthropic.messages, "create", side_effect=_create):
            result = run(agent._loop("Analyse example dataset.", [], session))

        assert "3" in result
        assert session.call_tool.await_count == 2

    # ------------------------------------------------------------------ #
    # 4. max_turns guard                                                   #
    # ------------------------------------------------------------------ #
    def test_max_turns_raises(self):
        """Loop raises RuntimeError if the model never stops calling tools."""
        agent = self._agent(max_turns=3)
        session = _make_session(["ping"], {"ping": "pong"})

        with patch.object(
            agent._anthropic.messages, "create",
            return_value=_tool_use_response("ping", {}),
        ):
            with pytest.raises(RuntimeError, match="exceeded"):
                run(agent._loop("Ping forever.", [], session))

    # ------------------------------------------------------------------ #
    # 5. Tool error forwarded to model                                     #
    # ------------------------------------------------------------------ #
    def test_tool_error_forwarded_to_model(self):
        """An MCP tool error is prefixed with [tool error] and sent back."""
        agent = self._agent()
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[_mcp_tool("load_dataset")])
        )
        # The tool call returns an error
        session.call_tool = AsyncMock(return_value=_mcp_err("file not found"))

        captured_messages: list = []

        responses = [
            _tool_use_response("load_dataset", {"abundance_path": "/bad/path"}),
            _end_turn("Could not load dataset: file not found."),
        ]
        idx = 0

        def _create(**kwargs):
            nonlocal idx
            captured_messages.append(list(kwargs.get("messages", [])))
            r = responses[idx]; idx += 1; return r

        with patch.object(agent._anthropic.messages, "create", side_effect=_create):
            result = run(agent._loop("Load bad data.", [], session))

        # The second call's messages should contain the [tool error] result
        second_call_messages = captured_messages[1]
        tool_result_turn = second_call_messages[-1]
        assert tool_result_turn["role"] == "user"
        assert any(
            "[tool error]" in item["content"]
            for item in tool_result_turn["content"]
            if isinstance(item, dict) and "content" in item
        )
        assert "not found" in result.lower() or "could not" in result.lower()

    # ------------------------------------------------------------------ #
    # 6. Conversation history is built correctly                           #
    # ------------------------------------------------------------------ #
    def test_conversation_history_shape(self):
        """After one tool call, messages alternate user/assistant/user."""
        agent = self._agent()
        session = _make_session(
            ["ping"],
            {"ping": '{"pong": true}'},
        )

        captured_messages: list = []

        responses = [
            _tool_use_response("ping", {}, id_="c99"),
            _end_turn("Done."),
        ]
        idx = 0

        def _create(**kwargs):
            nonlocal idx
            captured_messages.append(list(kwargs.get("messages", [])))
            r = responses[idx]; idx += 1; return r

        with patch.object(agent._anthropic.messages, "create", side_effect=_create):
            run(agent._loop("Ping.", [], session))

        # First call: [user]
        assert captured_messages[0][0]["role"] == "user"
        # Second call: [user, assistant, user(tool_results)]
        roles = [m["role"] for m in captured_messages[1]]
        assert roles == ["user", "assistant", "user"]
