"""Run a single eval case against the live agent, recording its tool-call trace.

`AgentLoop.run()` (Phase 3) only returns the final text answer -- exactly
right for the CLI, not enough for eval, which needs to check *which* tools
were called with *what* arguments and *what* came back. Rather than modify
`agent/loop.py`, this module wraps the MCP `ClientSession` in a thin recording
proxy and drives `AgentLoop._loop()` directly -- which the loop module's own
docstring says is "separated so tests can call it directly," so this is using
that seam for its intended purpose.
"""

from __future__ import annotations

import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from microbiome_agent.agent.loop import AgentLoop, _mcp_content_to_str, _mcp_tool_to_anthropic
from microbiome_agent.eval.types import CaseRunResult, EvalCase, ToolCallRecord

DEFAULT_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "microbiome_agent.mcp_server.server"],
)


class _RecordingSession:
    """Transparent proxy around a live `ClientSession` that logs every
    `call_tool` invocation (name, input, result text, error flag) before
    delegating to the real session. Every other attribute passes through
    untouched via `__getattr__`."""

    def __init__(self, session: ClientSession) -> None:
        self._session = session
        self.calls: list[ToolCallRecord] = []

    def __getattr__(self, item: str) -> Any:
        return getattr(self._session, item)

    async def call_tool(self, name: str, arguments: dict) -> Any:
        result = await self._session.call_tool(name, arguments)
        self.calls.append(
            ToolCallRecord(
                name=name,
                input=dict(arguments) if arguments else {},
                result_text=_mcp_content_to_str(result),
                is_error=bool(result.isError),
            )
        )
        return result


async def run_case_once(
    case: EvalCase,
    agent: AgentLoop,
    server_params: StdioServerParameters = DEFAULT_SERVER_PARAMS,
) -> CaseRunResult:
    """Run one case end-to-end against a fresh MCP server subprocess.

    A new subprocess per run means each case (and each repeat run, for the
    consistency metric) starts from clean, empty in-memory registries --
    matching the project's own "session-scoped state" design, and keeping
    cases from leaking into each other.
    """
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            tools = [_mcp_tool_to_anthropic(t) for t in tools_response.tools]

            proxy = _RecordingSession(session)
            final_answer = await agent._loop(case.question, tools, proxy)  # noqa: SLF001

    return CaseRunResult(case_id=case.id, tool_calls=proxy.calls, final_answer=final_answer)
