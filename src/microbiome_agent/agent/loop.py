"""Agent loop: natural-language question → MCP tool calls → final answer.

Phase 3. Connects the Anthropic API to the Phase 2 MCP server. The loop feeds
the model a question, dispatches each tool call it makes to the MCP server,
feeds the results back, and repeats until the model produces a final text
answer.

No big framework — just the Anthropic Python SDK and the MCP client, so the
mechanics are transparent and testable.

Typical use::

    agent = AgentLoop()
    answer = asyncio.run(agent.run("Analyse the example dataset.", server_params))
    print(answer)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# System prompt                                                                #
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are a rigorous microbiome-analysis agent. When given a question about a
dataset, you use your tools to run a complete, statistically valid analysis and
then summarise your findings clearly.

Planning protocol
-----------------
Before calling any tool, briefly sketch your plan: which dataset to load, which
analyses to run, and what question each answers. Then execute that plan step by
step.

Statistical discipline
----------------------
- Differential abundance: judge features by q_value (BH-FDR-corrected), never
  by raw p_value. Report the FDR threshold you used (default 0.05).
- PERMANOVA / beta diversity: the smallest possible p-value is bounded by
  sample size, not effect size. When a group has fewer than 4 samples, call out
  this resolution limit explicitly — a non-significant result is
  uninterpretable under those conditions, not evidence of no effect.
- Alpha diversity: the tool reports per-sample Shannon values and group
  summaries. It does NOT test group differences. If you want to test whether
  diversity differs between groups, note that a Kruskal-Wallis or Mann-Whitney
  test would be the appropriate follow-up (not available in this toolset), and
  comment on whether the reported means look meaningfully different.
- Never confuse statistical significance with biological importance. Always
  report effect size (log2 fold change) alongside q-values.

Reporting
---------
Once you have run the relevant analyses, call generate_report to write a
markdown and HTML report to disk. Then give the user a concise plain-language
summary covering:
  - which taxa (if any) were significantly different between groups, and in
    which direction (log2 fold change sign)
  - whether overall community composition differs (beta diversity result)
  - alpha diversity observations
  - enrichment results if run
  - the path to the generated report file
  - any caveats (small sample sizes, borderline FDR, uninformative results)

"No significant difference was found" is a valid and important result — do not
overstate findings to appear more interesting.
"""

MAX_TURNS = 20  # hard cap — prevents runaway loops on unexpected model behaviour


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _mcp_tool_to_anthropic(tool: Any) -> dict:
    """Convert an MCP ToolInfo to the dict format expected by the Anthropic SDK."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema,
    }


def _extract_text(content: list) -> str:
    """Pull plain text out of an Anthropic response content list."""
    parts = [block.text for block in content if hasattr(block, "text")]
    return "\n".join(parts).strip()


def _mcp_content_to_str(result: Any) -> str:
    """Stringify an MCP CallToolResult so the model can read it."""
    prefix = "[tool error] " if result.isError else ""
    text = _content_list_to_str(result.content)
    return prefix + text


def _content_list_to_str(content: list) -> str:
    parts = []
    for item in content:
        if hasattr(item, "text"):
            parts.append(item.text)
        else:
            parts.append(str(item))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Agent loop                                                                   #
# --------------------------------------------------------------------------- #

class AgentLoop:
    """Run the microbiome-agent against a live MCP server.

    Parameters
    ----------
    model:
        Anthropic model string. Default ``"claude-opus-4-5"``.
    max_tokens:
        Maximum tokens the model may generate per turn. Default 4096.
    max_turns:
        Hard cap on tool-call rounds before the loop raises ``RuntimeError``.
        Default 20.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-5",
        max_tokens: int = 4096,
        max_turns: int = MAX_TURNS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self._anthropic = anthropic.Anthropic()

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    async def run(
        self,
        question: str,
        server_params: StdioServerParameters,
    ) -> str:
        """Ask *question* and return the model's final plain-text answer.

        Spawns the MCP server specified by *server_params* as a subprocess,
        connects to it over stdio, retrieves the tool list, then runs the
        agent loop until the model produces a final ``end_turn`` response or
        ``max_turns`` is exceeded.
        """
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                tools = [_mcp_tool_to_anthropic(t) for t in tools_response.tools]
                log.info("Connected to MCP server; %d tools available.", len(tools))
                return await self._loop(question, tools, session)

    # ---------------------------------------------------------------------- #
    # Internal loop (separated so tests can call it directly)                 #
    # ---------------------------------------------------------------------- #

    async def _loop(
        self,
        question: str,
        tools: list[dict],
        session: ClientSession,
    ) -> str:
        """Core agentic loop: call model → dispatch tools → repeat."""
        messages: list[dict] = [{"role": "user", "content": question}]

        for turn in range(1, self.max_turns + 1):
            log.debug("Turn %d: calling model.", turn)
            response = self._anthropic.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
            log.debug("stop_reason=%s  content_blocks=%d",
                      response.stop_reason, len(response.content))

            # Append the assistant's response to conversation history
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return _extract_text(response.content)

            if response.stop_reason != "tool_use":
                raise RuntimeError(
                    f"Unexpected stop_reason from model: {response.stop_reason!r}. "
                    "Expected 'end_turn' or 'tool_use'.")

            # Dispatch every tool call in this response and collect results
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                log.info(
                    "Tool call → %s(%s)",
                    block.name,
                    json.dumps(block.input)[:300],
                )
                mcp_result = await session.call_tool(block.name, block.input)
                result_str = _mcp_content_to_str(mcp_result)
                log.info("Tool result ← %.300s", result_str)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(
            f"Agent exceeded the maximum of {self.max_turns} turns without "
            "producing a final answer. Increase max_turns or inspect the loop.")
