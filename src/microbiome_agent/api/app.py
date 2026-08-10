"""FastAPI service wrapper for the microbiome-agent.

Phase 5. Turns the CLI (`agent/__main__.py`) into a small HTTP service so the
agent is deployable, not just runnable from a terminal.

Request scoping — the registry question flagged in ROADMAP.md
----------------------------------------------------------------
Phase 2's MCP server (`mcp_server/server.py`) keeps its dataset/analysis
registries as module-level, in-memory dicts. That's fine for a single CLI
invocation but was flagged as "a real design question worth revisiting" for
a service, since a naive long-lived server process would leak state (or
worse, mix state) across concurrent requests.

The resolution: `AgentLoop.run()` already spawns the MCP server as a fresh
*subprocess* per call (see `agent/loop.py`, `StdioServerParameters` +
`stdio_client`) — the same pattern `eval/tracer.py` relies on for isolating
eval cases from each other. So each HTTP request below gets its own
subprocess with its own empty in-memory registries just by calling
`agent.run()` again; no extra session-management code needed here. The
trade-off is explicit: one MCP server subprocess (and one model
conversation) per request, not a shared pool — the right choice for a
demo/portfolio service where correctness and isolation matter more than
raw throughput. A future iteration could pool warm subprocesses per
request if latency became a problem.

Run::

    uvicorn microbiome_agent.api.app:app --reload

    curl -X POST localhost:8000/analyze \\
        -H 'Content-Type: application/json' \\
        -d '{"question": "Analyse the example dataset and report what taxa differ between groups."}'
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI, HTTPException
from mcp import StdioServerParameters
from pydantic import BaseModel, Field

from microbiome_agent.agent.loop import AgentLoop

log = logging.getLogger(__name__)

app = FastAPI(
    title="microbiome-agent",
    description=(
        "Autonomous microbiome differential-abundance and functional-"
        "enrichment agent with built-in statistical validation and "
        "reproducible reporting."
    ),
    version="0.1.0",
)

# Same subprocess-launch pattern as agent/__main__.py: point at the current
# interpreter so the spawned MCP server shares this process's venv/packages.
_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "microbiome_agent.mcp_server.server"],
)


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language analysis question.")
    model: str = Field("claude-opus-4-5", description="Anthropic model to use for this run.")


class TraceSummary(BaseModel):
    turns: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    wall_time_s: float


class AnalyzeResponse(BaseModel):
    answer: str
    trace: TraceSummary | None = None


@app.get("/health")
def health() -> dict:
    """Liveness check — does not touch the model or spawn a subprocess."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run one question through the agent end-to-end.

    Spawns a fresh MCP server subprocess for this request only (see module
    docstring) — concurrent requests do not share dataset/analysis state.
    """
    agent = AgentLoop(model=request.model)
    try:
        answer = await agent.run(request.question, _SERVER_PARAMS)
    except RuntimeError as exc:
        # Agent-side failure (e.g. max_turns exceeded) — not a server bug.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface unexpected failures as 500s, not crashes
        log.exception("Unhandled error running agent for question: %s", request.question)
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}") from exc

    trace = None
    if agent.last_trace is not None:
        t = agent.last_trace
        trace = TraceSummary(
            turns=len(t.turns),
            tool_calls=t.total_tool_calls,
            input_tokens=t.total_input_tokens,
            output_tokens=t.total_output_tokens,
            cost_usd=t.total_cost_usd,
            wall_time_s=t.wall_time_s,
        )

    return AnalyzeResponse(answer=answer, trace=trace)
