"""Step-level tracing and token/cost logging for the agent loop.

Phase 5. `agent/loop.py` already logs turn/tool activity via the `logging`
module, but that's ephemeral — nothing structured survives the run. This
module gives every `AgentLoop.run()` / `AgentLoop._loop()` call a `RunTrace`:
a small, serializable record of each model turn (latency, token usage,
estimated cost) and each tool call (name, input, latency, error flag), built
on the same "just enough structure to be useful" philosophy as the rest of
the project.

Deliberately independent of `eval/tracer.py`'s `_RecordingSession` proxy,
which records tool calls only, for a different purpose (deterministic
checks against a call trace, not cost/latency observability). The two could
be unified later; kept separate for now since `_RecordingSession` predates
this module and eval fixtures already depend on its exact shape.

Typical use::

    agent = AgentLoop()
    answer = await agent.run(question, server_params)
    print(agent.last_trace.summary())
    agent.last_trace.write_json("trace.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

# Per-MTok pricing, USD. Extend as new models are used. Unknown models fall
# back to a $0/token estimate rather than raising — cost logging should never
# be the reason a run fails.
MODEL_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    # model_name: (input $/MTok, output $/MTok)
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for one API call. Returns 0.0 for unpriced models."""
    in_rate, out_rate = MODEL_PRICING_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


@dataclass
class ToolCallStep:
    """One tool invocation within a turn."""

    name: str
    input: dict
    latency_s: float
    is_error: bool
    result_preview: str  # truncated, for human-readable logs — not the full payload


@dataclass
class ModelTurn:
    """One round-trip to the model: a `messages.create` call plus the tool
    calls (if any) it triggered."""

    turn: int
    stop_reason: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tool_calls: list[ToolCallStep] = field(default_factory=list)


@dataclass
class RunTrace:
    """Full trace of one `AgentLoop._loop()` execution."""

    model: str
    question: str
    turns: list[ModelTurn] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    error: str | None = None

    # ------------------------------------------------------------------ #
    # Recording (called from loop.py)                                       #
    # ------------------------------------------------------------------ #

    def start_turn(self) -> float:
        """Call before a model request; returns a start timestamp to pass
        to `record_turn`."""
        return time.monotonic()

    def record_turn(
        self,
        turn: int,
        stop_reason: str,
        start_time: float,
        input_tokens: int,
        output_tokens: int,
    ) -> ModelTurn:
        latency = time.monotonic() - start_time
        model_turn = ModelTurn(
            turn=turn,
            stop_reason=stop_reason,
            latency_s=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost_usd(self.model, input_tokens, output_tokens),
        )
        self.turns.append(model_turn)
        return model_turn

    def record_tool_call(
        self,
        turn: int,
        name: str,
        input_: dict,
        latency_s: float,
        is_error: bool,
        result_text: str,
    ) -> None:
        # Attach to the ModelTurn with this turn number (recorded just above
        # in the same iteration, so it always exists by the time this runs).
        target = next((t for t in self.turns if t.turn == turn), None)
        step = ToolCallStep(
            name=name,
            input=input_,
            latency_s=latency_s,
            is_error=is_error,
            result_preview=result_text[:200],
        )
        if target is not None:
            target.tool_calls.append(step)

    def finish(self, error: str | None = None) -> None:
        self.ended_at = time.monotonic()
        self.error = error

    # ------------------------------------------------------------------ #
    # Reporting                                                               #
    # ------------------------------------------------------------------ #

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.turns)

    @property
    def total_tool_calls(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns)

    @property
    def wall_time_s(self) -> float:
        if self.ended_at is None:
            return time.monotonic() - self.started_at
        return self.ended_at - self.started_at

    def summary(self) -> str:
        lines = [
            f"model={self.model}  turns={len(self.turns)}  "
            f"tool_calls={self.total_tool_calls}  wall_time={self.wall_time_s:.1f}s",
            f"tokens: {self.total_input_tokens} in / {self.total_output_tokens} out  "
            f"~${self.total_cost_usd:.4f}",
        ]
        if self.error:
            lines.append(f"error: {self.error}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "question": self.question,
            "wall_time_s": self.wall_time_s,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "error": self.error,
            "turns": [
                {
                    "turn": t.turn,
                    "stop_reason": t.stop_reason,
                    "latency_s": t.latency_s,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "cost_usd": t.cost_usd,
                    "tool_calls": [
                        {
                            "name": c.name,
                            "input": c.input,
                            "latency_s": c.latency_s,
                            "is_error": c.is_error,
                            "result_preview": c.result_preview,
                        }
                        for c in t.tool_calls
                    ],
                }
                for t in self.turns
            ],
        }

    def write_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
