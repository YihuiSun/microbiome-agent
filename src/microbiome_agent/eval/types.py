"""Shared data shapes for the Phase 4 eval harness.

Kept in one small module, in the same style as the rest of the project: plain
dataclasses, no behavior smuggled into `__init__`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ToolCallRecord:
    """One MCP tool call made by the agent during a case run.

    ``result_text`` is the same string the agent itself saw (including the
    ``"[tool error] "`` prefix on failure) -- captured via a recording proxy
    around the MCP session, not reconstructed after the fact.
    """

    name: str
    input: dict
    result_text: str
    is_error: bool


@dataclass
class CaseRunResult:
    """One end-to-end run of a single eval case against the live agent."""

    case_id: str
    tool_calls: list[ToolCallRecord]
    final_answer: str


@dataclass
class CheckOutcome:
    """Result of one deterministic check against a `CaseRunResult`."""

    name: str
    passed: bool
    detail: str = ""


CheckFn = Callable[[CaseRunResult], CheckOutcome]


@dataclass
class EvalCase:
    """One curated test case: a question with a known answer.

    Parameters
    ----------
    id:
        Short slug, e.g. ``"crc-fuso-enrichment"``.
    category:
        One of ``differential_abundance``, ``alpha_diversity``,
        ``beta_diversity``, ``enrichment``, ``chained``, ``negative_control``,
        ``edge_case``.
    question:
        The natural-language prompt fed to the agent.
    ground_truth:
        The known answer plus its literature source (or the planted signal,
        for synthetic/edge cases) -- shown to the LLM-judge, not the agent.
    notes:
        Caveats the agent is expected to surface, also shown to the judge.
    checks:
        Deterministic, LLM-free assertions run against the case's
        `CaseRunResult` (see `eval/checks.py`).
    """

    id: str
    category: str
    question: str
    ground_truth: str
    notes: str
    checks: list[CheckFn] = field(default_factory=list)


@dataclass
class JudgeResult:
    """LLM-as-judge score for one case's final answer."""

    soundness: int  # 1-5: does the conclusion match ground_truth without overclaiming?
    overstatement: bool  # does the agent claim more certainty than the data supports?
    caveat_correctness: Optional[int]  # 1-5, or None if this case has no `notes` caveat to check
    explanation: str
