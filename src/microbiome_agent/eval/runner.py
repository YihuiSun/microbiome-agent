"""Orchestrates running every eval case against the live agent and computing
the Phase 4 scorecard metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from microbiome_agent.agent.loop import AgentLoop
from microbiome_agent.eval.cases import ALL_CASES
from microbiome_agent.eval.judge import judge_case
from microbiome_agent.eval.tracer import run_case_once
from microbiome_agent.eval.types import CaseRunResult, CheckOutcome, EvalCase, JudgeResult

log = logging.getLogger(__name__)

# Check names that speak to *tool selection* rather than result correctness --
# used by tool_selection_accuracy() below.
_TOOL_SELECTION_CHECK_NAMES = {"expects_tool_calls", "only_tools_called"}


@dataclass
class CaseReport:
    """Everything gathered for one case: every run, its check results, and
    one judge score (scored on the first run only, to keep judge calls --
    and cost -- from scaling with the consistency-check run count)."""

    case: EvalCase
    runs: list[CaseRunResult] = field(default_factory=list)
    check_results_per_run: list[list[CheckOutcome]] = field(default_factory=list)
    judge: Optional[JudgeResult] = None

    @property
    def first_run_all_checks_passed(self) -> bool:
        if not self.check_results_per_run:
            return False
        return all(c.passed for c in self.check_results_per_run[0])

    @property
    def consistent(self) -> bool:
        """True if every run (not just the first) passed every check --
        the run-to-run consistency signal."""
        if not self.check_results_per_run:
            return False
        return all(all(c.passed for c in run_checks) for run_checks in self.check_results_per_run)


async def run_case(case: EvalCase, agent: AgentLoop, n_runs: int) -> CaseReport:
    report = CaseReport(case=case)
    for i in range(n_runs):
        log.info("Running case %s (run %d/%d)...", case.id, i + 1, n_runs)
        run_result = await run_case_once(case, agent)
        report.runs.append(run_result)
        report.check_results_per_run.append([chk(run_result) for chk in case.checks])
    report.judge = judge_case(case, report.runs[0])
    return report


async def run_all(
    cases: Optional[list[EvalCase]] = None,
    n_runs: int = 3,
    agent_model: str = "claude-opus-4-5",
) -> list[CaseReport]:
    cases = cases if cases is not None else ALL_CASES
    agent = AgentLoop(model=agent_model)
    reports = []
    for case in cases:
        reports.append(await run_case(case, agent, n_runs))
    return reports


# --------------------------------------------------------------------------- #
# Aggregate metrics (per phase4_eval_design.md §6)
# --------------------------------------------------------------------------- #
def tool_selection_accuracy(reports: list[CaseReport]) -> float:
    """Fraction of cases whose tool-selection checks (which tools were
    called, and in the right relative order) passed on the first run."""
    if not reports:
        return float("nan")
    hits = 0
    for r in reports:
        if not r.check_results_per_run:
            continue
        relevant = [
            c for c in r.check_results_per_run[0] if c.name in _TOOL_SELECTION_CHECK_NAMES
        ]
        if relevant and all(c.passed for c in relevant):
            hits += 1
    return hits / len(reports)


def incorrect_claim_rate(reports: list[CaseReport]) -> float:
    """Fraction of cases where a deterministic check failed on the first run,
    or the LLM judge flagged overstatement."""
    if not reports:
        return float("nan")
    bad = sum(
        1
        for r in reports
        if not r.first_run_all_checks_passed or bool(r.judge and r.judge.overstatement)
    )
    return bad / len(reports)


def run_to_run_consistency(reports: list[CaseReport]) -> float:
    """Fraction of cases where ALL N runs passed every deterministic check."""
    if not reports:
        return float("nan")
    return sum(1 for r in reports if r.consistent) / len(reports)
