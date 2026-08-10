"""CLI entry point for the Phase 4 evaluation harness.

Usage::

    # No API key / no live agent needed -- verifies fixtures, cases, and
    # check functions all agree with each other:
    python -m microbiome_agent.eval --dry-run

    # Live run against the real agent (needs ANTHROPIC_API_KEY + credits):
    python -m microbiome_agent.eval
    python -m microbiome_agent.eval --n-runs 1                  # cheaper, no consistency signal
    python -m microbiome_agent.eval --cases crc-fuso-enrichment,ra-prevotella-expansion
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from microbiome_agent.eval.cases import ALL_CASES, CASES_BY_ID
from microbiome_agent.eval.dry_run import run_dry_run
from microbiome_agent.eval.runner import (
    CaseReport,
    incorrect_claim_rate,
    run_all,
    run_to_run_consistency,
    tool_selection_accuracy,
)


def _print_scorecard(reports: list[CaseReport]) -> None:
    print()
    print(f"{'CASE':38} {'RUN 1':8} {'CONSISTENT':12} {'JUDGE SOUND':12} {'OVERSTATE':10}")
    print("-" * 84)
    for r in reports:
        run1 = "PASS" if r.first_run_all_checks_passed else "FAIL"
        consistent = "yes" if r.consistent else "no"
        soundness = str(r.judge.soundness) if r.judge else "-"
        overstate = ("yes" if r.judge.overstatement else "no") if r.judge else "-"
        print(f"{r.case.id:38} {run1:8} {consistent:12} {soundness:12} {overstate:10}")
        if not r.first_run_all_checks_passed:
            for c in r.check_results_per_run[0]:
                if not c.passed:
                    print(f"    FAILED CHECK: {c.name} -- {c.detail}")
        if r.judge and r.judge.explanation:
            print(f"    judge: {r.judge.explanation}")
    print("-" * 84)
    print(f"Tool-selection accuracy: {tool_selection_accuracy(reports):.0%}")
    print(f"Incorrect-claim rate:    {incorrect_claim_rate(reports):.0%}")
    print(f"Run-to-run consistency:  {run_to_run_consistency(reports):.0%}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 evaluation harness")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify fixtures/cases/checks with no LLM or API key (no live agent run).",
    )
    parser.add_argument(
        "--n-runs", type=int, default=3,
        help="Repeat each case this many times for the consistency metric (default 3).",
    )
    parser.add_argument(
        "--model", type=str, default="claude-opus-4-5",
        help="Agent-under-test model (default claude-opus-4-5, matching Phase 3).",
    )
    parser.add_argument(
        "--cases", type=str, default=None,
        help="Comma-separated case ids to run (default: all 14).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO-level logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cases = ALL_CASES
    if args.cases:
        ids = [c.strip() for c in args.cases.split(",")]
        unknown = [i for i in ids if i not in CASES_BY_ID]
        if unknown:
            print(f"Unknown case id(s): {unknown}\nKnown: {sorted(CASES_BY_ID)}", file=sys.stderr)
            sys.exit(1)
        cases = [CASES_BY_ID[i] for i in ids]

    if args.dry_run:
        ok = run_dry_run(cases)
        sys.exit(0 if ok else 1)

    reports = asyncio.run(run_all(cases, n_runs=args.n_runs, agent_model=args.model))
    _print_scorecard(reports)


if __name__ == "__main__":
    main()
