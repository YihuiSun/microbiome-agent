"""Dry-run verification: no LLM, no API key, no MCP subprocess.

This does NOT test the agent -- it tests that the fixtures, the case
definitions, and the deterministic check functions all agree with each other,
which is exactly what's verifiable without live API credits (the Phase 3
handoff flagged the same gap: "no live end-to-end test yet -- account needs
API credits"). It works by calling the MCP server's tool functions *directly*
as plain Python (the `@mcp.tool()` decorator returns the original callable
unmodified -- confirmed empirically before writing this module), building the
exact same JSON-shaped response a real MCP round-trip would produce, and
running each case's `checks` against that trace.

For cases with a `final_answer_contains` check, there is no real agent text to
test against here -- those checks are exercised against a synthetic
placeholder built from the case's own `ground_truth`/`notes`, which only
proves the check's parsing logic doesn't crash, not that a real agent's
phrasing would satisfy it. That gap is called out explicitly in the printed
report.

Run with::

    python -m microbiome_agent.eval --dry-run
"""

from __future__ import annotations

import json
import tempfile
from typing import Callable

from microbiome_agent.eval import cases
from microbiome_agent.eval.cases import ALL_CASES
from microbiome_agent.eval.types import CaseRunResult, CheckOutcome, EvalCase, ToolCallRecord
from microbiome_agent.mcp_server import server

TEXT_ONLY_CHECK_NAMES = {"final_answer_contains"}


def _record(name: str, arguments: dict, response: dict) -> ToolCallRecord:
    return ToolCallRecord(
        name=name, input=arguments, result_text=json.dumps(response), is_error=False
    )


def _error_record(name: str, arguments: dict, exc: Exception) -> ToolCallRecord:
    return ToolCallRecord(
        name=name, input=arguments, result_text=f"[tool error] {exc}", is_error=True
    )


# --------------------------------------------------------------------------- #
# One builder per case: replay the exact expected tool sequence for real,
# against the server's own functions.
# --------------------------------------------------------------------------- #
def _build_crc_fuso() -> list[ToolCallRecord]:
    r1 = server.load_example_dataset()
    calls = [_record("load_example_dataset", {}, r1)]
    r2 = server.run_differential_abundance(r1["dataset_id"], "study_condition")
    calls.append(_record(
        "run_differential_abundance",
        {"dataset_id": r1["dataset_id"], "group_column": "study_condition"}, r2))
    return calls


def _build_fixture_diff(case_id: str, abundance: str, metadata: str, group_col: str) -> list[ToolCallRecord]:
    r1 = server.load_dataset(abundance, metadata)
    calls = [_record("load_dataset", {"abundance_path": abundance, "metadata_path": metadata}, r1)]
    r2 = server.run_differential_abundance(r1["dataset_id"], group_col)
    calls.append(_record(
        "run_differential_abundance", {"dataset_id": r1["dataset_id"], "group_column": group_col}, r2))
    return calls, r1, r2  # type: ignore[return-value]


def _build_ra() -> list[ToolCallRecord]:
    abundance, metadata = cases.fixture_paths("ra")
    calls, _, _ = _build_fixture_diff("ra", abundance, metadata, "study_condition")
    return calls


def _build_t2d() -> list[ToolCallRecord]:
    abundance, metadata = cases.fixture_paths("t2d")
    calls, r1, r2 = _build_fixture_diff("t2d", abundance, metadata, "study_condition")
    feature_sets = {
        "scfa_producing_bacteria": ["Roseburia_intestinalis", "Eubacterium_rectale", "Faecalibacterium_prausnitzii"],
        "opportunistic_pathogens": ["Escherichia_coli", "Enterococcus_faecalis", "Klebsiella_pneumoniae"],
    }
    r3 = server.run_enrichment(r2["analysis_id"], feature_sets)
    calls.append(_record(
        "run_enrichment",
        {"differential_analysis_id": r2["analysis_id"], "feature_sets": feature_sets}, r3))
    return calls


def _build_crc_beta() -> list[ToolCallRecord]:
    r1 = server.load_example_dataset()
    calls = [_record("load_example_dataset", {}, r1)]
    r2 = server.run_beta_diversity(r1["dataset_id"], "study_condition", seed=1)
    calls.append(_record(
        "run_beta_diversity", {"dataset_id": r1["dataset_id"], "group_column": "study_condition"}, r2))
    return calls


def _build_cdiff() -> list[ToolCallRecord]:
    abundance, metadata = cases.fixture_paths("cdiff")
    r1 = server.load_dataset(abundance, metadata)
    calls = [_record("load_dataset", {"abundance_path": abundance, "metadata_path": metadata}, r1)]
    r2 = server.run_alpha_diversity(r1["dataset_id"], "study_condition")
    calls.append(_record(
        "run_alpha_diversity", {"dataset_id": r1["dataset_id"], "group_column": "study_condition"}, r2))
    return calls


def _build_obesity() -> list[ToolCallRecord]:
    abundance, metadata = cases.fixture_paths("obesity")
    calls, _, _ = _build_fixture_diff("obesity", abundance, metadata, "study_condition")
    return calls


def _build_null() -> list[ToolCallRecord]:
    abundance, metadata = cases.fixture_paths("null")
    calls, _, _ = _build_fixture_diff("null", abundance, metadata, "study_condition")
    return calls


def _build_small_n() -> list[ToolCallRecord]:
    abundance, metadata = cases.fixture_paths("smalln")
    r1 = server.load_dataset(abundance, metadata)
    calls = [_record("load_dataset", {"abundance_path": abundance, "metadata_path": metadata}, r1)]
    r2 = server.run_beta_diversity(r1["dataset_id"], "study_condition", seed=8)
    calls.append(_record(
        "run_beta_diversity", {"dataset_id": r1["dataset_id"], "group_column": "study_condition"}, r2))
    return calls


def _build_fdr_trap() -> list[ToolCallRecord]:
    abundance, metadata = cases.fixture_paths("fdrtrap")
    calls, _, _ = _build_fixture_diff("fdrtrap", abundance, metadata, "study_condition")
    return calls


def _build_summary_only() -> list[ToolCallRecord]:
    r1 = server.load_example_dataset()
    calls = [_record("load_example_dataset", {}, r1)]
    r2 = server.dataset_summary(r1["dataset_id"])
    calls.append(_record("dataset_summary", {"dataset_id": r1["dataset_id"]}, r2))
    return calls


def _build_full_pipeline(dataset_id: str, abundance: str | None, metadata: str | None, title: str) -> list[ToolCallRecord]:
    calls = []
    if abundance is None:
        r1 = server.load_example_dataset()
        calls.append(_record("load_example_dataset", {}, r1))
    else:
        r1 = server.load_dataset(abundance, metadata)
        calls.append(_record("load_dataset", {"abundance_path": abundance, "metadata_path": metadata}, r1))

    r2 = server.run_differential_abundance(r1["dataset_id"], "study_condition")
    calls.append(_record(
        "run_differential_abundance", {"dataset_id": r1["dataset_id"], "group_column": "study_condition"}, r2))
    r3 = server.run_alpha_diversity(r1["dataset_id"], "study_condition")
    calls.append(_record(
        "run_alpha_diversity", {"dataset_id": r1["dataset_id"], "group_column": "study_condition"}, r3))
    r4 = server.run_beta_diversity(r1["dataset_id"], "study_condition", seed=1)
    calls.append(_record(
        "run_beta_diversity", {"dataset_id": r1["dataset_id"], "group_column": "study_condition"}, r4))

    out_dir = tempfile.mkdtemp(prefix=f"eval-dryrun-{dataset_id}-")
    analysis_ids = [r2["analysis_id"], r3["analysis_id"], r4["analysis_id"]]
    r5 = server.generate_report(title, analysis_ids, out_dir)
    calls.append(_record(
        "generate_report", {"title": title, "analysis_ids": analysis_ids, "output_dir": out_dir}, r5))
    return calls


def _build_chained_full_pipeline() -> list[ToolCallRecord]:
    return _build_full_pipeline("example", None, None, "Example Dataset Analysis")


def _build_ra_chained_full_pipeline() -> list[ToolCallRecord]:
    abundance, metadata = cases.fixture_paths("ra")
    return _build_full_pipeline("ra", abundance, metadata, "RA Microbiome Analysis")


def _build_tool_error() -> list[ToolCallRecord]:
    """Demonstrates the reactive path for mcp-tool-error-recovery (calls
    run_differential_abundance with the bad column and catches the real
    ValueError). The check this feeds,
    expects_graceful_bad_input_handling, also accepts the proactive path
    (never calling run_differential_abundance at all, having caught the
    bad column via dataset_summary first) -- that's what a live agent run
    actually did; both are valid, this builder just exercises the other
    one since it's the one that touches the real error-forwarding path."""
    r1 = server.load_example_dataset()
    calls = [_record("load_example_dataset", {}, r1)]
    try:
        server.run_differential_abundance(r1["dataset_id"], "diagnosis_status")
    except ValueError as exc:
        calls.append(_error_record(
            "run_differential_abundance",
            {"dataset_id": r1["dataset_id"], "group_column": "diagnosis_status"}, exc))
    else:
        raise AssertionError("Expected run_differential_abundance to raise for a bad group_column.")
    return calls


def _build_ibd() -> list[ToolCallRecord]:
    abundance, metadata = cases.fixture_paths("ibd")
    calls, r1, r2 = _build_fixture_diff("ibd", abundance, metadata, "study_condition")
    feature_sets = {
        "facultative_anaerobes": ["Escherichia_coli", "Enterococcus_faecalis", "Klebsiella_pneumoniae"],
        "scfa_producing_obligate_anaerobes": ["Faecalibacterium_prausnitzii", "Roseburia_intestinalis", "Eubacterium_rectale"],
    }
    r3 = server.run_enrichment(r2["analysis_id"], feature_sets)
    calls.append(_record(
        "run_enrichment", {"differential_analysis_id": r2["analysis_id"], "feature_sets": feature_sets}, r3))
    return calls


_BUILDERS: dict[str, Callable[[], list[ToolCallRecord]]] = {
    "crc-fuso-enrichment": _build_crc_fuso,
    "ra-prevotella-expansion": _build_ra,
    "t2d-dysbiosis-enrichment": _build_t2d,
    "crc-beta-diversity-classification": _build_crc_beta,
    "cdiff-diversity-collapse": _build_cdiff,
    "obesity-mouse-caveat": _build_obesity,
    "no-signal-null-result": _build_null,
    "permanova-small-n": _build_small_n,
    "fdr-vs-raw-p-trap": _build_fdr_trap,
    "tool-selection-summary-only": _build_summary_only,
    "chained-full-pipeline": _build_chained_full_pipeline,
    "mcp-tool-error-recovery": _build_tool_error,
    "ibd-functional-enrichment": _build_ibd,
    "ra-chained-full-pipeline": _build_ra_chained_full_pipeline,
}


def _placeholder_final_answer(case: EvalCase) -> str:
    """A synthetic stand-in for what a real agent would write -- built from
    the case's own ground_truth/notes so text-based checks have *something*
    to run against. This is a mechanical smoke test of the check's parsing
    logic only; it does not validate that a real agent's phrasing would
    satisfy the check."""
    return f"{case.ground_truth}\n\n{case.notes}"


def run_dry_run(cases: list[EvalCase] | None = None) -> bool:
    """Run every case's builder + checks with no LLM/API involved. Returns
    True if every check passed, printing a case-by-case report either way."""
    cases = cases if cases is not None else ALL_CASES
    all_ok = True
    print(f"{'CASE':38} {'CHECK':32} RESULT")
    print("-" * 90)
    for case in cases:
        builder = _BUILDERS.get(case.id)
        if builder is None:
            print(f"{case.id:38} {'(no dry-run builder registered)':32} SKIP")
            continue
        try:
            tool_calls = builder()
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole run
            print(f"{case.id:38} {'(builder raised)':32} FAIL  {exc!r}")
            all_ok = False
            continue

        run_result = CaseRunResult(
            case_id=case.id, tool_calls=tool_calls,
            final_answer=_placeholder_final_answer(case))

        for check_fn in case.checks:
            outcome: CheckOutcome = check_fn(run_result)
            tag = "(text, mechanical only)" if outcome.name in TEXT_ONLY_CHECK_NAMES else ""
            status = "PASS" if outcome.passed else "FAIL"
            if not outcome.passed:
                all_ok = False
            print(f"{case.id:38} {outcome.name + ' ' + tag:32} {status}  {outcome.detail}")
    print("-" * 90)
    print("ALL DRY-RUN CHECKS PASSED" if all_ok else "SOME DRY-RUN CHECKS FAILED")
    return all_ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_dry_run() else 1)
