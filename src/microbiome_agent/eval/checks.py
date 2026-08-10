"""Deterministic (LLM-free) checks against a `CaseRunResult`.

Each function here is a *builder*: it takes case-specific parameters and
returns a `CheckFn` closure (`CaseRunResult -> CheckOutcome`), so `eval/cases.py`
can compose a list of checks per case declaratively.

Important, hard-won detail: `differential_abundance` assigns `group_a`/
`group_b` by *alphabetical sort* of the group labels, not by the order a
caller happened to pass them in (e.g. "RA" sorts before "control" because
uppercase precedes lowercase in ASCII). `fixtures/generate_fixtures.py` hit
this directly while verifying planted signals, and every direction check
below reads the actual `group_a`/`group_b` values out of the tool's JSON
response rather than assuming a fixed sign -- never assume fold-change sign
maps to a group without checking which group is actually A vs B.
"""

from __future__ import annotations

import json

from microbiome_agent.eval.types import CaseRunResult, CheckFn, CheckOutcome, ToolCallRecord


def _last_call(run: CaseRunResult, tool_name: str) -> ToolCallRecord | None:
    for call in reversed(run.tool_calls):
        if call.name == tool_name:
            return call
    return None


def _parse_result(call: ToolCallRecord) -> dict:
    if call.is_error:
        raise ValueError(f"tool call {call.name!r} errored: {call.result_text}")
    return json.loads(call.result_text)


# --------------------------------------------------------------------------- #
# Tool-selection checks
# --------------------------------------------------------------------------- #
def expects_tool_calls(required: list[str]) -> CheckFn:
    """The named tools were all called, in this relative order (as a
    subsequence -- other calls, e.g. an exploratory `dataset_summary`, are
    allowed in between and don't fail the check)."""

    def check(run: CaseRunResult) -> CheckOutcome:
        actual = [c.name for c in run.tool_calls]
        it = iter(actual)
        passed = all(req in it for req in required)
        return CheckOutcome(
            "expects_tool_calls", passed, f"required={required}, actual={actual}"
        )

    return check


def only_tools_called(allowed: list[str]) -> CheckFn:
    """No tool outside `allowed` was called -- tests tool-selection restraint
    (e.g. a question that only needs `dataset_summary` shouldn't trigger a
    full differential-abundance run)."""

    def check(run: CaseRunResult) -> CheckOutcome:
        actual = {c.name for c in run.tool_calls}
        extra = actual - set(allowed)
        return CheckOutcome(
            "only_tools_called", len(extra) == 0,
            f"unexpected tools called: {sorted(extra)}" if extra else "",
        )

    return check


def no_tool_errors() -> CheckFn:
    def check(run: CaseRunResult) -> CheckOutcome:
        errored = [c.name for c in run.tool_calls if c.is_error]
        return CheckOutcome(
            "no_tool_errors", len(errored) == 0,
            f"errored calls: {errored}" if errored else "",
        )

    return check


def expects_tool_error(tool_name: str | None = None) -> CheckFn:
    """At least one call (optionally a specific tool) errored -- for cases
    that deliberately reference a bad handle/column."""

    def check(run: CaseRunResult) -> CheckOutcome:
        calls = run.tool_calls if tool_name is None else [
            c for c in run.tool_calls if c.name == tool_name
        ]
        errored = [c for c in calls if c.is_error]
        return CheckOutcome(
            "expects_tool_error", len(errored) > 0,
            f"{len(errored)} errored call(s) among {[c.name for c in calls]}",
        )

    return check


def expects_graceful_bad_input_handling(load_tool: str, risky_tool: str) -> CheckFn:
    """Accepts either of two valid ways an agent can handle a deliberately
    bad tool argument (e.g. a nonexistent group_column):

    - Reactive: it calls `risky_tool`, the call errors, and it recovers.
    - Proactive: it never calls `risky_tool` at all, having caught the
      problem some other way first (e.g. inspecting `dataset_summary`
      before attempting the analysis) -- arguably the better behavior, and
      what a real agent was observed doing on a live run of this case.

    Only fails if `load_tool` was never called, or if `risky_tool` was
    called with the bad input and did NOT error -- i.e. it silently
    "succeeded," which would mean a fabricated result on invalid input.
    """

    def check(run: CaseRunResult) -> CheckOutcome:
        if not any(c.name == load_tool for c in run.tool_calls):
            return CheckOutcome(
                "expects_graceful_bad_input_handling", False,
                f"{load_tool} was never called",
            )

        risky_calls = [c for c in run.tool_calls if c.name == risky_tool]
        if not risky_calls:
            return CheckOutcome(
                "expects_graceful_bad_input_handling", True,
                f"{risky_tool} never attempted (proactive avoidance) -- ok",
            )

        errored = [c for c in risky_calls if c.is_error]
        if errored:
            return CheckOutcome(
                "expects_graceful_bad_input_handling", True,
                f"{risky_tool} called and correctly errored (reactive recovery) -- ok",
            )
        return CheckOutcome(
            "expects_graceful_bad_input_handling", False,
            f"{risky_tool} was called with the bad input and did NOT error -- "
            "unexpected success on invalid input.",
        )

    return check


# --------------------------------------------------------------------------- #
# Differential-abundance checks
# --------------------------------------------------------------------------- #
def differential_feature_direction(
    feature_substring: str,
    higher_in_group: str,
    tool_name: str = "run_differential_abundance",
) -> CheckFn:
    """A significant feature (matched by substring) is reported with fold
    change pointing the right way, reading `group_a`/`group_b` from the
    tool's own response rather than assuming a fixed sign."""

    def check(run: CaseRunResult) -> CheckOutcome:
        call = _last_call(run, tool_name)
        if call is None:
            return CheckOutcome(
                "differential_feature_direction", False, f"{tool_name} was never called"
            )
        try:
            data = _parse_result(call)
        except ValueError as exc:
            return CheckOutcome("differential_feature_direction", False, str(exc))

        rows = data.get("significant_features", [])
        match = next((r for r in rows if feature_substring in r["feature"]), None)
        if match is None:
            return CheckOutcome(
                "differential_feature_direction", False,
                f"{feature_substring!r} not among significant_features: "
                f"{[r['feature'] for r in rows]}",
            )

        groups = data.get("groups", [])
        group_a, group_b = groups if len(groups) == 2 else (None, None)
        lfc = match["log2_fold_change"]
        if group_b == higher_in_group:
            ok = lfc > 0
        elif group_a == higher_in_group:
            ok = lfc < 0
        else:
            return CheckOutcome(
                "differential_feature_direction", False,
                f"{higher_in_group!r} not among reported groups {groups}",
            )
        return CheckOutcome(
            "differential_feature_direction", ok,
            f"{feature_substring}: log2fc={lfc:.3g}, group_a={group_a}, "
            f"group_b={group_b}, expected higher in {higher_in_group}",
        )

    return check


def no_significant_features(tool_name: str = "run_differential_abundance") -> CheckFn:
    def check(run: CaseRunResult) -> CheckOutcome:
        call = _last_call(run, tool_name)
        if call is None:
            return CheckOutcome(
                "no_significant_features", False, f"{tool_name} was never called"
            )
        try:
            data = _parse_result(call)
        except ValueError as exc:
            return CheckOutcome("no_significant_features", False, str(exc))
        n_sig = data.get("n_significant")
        return CheckOutcome("no_significant_features", n_sig == 0, f"n_significant={n_sig}")

    return check


# --------------------------------------------------------------------------- #
# Beta-diversity checks
# --------------------------------------------------------------------------- #
def beta_diversity_note_present(tool_name: str = "run_beta_diversity") -> CheckFn:
    """`run_beta_diversity` emits a literal `note` key when a group has
    fewer than 4 samples -- check for the field itself, not narrative text."""

    def check(run: CaseRunResult) -> CheckOutcome:
        call = _last_call(run, tool_name)
        if call is None:
            return CheckOutcome(
                "beta_diversity_note_present", False, f"{tool_name} was never called"
            )
        try:
            data = _parse_result(call)
        except ValueError as exc:
            return CheckOutcome("beta_diversity_note_present", False, str(exc))
        present = "note" in data
        return CheckOutcome(
            "beta_diversity_note_present", present, f"response keys={list(data.keys())}"
        )

    return check


def beta_diversity_significant(
    expect_significant: bool, tool_name: str = "run_beta_diversity"
) -> CheckFn:
    def check(run: CaseRunResult) -> CheckOutcome:
        call = _last_call(run, tool_name)
        if call is None:
            return CheckOutcome(
                "beta_diversity_significant", False, f"{tool_name} was never called"
            )
        try:
            data = _parse_result(call)
        except ValueError as exc:
            return CheckOutcome("beta_diversity_significant", False, str(exc))
        p = data.get("permanova", {}).get("p_value")
        if p is None:
            return CheckOutcome("beta_diversity_significant", False, "no permanova.p_value in response")
        ok = (p < 0.05) == expect_significant
        return CheckOutcome("beta_diversity_significant", ok, f"p_value={p:.3g}")

    return check


# --------------------------------------------------------------------------- #
# Enrichment checks
# --------------------------------------------------------------------------- #
def enrichment_set_significant(
    set_substring: str, expect_significant: bool = True, tool_name: str = "run_enrichment"
) -> CheckFn:
    def check(run: CaseRunResult) -> CheckOutcome:
        call = _last_call(run, tool_name)
        if call is None:
            return CheckOutcome(
                "enrichment_set_significant", False, f"{tool_name} was never called"
            )
        try:
            data = _parse_result(call)
        except ValueError as exc:
            return CheckOutcome("enrichment_set_significant", False, str(exc))
        rows = data.get("top_results", [])
        match = next((r for r in rows if set_substring in r["set"]), None)
        if match is None:
            return CheckOutcome(
                "enrichment_set_significant", False,
                f"{set_substring!r} not in top_results sets: {[r['set'] for r in rows]}",
            )
        ok = bool(match["significant"]) == expect_significant
        return CheckOutcome(
            "enrichment_set_significant", ok,
            f"{set_substring}: significant={match['significant']}, q={match['q_value']:.3g}",
        )

    return check


# --------------------------------------------------------------------------- #
# Final-answer text checks (coarse pre-filter; LLM judge does the real work)
# --------------------------------------------------------------------------- #
def final_answer_contains(keywords: list[str], mode: str = "any") -> CheckFn:
    def check(run: CaseRunResult) -> CheckOutcome:
        text = run.final_answer.lower()
        hits = [k for k in keywords if k.lower() in text]
        passed = (len(hits) > 0) if mode == "any" else (len(hits) == len(keywords))
        return CheckOutcome(
            "final_answer_contains", passed, f"matched keywords: {hits}" if hits else "no keywords matched"
        )

    return check
