"""LLM-as-judge scoring for one case's final answer.

Uses a cheaper model (Haiku) than the agent-under-test (opus, per Phase 3) --
judging against a known ground truth is a simpler rubric-classification task
than running the analysis itself, and this keeps repeated eval runs (the
N-run consistency pass in `runner.py`) cheap.
"""

from __future__ import annotations

import json
import logging
import re

import anthropic

from microbiome_agent.eval.types import CaseRunResult, EvalCase, JudgeResult

log = logging.getLogger(__name__)

JUDGE_MODEL = "claude-haiku-4-5-20251001"

_JUDGE_SYSTEM_PROMPT = """\
You are a strict scientific-reviewer judge for a microbiome-analysis agent's
answers. You are given the KNOWN ground truth for a test case (established by
the literature or by a planted synthetic signal, already verified), any
caveats the agent was expected to surface, and the agent's actual final
answer. Score the answer honestly and skeptically -- your job is to catch
overclaiming and missed caveats, not to be charitable.

Respond with ONLY a JSON object (no other text, no markdown fences), with
exactly these keys:
{
  "soundness": <int 1-5, does the stated conclusion match the ground truth
                without overclaiming causality or ignoring effect size?>,
  "overstatement": <bool, does the agent claim more certainty/significance
                    than the data supports?>,
  "caveat_correctness": <int 1-5, or null if there is no caveat to check for
                         this case -- did the agent surface the RIGHT caveat,
                         not just some caveat?>,
  "explanation": <string, 1-3 sentences justifying the scores>
}
"""


def _parse_judge_json(text: str) -> dict | None:
    """Parse the judge's JSON reply, tolerating the common failure mode where
    the model wraps it in a ```json ... ``` fence despite being told not to
    (observed live with Haiku: it ignored the "no markdown fences"
    instruction on a real run). Tries a plain parse first, then strips a
    fenced code block, then falls back to grabbing the first {...} span."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    braces = re.search(r"\{.*\}", text, re.DOTALL)
    if braces:
        try:
            return json.loads(braces.group(0))
        except json.JSONDecodeError:
            pass

    return None


def judge_case(case: EvalCase, run: CaseRunResult) -> JudgeResult:
    """Score one case run's final answer against its ground truth and notes."""
    client = anthropic.Anthropic()
    user_prompt = (
        f"CASE: {case.id}\n\n"
        f"GROUND TRUTH:\n{case.ground_truth}\n\n"
        f"EXPECTED CAVEATS (if any):\n{case.notes}\n\n"
        f"AGENT'S FINAL ANSWER:\n{run.final_answer}\n"
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        system=_JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "\n".join(
        block.text for block in response.content if hasattr(block, "text")
    ).strip()
    data = _parse_judge_json(text)
    if data is None:
        log.warning("Judge response for %s was not valid JSON: %.200s", case.id, text)
        return JudgeResult(
            soundness=1, overstatement=True, caveat_correctness=None,
            explanation=f"Judge output was not parseable JSON: {text[:200]}",
        )

    caveat = data.get("caveat_correctness")
    return JudgeResult(
        soundness=int(data.get("soundness", 1)),
        overstatement=bool(data.get("overstatement", True)),
        caveat_correctness=(int(caveat) if caveat is not None else None),
        explanation=str(data.get("explanation", "")),
    )
