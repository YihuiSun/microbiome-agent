# microbiome-agent — build roadmap

A living plan for building a microbiome-analysis AI agent, end to end. It exists
to keep the project pointed at its real purpose and to let any future working
session pick up instantly. Update the status markers as phases complete.

## Why this project exists (the strategy in one paragraph)

The goal is one artifact serving two jobs. Near-term: get into industry as a
comp-bio person who can also *ship robust software* — a rarer, stronger profile
than the typical academic bioinformatician. Longer-term: seed a pivot toward
ML/AI-engineering by building the same project the way an engineer would, with
real agent architecture and — most importantly — an **evaluation harness**.
Domain depth (microbiome) is the near-term edge; the engineering rigor and evals
are the pivot fuel. We don't choose between them; the build serves both.

Design principle throughout: **narrow and deep, tools first.** Correct, tested
analysis functions first; then an agent that orchestrates them; then an eval
harness that measures whether the agent's conclusions are actually right. Every
tool follows the same shape — clear typed signature, loud failure on bad input,
one tidy return value — because that discipline is what makes a tool safe for an
LLM to call and easy to evaluate.

## Current status

- [x] **Phase 0** — environment (venv on Python 3.12 from conda `py312`), repo,
      first commit pushed to github.com/YihuiSun/microbiome-agent.
- [x] **Phase 1** — analysis tools. DONE: `differential_abundance`, data
      `loader` + synthetic example dataset, diversity (alpha Shannon + beta
      Bray-Curtis/PERMANOVA), enrichment (ORA), report assembler. 51 passing
      tests, `bootstrap.sh`.
- [ ] **Phase 2** — MCP server
- [ ] **Phase 3** — agent loop
- [ ] **Phase 4** — evaluation harness
- [ ] **Phase 5** — engineering polish

(Phase 0's "hello-world tool use" learning exercise is still worth doing before
Phase 3 — see that phase.)

---

## Phase 0 — Setup & orientation  (~1 week — mostly done)

Goal: a working environment and a *feel* for the new layer (LLM tool use + MCP).

Environment is complete. Still valuable before the agent work: run the Anthropic
"hello world" tool-use example (one script where the model calls a single fake
function) and the MCP quickstart server, purely to see how the pieces move. No
real code yet.

Deliverable: a script that makes the model call a tool and return a result; a
running example MCP server.

---

## Phase 1 — Analysis tools as plain Python  (~2 weeks — complete)

Goal: correct, tested microbiome analysis on pre-computed abundance tables. No
AI yet — this is the comfort-zone phase that builds momentum and, crucially, the
ground truth the agent will later be judged against.

Data strategy: work from pre-computed abundance tables, not raw reads, to stay
laptop-scale. Synthetic example data is bundled for instant testing; real studies
come via a one-time R export (`scripts/export_curatedMetagenomicData.R`) that
drops published cohorts into `data/` in the same CSV format. Real published
case-control cohorts matter because they give known-answer ground truth for evals.

Tools (each: defensive function + known-answer tests + tidy return value):
- `differential_abundance` — per-feature test + FDR correction + log2 fold change. DONE.
- data `loader` — aligns an abundance table with sample metadata; feeds the tools. DONE.
- diversity — alpha (Shannon) and beta (Bray-Curtis + PERMANOVA), via
  `scikit-bio`. DONE.
- functional/pathway enrichment — ORA via Fisher's exact test + BH FDR. DONE
  (code-complete and tested; real runs still need a real pathway catalogue —
  e.g. KEGG/MetaCyc set memberships — which is a one-time data-sourcing task,
  not code. A rank-based GSEA variant is a natural later addition.)
- report assembler — tool outputs + figures into markdown + a self-contained
  HTML file (volcano, alpha boxplot, PCoA, enrichment bar). DONE.

Deliverable: a small, fully-tested analysis library. COMPLETE.

---

## Phase 2 — Wrap the tools as an MCP server  (~1 week)

Goal: expose each Phase 1 function as an MCP tool with a clear schema and a
strong docstring. The docstrings matter more than expected — they are how the
agent decides which tool to use, so this is where domain knowledge becomes
machine-usable. Use the Python MCP SDK (FastMCP).

New concepts: MCP protocol, tool schemas, structured tool I/O.

Deliverable: an MCP server that lists the tools and runs them when called.

---

## Phase 3 — The agent loop  (~2 weeks)

Goal: connect the LLM. A loop that takes a natural-language question, sends it to
the model with the MCP tools attached, lets the model call tools, feeds results
back, and repeats until it produces a final report.

Recommendation: write the loop yourself with the Anthropic SDK rather than a big
framework — at this stage understanding the mechanics is worth more than the
convenience, and it isn't much code. Invest real effort in the system prompt:
plan first, apply FDR, flag small sample sizes, don't overstate findings.

New concepts: the agent loop, tool-call handling, system-prompt design,
multi-step orchestration.

Deliverable (this is the demo): ask a question -> it runs an analysis -> returns
a report with figures. Everything after this turns the demo into a job-getting
project.

---

## Phase 4 — The evaluation harness  (~2–3 weeks — DO NOT SKIP)

Goal: measure whether the agent's scientific conclusions are correct. This is the
single highest-leverage part for the AI-eng pivot, and it's largely
domain-agnostic, so the skill transfers even though it's built on microbiome data.

Build: curate 8–15 test cases from published studies with known answers. Score at
two levels — deterministic checks (did it pick the right tool? apply FDR? flag the
right taxa?) and an LLM-as-judge check on the soundness of the biological
interpretation. Compute metrics across cases: tool-selection accuracy,
incorrect-claim rate, run-to-run consistency.

New concepts: eval design, ground-truth curation, LLM-as-judge, agent metrics.

Deliverable: `python eval.py` prints a scorecard. The sentence "I built an eval
harness that measures whether my agent's conclusions are scientifically correct"
is the one that earns AI-eng interviews.

---

## Phase 5 — Engineering polish  (~2–3 weeks)

Goal: the signals that read as engineering maturity to both audiences.

Build: step-level tracing + token/cost logging (observability); a Dockerfile; a
GitHub Actions workflow running the tests on every push (CI); a thin FastAPI
wrapper so it's a deployable service, not a notebook; a strong README with an
architecture diagram and example runs.

New concepts: observability, containerization, CI, serving an API.

Deliverable: a public repo someone can clone, run, and understand in five minutes.

---

## Timeline & sequencing

Roughly **2–3 months** at ~8–12 hrs/week. Demoable end-to-end agent by ~4–6
weeks (through Phase 3); the back half (Phases 4–5) is what converts it from a
demo into the portfolio piece that carries the pivot. The eval harness is the
part that's easiest to deprioritize and most important to keep — protect it.

## Resume framing (one project, two directions)

- Domain line: "autonomous microbiome differential-abundance and functional-
  enrichment agent with built-in statistical validation and reproducible
  reporting." (Lands the industry job now.)
- Engineering line: "MCP-based tool-orchestration agent with an automated
  evaluation harness measuring tool-selection accuracy and correctness;
  containerized with CI." (Seeds the pivot.)

## Working conventions (so every session stays consistent)

- Each tool: defensive function, loud failure on bad input, one tidy return
  value, known-answer tests.
- venv on Python 3.12 (sourced from conda `py312`); `which python` is the source
  of truth, not the prompt.
- Git rhythm: `git add . && git commit -m "..." && git push`. Never commit
  `.venv/`, `data/`, or generated reports (the assembler's output dir).
- Realistic expectation: this project gets you into industry now and makes the
  AI-eng move reachable later; it is not a one-step conversion, and that's fine.
