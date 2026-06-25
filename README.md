# microbiome-agent

An AI agent that plans and executes microbiome analyses (differential abundance,
diversity, functional enrichment) by orchestrating well-tested analysis tools via
the Model Context Protocol (MCP), with a built-in evaluation harness that scores
the agent on tool selection and scientific correctness.

> Status: **Phase 1 — building and testing the core analysis tools** (no AI yet).

## Why this exists

Most "AI for bioinformatics" demos are thin wrappers around a single prompt. This
project is built the other way around: a small set of correct, defensible analysis
functions first, then an agent layer that orchestrates them, then an evaluation
harness that measures whether the agent's conclusions are actually right.

## Roadmap

1. **Phase 1 — analysis tools (current).** Plain, tested Python functions for
   microbiome analysis on pre-computed abundance tables.
2. **Phase 2 — MCP server.** Expose each tool over MCP with clear schemas.
3. **Phase 3 — agent loop.** Natural-language question -> plan -> tool calls -> report.
4. **Phase 4 — evaluation harness.** Test cases with known answers; score tool
   selection, FDR discipline, and interpretation correctness.
5. **Phase 5 — engineering polish.** Tracing, Docker, CI, a small API, docs.

## Setup (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                   # makes `microbiome_agent` importable
```

## Run the tests

```bash
pytest -q
```

You should see 4 passing tests.

## Project layout

```
microbiome-agent/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/
│   └── microbiome_agent/
│       ├── __init__.py
│       └── analysis/
│           ├── __init__.py
│           └── differential_abundance.py   # first tool
└── tests/
    └── test_differential_abundance.py
```
