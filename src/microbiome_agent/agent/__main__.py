"""CLI entry point for the microbiome-agent.

Usage::

    python -m microbiome_agent.agent "<question>"

    # Example — analyse the bundled synthetic dataset end-to-end:
    python -m microbiome_agent.agent \\
        "Analyse the example dataset. Run differential abundance, alpha and
         beta diversity, then generate a report."

    # Example — load your own data:
    python -m microbiome_agent.agent \\
        "Load data/abundance.csv and data/metadata.csv, run a full
         microbiome analysis grouped by study_condition, and write a report
         to reports/."

The agent spawns its own MCP server subprocess (same Python interpreter, same
venv), so all you need is an ANTHROPIC_API_KEY in your environment.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from mcp import StdioServerParameters

from .loop import AgentLoop


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m microbiome_agent.agent \"<question>\"\n\n"
            "Example:\n"
            "  python -m microbiome_agent.agent \\\n"
            '      "Analyse the example dataset and report what taxa differ '
            'between groups."\n',
            file=sys.stderr,
        )
        sys.exit(1)

    # Accept the question as a single quoted string or multiple bare words
    question = " ".join(sys.argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Point at the same Python that's running right now so the MCP server
    # picks up the same venv and installed packages without any extra setup.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "microbiome_agent.mcp_server.server"],
    )

    agent = AgentLoop()
    answer = asyncio.run(agent.run(question, server_params))

    print("\n" + "=" * 70)
    print(answer)
    print("=" * 70)


if __name__ == "__main__":
    main()
