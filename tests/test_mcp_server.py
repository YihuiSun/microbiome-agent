"""Tests for the Phase 2 MCP server.

Covers the deliverable directly: the server lists its tools (with schemas and
docstrings) and runs them when called. The end-to-end workflow writes its own
controlled CSVs with a known planted signal, so it does not depend on the
bundled example dataset's exact shape; a separate smoke test covers that loader.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microbiome_agent.mcp_server import server as srv
from microbiome_agent.mcp_server.server import (
    mcp,
    load_dataset,
    load_example_dataset,
    dataset_summary,
    run_differential_abundance,
    run_alpha_diversity,
    run_beta_diversity,
    run_enrichment,
    generate_report,
)

_EXPECTED_TOOLS = {
    "load_dataset", "load_example_dataset", "dataset_summary",
    "run_differential_abundance", "run_alpha_diversity", "run_beta_diversity",
    "run_enrichment", "generate_report",
}


@pytest.fixture(autouse=True)
def _clean_registries():
    """Reset the server's in-memory state before each test (it is global)."""
    srv._DATASETS.clear()
    srv._RESULTS.clear()
    srv._COUNTERS.clear()
    yield


@pytest.fixture
def synthetic_csvs(tmp_path):
    """Write a small dataset with Bug_A planted as elevated in the 'case' group."""
    rng = np.random.default_rng(0)
    ctrl = [f"c{i}" for i in range(8)]
    case = [f"k{i}" for i in range(8)]
    samples = ctrl + case
    features = [f"Bug_{c}" for c in "ABCDEF"]
    mat = pd.DataFrame(rng.lognormal(2, 0.3, size=(16, 6)),
                       index=samples, columns=features)
    mat.loc[case, "Bug_A"] *= 20.0          # the planted signal
    ab = mat.round(3).reset_index().rename(columns={"index": "sample_id"})
    meta = pd.DataFrame({"sample_id": samples,
                         "condition": ["control"] * 8 + ["case"] * 8})
    ab_path = tmp_path / "abundance.csv"
    meta_path = tmp_path / "metadata.csv"
    ab.to_csv(ab_path, index=False)
    meta.to_csv(meta_path, index=False)
    return str(ab_path), str(meta_path)


def _call(name, **arguments):
    """Invoke a tool through the real MCP path and parse its JSON result."""
    res = asyncio.run(mcp.call_tool(name, arguments))
    if isinstance(res, tuple):       # some versions return (content, structured)
        res = res[0]
    for block in res:
        if getattr(block, "text", None):
            return json.loads(block.text)
    raise AssertionError("tool returned no text content")


# --------------------------------------------------------------------------- #
# The server lists its tools
# --------------------------------------------------------------------------- #
def test_lists_all_tools_with_schemas():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert _EXPECTED_TOOLS <= names
    for t in tools:
        assert t.description and len(t.description) > 20   # real docstrings
        assert "properties" in t.inputSchema               # JSON input schema


def test_required_args_in_schema():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    schema = tools["run_differential_abundance"].inputSchema
    assert "dataset_id" in schema["properties"]
    assert set(schema["required"]) >= {"dataset_id", "group_column"}


# --------------------------------------------------------------------------- #
# Full workflow on self-contained synthetic data (direct calls)
# --------------------------------------------------------------------------- #
def test_end_to_end_recovers_planted_signal(synthetic_csvs, tmp_path):
    ab_path, meta_path = synthetic_csvs
    ds = load_dataset(ab_path, meta_path)
    did = ds["dataset_id"]
    assert ds["n_samples"] == 16 and ds["n_features"] == 6

    summary = dataset_summary(did)
    assert summary["metadata"]["condition"] == {"control": 8, "case": 8}

    diff = run_differential_abundance(did, "condition")
    sig = [f["feature"] for f in diff["significant_features"]]
    assert "Bug_A" in sig                      # the planted signal is recovered

    alpha = run_alpha_diversity(did, "condition")
    assert set(alpha["by_group"]) == {"control", "case"}

    beta = run_beta_diversity(did, "condition", seed=0)
    assert beta["group_sizes"] == {"control": 8, "case": 8}
    assert beta["permanova"]["method"] == "PERMANOVA"

    ora = run_enrichment(
        diff["analysis_id"],
        {"setX": ["Bug_A", "Bug_B"], "setY": ["Bug_C", "Bug_D", "Bug_E"]},
    )
    assert ora["n_sets_tested"] == 2

    rep = generate_report(
        title="synthetic",
        analysis_ids=[diff["analysis_id"], alpha["analysis_id"],
                      beta["analysis_id"], ora["analysis_id"]],
        output_dir=str(tmp_path / "report"),
    )
    assert set(rep["figures"]) == {"volcano", "alpha", "pcoa", "enrichment"}
    assert Path(rep["html_path"]).exists()


# --------------------------------------------------------------------------- #
# The MCP call path and the bundled example loader
# --------------------------------------------------------------------------- #
def test_call_tool_round_trip(synthetic_csvs):
    ab_path, meta_path = synthetic_csvs
    out = _call("load_dataset", abundance_path=ab_path, metadata_path=meta_path)
    assert out["dataset_id"].startswith("ds-")
    summ = _call("dataset_summary", dataset_id=out["dataset_id"])
    assert summ["n_features"] == 6


def test_load_example_dataset_smoke():
    # No hard-coded dimensions: just confirm the bundled data loads and registers.
    out = load_example_dataset()
    assert out["dataset_id"].startswith("ds-")
    assert out["n_samples"] > 0 and out["n_features"] > 0
    assert isinstance(out["metadata_columns"], list)


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
def test_unknown_dataset_id_raises():
    with pytest.raises(ValueError, match="Unknown dataset_id"):
        run_differential_abundance("ds-999", "condition")


def test_enrichment_requires_differential_kind(synthetic_csvs):
    ab_path, meta_path = synthetic_csvs
    ds = load_dataset(ab_path, meta_path)
    alpha = run_alpha_diversity(ds["dataset_id"], "condition")
    with pytest.raises(ValueError, match="expected differential"):
        run_enrichment(alpha["analysis_id"], {"setX": ["Bug_A", "Bug_B"]})


def test_enrichment_with_no_hits_raises(synthetic_csvs):
    ab_path, meta_path = synthetic_csvs
    ds = load_dataset(ab_path, meta_path)
    # fdr_alpha = 0 makes nothing significant -> no hits to enrich
    diff = run_differential_abundance(ds["dataset_id"], "condition", fdr_alpha=0.0)
    with pytest.raises(ValueError, match="no significant features"):
        run_enrichment(diff["analysis_id"], {"setX": ["Bug_A", "Bug_B"]})
