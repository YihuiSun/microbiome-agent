"""MCP server exposing the microbiome-agent analysis tools.

Phase 2. Wraps the Phase 1 analysis library as Model Context Protocol tools so an
LLM agent can call them. Built with the FastMCP server from the official MCP SDK.

Design: handles, not tables
---------------------------
The Phase 1 functions take and return pandas objects, which an LLM cannot pass
over MCP -- and a real abundance table is far too large to shuttle through the
model's context anyway. So every tool here speaks in *handles* and *summaries*:

1. ``load_dataset`` reads the CSVs server-side and returns a small ``dataset_id``
   plus a shape/metadata summary. The full table stays in this process.
2. Each analysis tool takes a ``dataset_id``, does the heavy work server-side,
   caches the full result, and returns the agent a compact JSON summary plus an
   ``analysis_id``.
3. ``generate_report`` takes those ``analysis_id``s and assembles the cached
   full results into a report on disk.

The agent therefore reasons over findings (which taxa, which p-values), never
over raw matrices. The state lives in two in-memory registries below; it is
session-scoped, which is the right lifetime for a single agent run.

Run it::

    python -m microbiome_agent.mcp_server.server      # stdio transport
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from mcp.server.fastmcp import FastMCP

from microbiome_agent.analysis.differential_abundance import differential_abundance
from microbiome_agent.analysis.diversity import alpha_diversity, beta_diversity
from microbiome_agent.analysis.enrichment import over_representation_analysis
from microbiome_agent.analysis.report import assemble_report
from microbiome_agent.datasets.loaders import (
    Dataset,
    example_dataset,
    load_dataset as _load_dataset,
)

mcp = FastMCP("microbiome-agent")


# --------------------------------------------------------------------------- #
# In-memory registries (session state)
# --------------------------------------------------------------------------- #
_DATASETS: dict[str, Dataset] = {}
_RESULTS: dict[str, dict] = {}
_COUNTERS: dict[str, int] = {}


def _new_id(kind: str) -> str:
    _COUNTERS[kind] = _COUNTERS.get(kind, 0) + 1
    return f"{kind}-{_COUNTERS[kind]}"


def _get_dataset(dataset_id: str) -> Dataset:
    if dataset_id not in _DATASETS:
        raise ValueError(
            f"Unknown dataset_id {dataset_id!r}. Call load_dataset (or "
            f"load_example_dataset) first; known: {sorted(_DATASETS)}.")
    return _DATASETS[dataset_id]


def _get_result(analysis_id: str, kind: Optional[str] = None) -> dict:
    if analysis_id not in _RESULTS:
        raise ValueError(
            f"Unknown analysis_id {analysis_id!r}; known: {sorted(_RESULTS)}.")
    entry = _RESULTS[analysis_id]
    if kind is not None and entry["kind"] != kind:
        raise ValueError(
            f"analysis_id {analysis_id!r} is a {entry['kind']} result, "
            f"expected {kind}.")
    return entry


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of row dicts (numpy types, NaN handled)."""
    return json.loads(df.to_json(orient="records"))


def _register_dataset(ds: Dataset) -> dict:
    dataset_id = _new_id("ds")
    _DATASETS[dataset_id] = ds
    return {
        "dataset_id": dataset_id,
        "n_samples": int(ds.abundance.shape[0]),
        "n_features": int(ds.abundance.shape[1]),
        "metadata_columns": list(ds.metadata.columns),
        "sample_ids_preview": list(ds.abundance.index[:10]),
    }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@mcp.tool()
def load_dataset(
    abundance_path: str,
    metadata_path: str,
    sample_id_col: str = "sample_id",
) -> dict:
    """Load a microbiome dataset from two CSV files and register it for analysis.

    Use this first. It reads, validates, and aligns an abundance table with its
    sample metadata, keeping only samples present in both. The returned
    ``dataset_id`` is the handle every other tool needs.

    CSV format: each file has a sample-id column (default ``sample_id``);
    abundance has one further column per feature (taxon), metadata one per
    sample variable (e.g. study_condition, age, sex).

    Returns a summary: ``dataset_id``, ``n_samples``, ``n_features``,
    ``metadata_columns`` (the candidate grouping variables), and a preview of
    sample IDs. Inspect ``metadata_columns`` then call ``dataset_summary`` to
    choose a grouping.
    """
    ds = _load_dataset(abundance_path, metadata_path, sample_id_col=sample_id_col)
    return _register_dataset(ds)


@mcp.tool()
def load_example_dataset() -> dict:
    """Load the bundled synthetic example dataset and register it.

    Handy for trying the pipeline end-to-end with no files of your own. The data
    is SYNTHETIC, with a deliberately planted signal (Fusobacterium elevated in
    the CRC group) so the tools have a known-correct answer to recover. Returns
    the same summary as ``load_dataset``.
    """
    return _register_dataset(example_dataset())


@mcp.tool()
def dataset_summary(dataset_id: str, max_levels: int = 12) -> dict:
    """Summarise a loaded dataset's metadata to help choose a grouping variable.

    For each metadata column with few distinct values, returns the level counts
    (e.g. ``{"control": 12, "CRC": 12}``). Use this before the analysis tools to
    pick a ``group_column`` and to check group sizes -- small groups weaken
    every downstream test, especially PERMANOVA (see ``run_beta_diversity``).

    Columns with more than ``max_levels`` distinct values (e.g. continuous age)
    are reported as ``{"distinct": N}`` rather than enumerated.
    """
    ds = _get_dataset(dataset_id)
    cols = {}
    for col in ds.metadata.columns:
        n_unique = int(ds.metadata[col].nunique(dropna=True))
        if n_unique <= max_levels:
            counts = ds.metadata[col].value_counts(dropna=False)
            cols[col] = {str(k): int(v) for k, v in counts.items()}
        else:
            cols[col] = {"distinct": n_unique}
    return {"dataset_id": dataset_id,
            "n_samples": int(ds.abundance.shape[0]),
            "n_features": int(ds.abundance.shape[1]),
            "metadata": cols}


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
@mcp.tool()
def run_differential_abundance(
    dataset_id: str,
    group_column: str,
    fdr_alpha: float = 0.05,
    top_n: int = 50,
) -> dict:
    """Find features that differ between two groups of samples.

    Runs a per-feature Mann-Whitney U test with Benjamini-Hochberg FDR
    correction and a log2 fold change. The ``group_column`` must name a metadata
    column with exactly two levels (check with ``dataset_summary`` first).

    Returns an ``analysis_id`` (reference it in ``generate_report`` and
    ``run_enrichment``) plus a summary: number significant at ``fdr_alpha``, the
    two group labels, every significant feature, and the top ``top_n`` rows by
    adjusted p-value. Judge findings by ``q_value`` (FDR-corrected), not raw
    ``p_value``.
    """
    ds = _get_dataset(dataset_id)
    result = differential_abundance(
        ds.abundance, ds.groups(group_column), fdr_alpha=fdr_alpha)
    analysis_id = _new_id("diff")
    _RESULTS[analysis_id] = {"kind": "differential", "result": result,
                             "dataset_id": dataset_id,
                             "group_column": group_column}
    sig = result[result["significant"]]
    return {
        "analysis_id": analysis_id,
        "group_column": group_column,
        "groups": [str(result.iloc[0]["group_a"]), str(result.iloc[0]["group_b"])],
        "n_features_tested": int(len(result)),
        "n_significant": int(len(sig)),
        "fdr_alpha": fdr_alpha,
        "significant_features": _records(
            sig[["feature", "log2_fold_change", "q_value"]]),
        "top_results": _records(result.head(top_n)),
    }


@mcp.tool()
def run_alpha_diversity(
    dataset_id: str,
    group_column: Optional[str] = None,
    base: float = 2.0,
) -> dict:
    """Compute within-sample (alpha) diversity -- the Shannon index per sample.

    Shannon captures both richness and evenness; higher means a more diverse
    community. If ``group_column`` is given, the per-sample values are summarised
    by group so you can compare (e.g. is diversity lower in disease?). Reported
    in bits (log base 2 by default).

    Returns an ``analysis_id`` plus, per group, the sample count and mean /
    median / sd of Shannon, with an overall mean. Note: this reports the values;
    it does not itself test whether groups differ.
    """
    ds = _get_dataset(dataset_id)
    groups = ds.groups(group_column) if group_column else None
    result = alpha_diversity(ds.abundance, groups, base=base)
    analysis_id = _new_id("alpha")
    _RESULTS[analysis_id] = {"kind": "alpha", "result": result,
                             "dataset_id": dataset_id,
                             "group_column": group_column}
    summary: dict = {"analysis_id": analysis_id,
                     "n_samples": int(len(result)),
                     "overall_mean_shannon": float(result["shannon"].mean())}
    if "group" in result.columns:
        by = result.groupby("group")["shannon"]
        summary["by_group"] = {
            str(g): {"n": int(by.size()[g]),
                     "mean": float(by.mean()[g]),
                     "median": float(by.median()[g]),
                     "sd": float(by.std(ddof=1)[g]) if by.size()[g] > 1 else None}
            for g in by.size().index}
    return summary


@mcp.tool()
def run_beta_diversity(
    dataset_id: str,
    group_column: str,
    metric: str = "braycurtis",
    permutations: int = 999,
    seed: Optional[int] = None,
) -> dict:
    """Test whether community composition differs between groups (beta diversity).

    Computes a pairwise Bray-Curtis dissimilarity matrix over samples, then runs
    PERMANOVA to ask whether ``group_column`` explains the between-sample
    structure. A significant result means the groups have distinguishable
    overall community composition.

    Returns an ``analysis_id`` plus the PERMANOVA result (pseudo-F, p-value,
    permutation count) and the per-group sample sizes. CAVEAT, important for
    interpretation: PERMANOVA's smallest possible p-value is bounded by sample
    size, not effect size -- with only a few samples per group the p-value cannot
    be small no matter how cleanly the groups separate. A ``note`` field flags
    this when a group is small.
    """
    ds = _get_dataset(dataset_id)
    groups = ds.groups(group_column)
    result = beta_diversity(ds.abundance, groups, metric=metric,
                            permutations=permutations, seed=seed)
    analysis_id = _new_id("beta")
    _RESULTS[analysis_id] = {"kind": "beta", "result": result,
                             "dataset_id": dataset_id,
                             "group_column": group_column}
    sizes = {str(k): int(v) for k, v in groups.value_counts().items()}
    out = {
        "analysis_id": analysis_id,
        "group_column": group_column,
        "metric": metric,
        "group_sizes": sizes,
        "permanova": _records(result.permanova)[0],
    }
    if min(sizes.values()) < 4:
        out["note"] = (
            "At least one group has fewer than 4 samples; the PERMANOVA p-value "
            "is resolution-limited and may not reach significance even under "
            "strong separation. Interpret a non-significant result cautiously.")
    return out


@mcp.tool()
def run_enrichment(
    differential_analysis_id: str,
    feature_sets: dict[str, list[str]],
    fdr_alpha: float = 0.05,
    top_n: int = 50,
) -> dict:
    """Test which functional categories are over-represented among the hits.

    Over-representation analysis (Fisher's exact test + FDR). Takes the
    ``analysis_id`` of a prior ``run_differential_abundance`` call: its
    significant features become the "hits" and all tested features the
    background. ``feature_sets`` maps each category name (e.g. a pathway) to its
    member features, supplied by the caller from a gene-set / pathway catalogue.

    Returns an ``analysis_id`` plus, per set, fold enrichment, overlap, and
    FDR-adjusted q-value, sorted most-enriched first. (For large catalogues a
    file-based loader is a natural future addition; today the sets are passed
    inline.)
    """
    entry = _get_result(differential_analysis_id, kind="differential")
    da = entry["result"]
    hits = da.loc[da["significant"], "feature"]
    if len(hits) == 0:
        raise ValueError(
            f"{differential_analysis_id} has no significant features, so there "
            "is nothing to test for enrichment.")
    result = over_representation_analysis(
        feature_sets, hits, da["feature"], fdr_alpha=fdr_alpha)
    analysis_id = _new_id("ora")
    _RESULTS[analysis_id] = {"kind": "enrichment", "result": result,
                             "dataset_id": entry["dataset_id"]}
    return {
        "analysis_id": analysis_id,
        "n_sets_tested": int(len(result)),
        "n_significant": int(result["significant"].sum()),
        "fdr_alpha": fdr_alpha,
        "top_results": _records(result.head(top_n)),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
@mcp.tool()
def generate_report(
    title: str,
    analysis_ids: list[str],
    output_dir: str,
    top_n: int = 20,
) -> dict:
    """Assemble prior analyses into a markdown + self-contained HTML report.

    Pass the ``analysis_id``s of the analyses to include (any mix of
    differential, alpha, beta, enrichment). Each is looked up, a figure is drawn
    (volcano, alpha boxplot, PCoA, enrichment bar), and everything is written to
    ``output_dir`` as ``report.md``, ``report.html`` and a ``figures/`` folder.
    If a beta-diversity analysis is included, its PCoA is coloured by the same
    grouping that analysis used.

    Returns the paths written and the figure list.
    """
    kwargs: dict = {}
    groups = None
    for aid in analysis_ids:
        entry = _get_result(aid)
        kind = entry["kind"]
        if kind == "differential":
            kwargs["differential"] = entry["result"]
        elif kind == "alpha":
            kwargs["alpha"] = entry["result"]
        elif kind == "beta":
            kwargs["beta"] = entry["result"]
            ds = _DATASETS.get(entry["dataset_id"])
            if ds is not None and entry.get("group_column"):
                groups = ds.groups(entry["group_column"])
        elif kind == "enrichment":
            kwargs["enrichment"] = entry["result"]

    report = assemble_report(
        title=title, output_dir=output_dir, groups=groups, top_n=top_n, **kwargs)
    return {
        "title": report.title,
        "markdown_path": str(report.markdown_path),
        "html_path": str(report.html_path),
        "figures": {k: str(v) for k, v in report.figures.items()},
    }


def main() -> None:
    """Entry point: run the server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
