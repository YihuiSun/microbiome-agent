"""Functional / pathway enrichment for microbiome feature rankings.

The same contract as every other tool in this project: a clear signature,
defensive input checks, and a tidy, predictable return value -- so it is
testable now and safe to expose as an MCP tool the agent can call later.

This module implements **over-representation analysis (ORA)**: given a set of
features of interest (the "hits" -- typically the significant features coming
out of ``differential_abundance``) drawn from a background "universe" (all the
features that were tested), it asks, for each functional category / pathway,
whether that category is over-represented among the hits more than chance would
predict. Each category is tested with a one-sided Fisher's exact test and the
resulting p-values are corrected across all categories with Benjamini-Hochberg.

ORA is chosen over a GSEA-style running-sum score because it is *deterministic*
(exact p-values, no permutation draw), which makes it cleanly testable and fast
-- the right first enrichment tool. A rank-based GSEA variant is a natural later
addition, layered on the same input shapes.

Typical use, downstream of differential abundance::

    da = differential_abundance(ds.abundance, ds.groups("study_condition"))
    hits = da.loc[da["significant"], "feature"]
    universe = da["feature"]                 # every feature that was tested
    ora = over_representation_analysis(pathway_sets, hits, universe)
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def over_representation_analysis(
    feature_sets: Mapping[str, Collection[str]],
    hits: Collection[str],
    universe: Collection[str],
    *,
    min_set_size: int = 2,
    max_set_size: int | None = None,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Test which feature sets are over-represented among a list of hits.

    For each set, a 2x2 contingency table of (in-set / not-in-set) x
    (hit / not-hit) is built over the universe and tested with a one-sided
    Fisher's exact test (alternative = "greater"). P-values are then corrected
    across all tested sets with the Benjamini-Hochberg (FDR) procedure.

    Parameters
    ----------
    feature_sets:
        Mapping of set name -> the features belonging to that set (e.g. pathway
        name -> taxa/genes in the pathway). Members not present in ``universe``
        are ignored, so a single global catalogue can be reused across studies.
    hits:
        The features of interest -- usually the significant features from
        ``differential_abundance``. Must all be present in ``universe``.
    universe:
        The background: every feature that was actually tested. This is the
        correct denominator for enrichment; using "all features in the database"
        instead would bias every p-value.
    min_set_size, max_set_size:
        After intersecting each set with the universe, sets whose size falls
        outside ``[min_set_size, max_set_size]`` are dropped before testing.
        Tiny or huge sets carry little interpretable signal and only inflate the
        multiple-testing correction. ``max_set_size=None`` means no upper bound.
    fdr_alpha:
        Significance threshold applied to the FDR-adjusted p-values.

    Returns
    -------
    pandas.DataFrame
        One row per tested set, sorted by adjusted p-value ascending, with
        columns: ``set``, ``set_size`` (members within the universe),
        ``n_hits_in_set``, ``n_hits`` (total hits), ``n_universe``,
        ``expected`` (hits expected in the set by chance),
        ``fold_enrichment`` (observed / expected), ``p_value``,
        ``q_value`` (BH-adjusted), ``significant`` (``q_value < fdr_alpha``) and
        ``overlap_features`` (comma-separated hit features in the set).

    Raises
    ------
    ValueError
        If any input is empty, if a hit is not in the universe, if the universe
        contains duplicates, or if no set survives the size filter. Failing
        loudly here is the same guardrail the agent relies on downstream.
    """
    if not feature_sets:
        raise ValueError("`feature_sets` is empty; nothing to test.")

    universe_list = list(universe)
    if len(universe_list) == 0:
        raise ValueError("`universe` is empty.")
    universe_set = set(universe_list)
    if len(universe_set) != len(universe_list):
        raise ValueError("`universe` contains duplicate features.")

    hit_set = set(hits)
    if len(hit_set) == 0:
        raise ValueError("`hits` is empty; nothing is enriched.")
    stray = hit_set - universe_set
    if stray:
        raise ValueError(
            f"{len(stray)} hit(s) not in the universe, e.g. "
            f"{sorted(stray)[:5]}. Hits must be a subset of the tested features."
        )

    n_universe = len(universe_set)
    n_hits = len(hit_set)
    non_hits = n_universe - n_hits

    records = []
    for name, members in feature_sets.items():
        in_universe = set(members) & universe_set
        set_size = len(in_universe)
        if set_size < min_set_size:
            continue
        if max_set_size is not None and set_size > max_set_size:
            continue

        overlap = in_universe & hit_set
        a = len(overlap)               # hits in set
        b = n_hits - a                 # hits not in set
        c = set_size - a               # non-hits in set
        d = non_hits - c               # non-hits not in set

        _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        expected = n_hits * set_size / n_universe
        fold = a / expected if expected > 0 else float("nan")

        records.append(
            {
                "set": name,
                "set_size": set_size,
                "n_hits_in_set": a,
                "n_hits": n_hits,
                "n_universe": n_universe,
                "expected": float(expected),
                "fold_enrichment": float(fold),
                "p_value": float(p_value),
                "overlap_features": ", ".join(sorted(overlap)),
            }
        )

    if not records:
        raise ValueError(
            "No feature sets remain after the size filter "
            f"(min_set_size={min_set_size}, max_set_size={max_set_size})."
        )

    results = pd.DataFrame.from_records(records)

    _, q_values, _, _ = multipletests(results["p_value"], method="fdr_bh")
    results["q_value"] = q_values
    results["significant"] = results["q_value"] < fdr_alpha

    ordered = [
        "set", "set_size", "n_hits_in_set", "n_hits", "n_universe",
        "expected", "fold_enrichment", "p_value", "q_value", "significant",
        "overlap_features",
    ]
    return (
        results[ordered]
        .sort_values(["q_value", "p_value"], kind="stable")
        .reset_index(drop=True)
    )
