"""Differential abundance testing for microbiome feature tables.

This module holds a single, well-documented analysis function. It's written in
the style we'll reuse for every tool in this project: a clear signature,
defensive input checks, and a tidy, predictable return value. That discipline
pays off twice -- it makes the code testable now, and later it makes each
function easy to expose as an MCP tool the agent can call reliably.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def differential_abundance(
    abundance: pd.DataFrame,
    groups: pd.Series,
    *,
    pseudocount: float = 1e-6,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare feature abundances between two groups of samples.

    For each feature (column) a two-sided Mann-Whitney U test compares the two
    groups, and p-values are corrected across all features with the
    Benjamini-Hochberg (FDR) procedure. A log2 fold change of the per-group mean
    abundance is reported as an effect-size companion to the p-value.

    Parameters
    ----------
    abundance:
        Samples-by-features table. Rows are samples, columns are features (e.g.
        taxa). Values are abundances (counts or relative abundances).
    groups:
        One label per sample, indexed the same way as ``abundance``'s rows. Must
        contain exactly two distinct groups.
    pseudocount:
        Small value added before the log2 fold-change calculation so that zero
        means don't produce infinities.
    fdr_alpha:
        Significance threshold applied to the FDR-adjusted p-values.

    Returns
    -------
    pandas.DataFrame
        One row per feature, sorted by adjusted p-value ascending, with columns:
        ``feature``, ``group_a``, ``group_b``, ``mean_a``, ``mean_b``,
        ``log2_fold_change``, ``p_value``, ``q_value`` (BH-adjusted) and
        ``significant`` (bool, ``q_value < fdr_alpha``).

    Raises
    ------
    ValueError
        If inputs are empty, misaligned, or ``groups`` does not have exactly two
        levels. Failing loudly here is deliberate: it's the same guardrail the
        agent will rely on later to avoid silently producing nonsense.
    """
    if abundance.empty:
        raise ValueError("`abundance` is empty; nothing to test.")

    # Align labels to the abundance rows, or fail clearly if we can't.
    if not abundance.index.equals(groups.index):
        try:
            groups = groups.loc[abundance.index]
        except KeyError as exc:
            raise ValueError(
                "`groups` must be indexed by the same sample IDs as `abundance`."
            ) from exc
    if groups.isna().any():
        raise ValueError("`groups` contains missing labels.")

    levels = sorted(groups.unique())
    if len(levels) != 2:
        raise ValueError(
            f"Expected exactly two groups, found {len(levels)}: {levels}."
        )

    group_a, group_b = levels
    mask_a = (groups == group_a).to_numpy()
    mask_b = (groups == group_b).to_numpy()

    records = []
    for feature in abundance.columns:
        values = abundance[feature].to_numpy(dtype=float)
        a_vals = values[mask_a]
        b_vals = values[mask_b]
        mean_a = float(a_vals.mean())
        mean_b = float(b_vals.mean())

        # A feature that is constant across every sample has no signal to test.
        if np.allclose(values, values[0]):
            p_value = 1.0
        else:
            _, p_value = mannwhitneyu(a_vals, b_vals, alternative="two-sided")

        log2fc = float(np.log2((mean_b + pseudocount) / (mean_a + pseudocount)))
        records.append(
            {
                "feature": feature,
                "group_a": group_a,
                "group_b": group_b,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "log2_fold_change": log2fc,
                "p_value": float(p_value),
            }
        )

    results = pd.DataFrame.from_records(records)

    # Benjamini-Hochberg FDR correction across all features tested.
    _, q_values, _, _ = multipletests(results["p_value"], method="fdr_bh")
    results["q_value"] = q_values
    results["significant"] = results["q_value"] < fdr_alpha

    return results.sort_values("q_value").reset_index(drop=True)
