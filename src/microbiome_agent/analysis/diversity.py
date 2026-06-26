"""Diversity metrics for microbiome feature tables.

Two analyses, written in the same style as ``differential_abundance``: a clear
signature, defensive input checks, and a tidy, predictable return value. As
before, that discipline makes the code testable now and easy to expose as an MCP
tool the agent can call reliably later.

- Alpha diversity (within-sample): Shannon index, one value per sample.
- Beta diversity (between-sample): Bray-Curtis dissimilarity matrix plus a
  PERMANOVA test of whether a grouping explains that between-sample structure.

Both wrap scikit-bio so the heavy lifting is a trusted, well-tested
implementation; this module's job is the guardrails and the tidy output shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skbio.diversity import alpha_diversity as _skbio_alpha
from skbio.diversity import beta_diversity as _skbio_beta
from skbio.stats.distance import permanova as _skbio_permanova


def _check_abundance(abundance: pd.DataFrame) -> None:
    """Shared input guardrail for both diversity functions.

    Failing loudly here is deliberate -- it is the same contract the agent will
    rely on later to avoid silently producing nonsense from a malformed table.
    """
    if not isinstance(abundance, pd.DataFrame):
        raise ValueError("`abundance` must be a pandas DataFrame.")
    if abundance.empty:
        raise ValueError("`abundance` is empty; nothing to compute.")

    values = abundance.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("`abundance` contains NaN or infinite values.")
    if (values < 0).any():
        raise ValueError(
            "`abundance` contains negative values; diversity metrics expect "
            "non-negative abundances (counts or relative abundances)."
        )

    # A sample with zero total has no community to describe: Shannon is
    # undefined (0/0) and Bray-Curtis silently reports it as maximally
    # dissimilar. Catch it here rather than let either leak downstream. Note
    # this is *not* caught by the loader's drop-all-zero-*features* step.
    sample_totals = values.sum(axis=1)
    empty = abundance.index[sample_totals == 0]
    if len(empty) > 0:
        raise ValueError(
            f"Samples with zero total abundance: {list(empty)}. "
            "Drop or inspect these before computing diversity."
        )


def _align_groups(abundance: pd.DataFrame, groups: pd.Series) -> pd.Series:
    """Align a grouping vector to the abundance rows, or fail clearly.

    Mirrors the alignment contract used in ``differential_abundance`` so the two
    tools behave identically when handed the same (abundance, groups) pair.
    """
    if not isinstance(groups, pd.Series):
        raise ValueError("`groups` must be a pandas Series, one label per sample.")
    if not abundance.index.equals(groups.index):
        try:
            groups = groups.loc[abundance.index]
        except KeyError as exc:
            raise ValueError(
                "`groups` must be indexed by the same sample IDs as `abundance`."
            ) from exc
    if groups.isna().any():
        raise ValueError("`groups` contains missing labels.")
    return groups


def alpha_diversity(
    abundance: pd.DataFrame,
    groups: pd.Series | None = None,
    *,
    base: float = 2.0,
) -> pd.DataFrame:
    """Per-sample Shannon diversity (within-sample richness + evenness).

    The Shannon index summarises, for a single sample, both how many features
    are present and how evenly the abundance is spread across them. It is
    computed on each sample independently, so no grouping is required; pass
    ``groups`` only if you want the group label carried into the output (handy
    for plotting or a downstream group comparison).

    Parameters
    ----------
    abundance:
        Samples-by-features table. Rows are samples, columns are features (e.g.
        taxa). Values are abundances (counts or relative abundances).
    groups:
        Optional one-label-per-sample vector, indexed like ``abundance``'s rows.
        If given, a ``group`` column is added to the output. It does *not* change
        the Shannon values, which are always per-sample.
    base:
        Logarithm base for the index. ``2.0`` (the scikit-bio default) reports
        bits; pass ``numpy.e`` for nats. Must be > 0 and != 1.

    Returns
    -------
    pandas.DataFrame
        One row per sample, in the input's sample order, with columns:
        ``sample``, ``shannon``, ``observed_features`` (count of features with
        non-zero abundance, the richness component), and ``group`` if ``groups``
        was supplied.

    Raises
    ------
    ValueError
        If ``abundance`` is empty, non-numeric, negative, or has a zero-total
        sample; if ``base`` is invalid; or if ``groups`` is misaligned or has
        missing labels.
    """
    _check_abundance(abundance)
    if base <= 0 or base == 1:
        raise ValueError(f"`base` must be > 0 and != 1, got {base!r}.")

    counts = abundance.to_numpy(dtype=float)
    ids = list(abundance.index)
    shannon = _skbio_alpha("shannon", counts, ids=ids, base=base)

    result = pd.DataFrame(
        {
            "sample": ids,
            "shannon": shannon.to_numpy(dtype=float),
            "observed_features": (counts > 0).sum(axis=1).astype(int),
        }
    )
    if groups is not None:
        groups = _align_groups(abundance, groups)
        result["group"] = groups.to_numpy()

    return result.reset_index(drop=True)


@dataclass
class BetaDiversityResult:
    """Result of a between-sample diversity analysis.

    Attributes
    ----------
    distances:
        Square, symmetric samples-by-samples DataFrame of pairwise
        dissimilarities (zero on the diagonal). This is the input you would feed
        to an ordination (e.g. PCoA) later.
    permanova:
        One-row DataFrame summarising the PERMANOVA test of whether ``groups``
        explains the between-sample structure. Columns: ``method``,
        ``statistic_name``, ``statistic`` (the pseudo-F value), ``p_value``,
        ``sample_size``, ``num_groups``, ``num_permutations``.
    metric:
        The dissimilarity metric used (e.g. ``"braycurtis"``).
    """

    distances: pd.DataFrame
    permanova: pd.DataFrame
    metric: str


def beta_diversity(
    abundance: pd.DataFrame,
    groups: pd.Series,
    *,
    metric: str = "braycurtis",
    permutations: int = 999,
    seed: int | None = None,
) -> BetaDiversityResult:
    """Between-sample dissimilarity plus a PERMANOVA test of group structure.

    Computes a pairwise dissimilarity matrix (Bray-Curtis by default) over the
    samples, then runs PERMANOVA to ask whether ``groups`` explains a
    significant share of that between-sample variation. PERMANOVA's p-value
    comes from permuting the group labels, so it is stochastic; pass ``seed``
    for a reproducible result.

    Parameters
    ----------
    abundance:
        Samples-by-features table (rows = samples, columns = features).
    groups:
        One label per sample, indexed like ``abundance``'s rows. Must resolve to
        at least two distinct groups (PERMANOVA compares groups).
    metric:
        Dissimilarity metric passed to scikit-bio's ``beta_diversity``. Defaults
        to ``"braycurtis"``.
    permutations:
        Number of label permutations used to compute the PERMANOVA p-value.
    seed:
        Optional seed for the permutation draw, for a reproducible p-value.

    Returns
    -------
    BetaDiversityResult
        ``.distances`` (square DataFrame), ``.permanova`` (one-row tidy
        DataFrame), and ``.metric``.

    Raises
    ------
    ValueError
        If ``abundance`` is empty/negative/has a zero-total sample, if there are
        fewer than two samples, or if ``groups`` is misaligned, has missing
        labels, or does not resolve to at least two distinct groups.
    """
    _check_abundance(abundance)
    if abundance.shape[0] < 2:
        raise ValueError(
            f"Need at least two samples for beta diversity, got "
            f"{abundance.shape[0]}."
        )

    groups = _align_groups(abundance, groups)
    levels = sorted(groups.unique())
    if len(levels) < 2:
        raise ValueError(
            f"Expected at least two groups for PERMANOVA, found {len(levels)}: "
            f"{levels}."
        )

    ids = list(abundance.index)
    counts = abundance.to_numpy(dtype=float)
    distance_matrix = _skbio_beta(metric, counts, ids=ids)

    raw = _skbio_permanova(
        distance_matrix,
        grouping=list(groups.to_numpy()),
        permutations=permutations,
        seed=seed,
    )
    permanova_df = pd.DataFrame(
        [
            {
                "method": raw["method name"],
                "statistic_name": raw["test statistic name"],
                "statistic": float(raw["test statistic"]),
                "p_value": float(raw["p-value"]),
                "sample_size": int(raw["sample size"]),
                "num_groups": int(raw["number of groups"]),
                "num_permutations": int(raw["number of permutations"]),
            }
        ]
    )

    return BetaDiversityResult(
        distances=distance_matrix.to_data_frame(),
        permanova=permanova_df,
        metric=metric,
    )
