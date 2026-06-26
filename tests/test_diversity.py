"""Tests for the diversity module: known-answer values plus guardrails.

Import path matches the src layout (mirrors how test_differential_abundance.py
imports from microbiome_agent.analysis). Runs under `pytest` from the repo root
with the package editable-installed (`pip install -e .`).
"""

import math

import numpy as np
import pandas as pd
import pytest

from microbiome_agent.analysis.diversity import (
    alpha_diversity,
    beta_diversity,
    BetaDiversityResult,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def simple_abundance():
    """Three samples with hand-computable diversity.

    s1: four equally-abundant taxa  -> Shannon = log2(4) = 2.0 bits, 4 observed
    s2: a single taxon              -> Shannon = 0.0,               1 observed
    s3: [20,10,5,5]                 -> 1.75 bits,                   4 observed
    """
    return pd.DataFrame(
        [[10, 10, 10, 10], [40, 0, 0, 0], [20, 10, 5, 5]],
        index=["s1", "s2", "s3"],
        columns=["t1", "t2", "t3", "t4"],
    )


@pytest.fixture
def separated_dataset():
    """Two clearly separated groups (A dominated by t1, B by t2)."""
    abundance = pd.DataFrame(
        [
            [100, 1, 1], [98, 2, 1], [102, 1, 2], [99, 3, 1], [101, 1, 1],
            [1, 100, 1], [2, 98, 1], [1, 102, 2], [1, 99, 3], [1, 101, 1],
        ],
        index=[f"a{i}" for i in range(1, 6)] + [f"b{i}" for i in range(1, 6)],
        columns=["t1", "t2", "t3"],
    )
    groups = pd.Series(
        ["A"] * 5 + ["B"] * 5, index=abundance.index, name="condition"
    )
    return abundance, groups


# --------------------------------------------------------------------------- #
# Alpha diversity: known answers
# --------------------------------------------------------------------------- #
def test_shannon_known_values_base2(simple_abundance):
    out = alpha_diversity(simple_abundance).set_index("sample")
    assert out.loc["s1", "shannon"] == pytest.approx(2.0)   # log2(4)
    assert out.loc["s2", "shannon"] == pytest.approx(0.0)   # single taxon
    assert out.loc["s3", "shannon"] == pytest.approx(1.75)


def test_shannon_base_e(simple_abundance):
    out = alpha_diversity(simple_abundance, base=math.e).set_index("sample")
    assert out.loc["s1", "shannon"] == pytest.approx(math.log(4))


def test_observed_features(simple_abundance):
    out = alpha_diversity(simple_abundance).set_index("sample")
    assert out.loc["s1", "observed_features"] == 4
    assert out.loc["s2", "observed_features"] == 1
    assert out.loc["s3", "observed_features"] == 4


def test_alpha_preserves_sample_order_and_shape(simple_abundance):
    out = alpha_diversity(simple_abundance)
    assert list(out["sample"]) == ["s1", "s2", "s3"]
    assert list(out.columns) == ["sample", "shannon", "observed_features"]


def test_alpha_carries_group_label_and_realigns(simple_abundance):
    # groups deliberately given in a different order than the abundance rows.
    groups = pd.Series(
        {"s3": "late", "s1": "early", "s2": "early"}, name="phase"
    )
    out = alpha_diversity(simple_abundance, groups).set_index("sample")
    assert out.loc["s1", "group"] == "early"
    assert out.loc["s3", "group"] == "late"


# --------------------------------------------------------------------------- #
# Beta diversity: Bray-Curtis known answers
# --------------------------------------------------------------------------- #
def test_braycurtis_known_values(simple_abundance):
    groups = pd.Series(["x", "y", "x"], index=simple_abundance.index)
    res = beta_diversity(simple_abundance, groups, seed=0)
    d = res.distances
    # diagonal is zero, matrix symmetric
    assert d.loc["s1", "s1"] == pytest.approx(0.0)
    assert d.loc["s1", "s3"] == pytest.approx(d.loc["s3", "s1"])
    # hand-computed Bray-Curtis dissimilarities
    assert d.loc["s1", "s3"] == pytest.approx(0.25)
    assert d.loc["s1", "s2"] == pytest.approx(0.75)


def test_beta_returns_dataclass(separated_dataset):
    abundance, groups = separated_dataset
    res = beta_diversity(abundance, groups, seed=0)
    assert isinstance(res, BetaDiversityResult)
    assert res.metric == "braycurtis"
    assert res.distances.shape == (10, 10)


# --------------------------------------------------------------------------- #
# PERMANOVA
# --------------------------------------------------------------------------- #
def test_permanova_detects_separation(separated_dataset):
    abundance, groups = separated_dataset
    res = beta_diversity(abundance, groups, permutations=999, seed=42)
    row = res.permanova.iloc[0]
    assert row["method"] == "PERMANOVA"
    assert row["statistic_name"] == "pseudo-F"
    assert row["sample_size"] == 10
    assert row["num_groups"] == 2
    assert row["num_permutations"] == 999
    # clearly separated groups -> large pseudo-F, small p-value
    assert row["statistic"] > 1.0
    assert row["p_value"] < 0.05


def test_permanova_reproducible_with_seed(separated_dataset):
    abundance, groups = separated_dataset
    p1 = beta_diversity(abundance, groups, seed=7).permanova.iloc[0]["p_value"]
    p2 = beta_diversity(abundance, groups, seed=7).permanova.iloc[0]["p_value"]
    assert p1 == p2


# --------------------------------------------------------------------------- #
# Guardrails (fail loudly)
# --------------------------------------------------------------------------- #
def test_empty_abundance_raises():
    with pytest.raises(ValueError, match="empty"):
        alpha_diversity(pd.DataFrame())


def test_negative_abundance_raises(simple_abundance):
    bad = simple_abundance.copy()
    bad.iloc[0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        alpha_diversity(bad)


def test_zero_total_sample_raises(simple_abundance):
    bad = simple_abundance.copy()
    bad.loc["s2"] = 0
    with pytest.raises(ValueError, match="zero total"):
        alpha_diversity(bad)


def test_invalid_base_raises(simple_abundance):
    with pytest.raises(ValueError, match="base"):
        alpha_diversity(simple_abundance, base=1)


def test_misaligned_groups_raises(simple_abundance):
    groups = pd.Series(["x", "y"], index=["other1", "other2"])
    with pytest.raises(ValueError, match="same sample IDs"):
        alpha_diversity(simple_abundance, groups)


def test_missing_group_label_raises(simple_abundance):
    groups = pd.Series([np.nan, "y", "x"], index=simple_abundance.index)
    with pytest.raises(ValueError, match="missing labels"):
        alpha_diversity(simple_abundance, groups)


def test_beta_single_group_raises(simple_abundance):
    groups = pd.Series(["only", "only", "only"], index=simple_abundance.index)
    with pytest.raises(ValueError, match="at least two groups"):
        beta_diversity(simple_abundance, groups)


def test_beta_single_sample_raises():
    one = pd.DataFrame([[1, 2, 3]], index=["s1"], columns=["t1", "t2", "t3"])
    groups = pd.Series(["x"], index=["s1"])
    with pytest.raises(ValueError, match="at least two samples"):
        beta_diversity(one, groups)
