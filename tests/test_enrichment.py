"""Tests for the enrichment (ORA) module: known-answer values plus guardrails.

Import path matches the src layout (mirrors test_differential_abundance.py).
"""

import pandas as pd
import pytest

from microbiome_agent.analysis.enrichment import over_representation_analysis


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def universe():
    """100 features, f0 .. f99."""
    return [f"f{i}" for i in range(100)]


@pytest.fixture
def hits():
    """10 hits: f0 .. f9 (so 10 of 100 features are 'of interest')."""
    return [f"f{i}" for i in range(10)]


@pytest.fixture
def feature_sets():
    """Three sets with hand-computable enrichment, against hits f0..f9 / N=100.

    setA: f0..f14   -> size 15, contains all 10 hits -> heavy over-representation
                       expected = 10 * 15/100 = 1.5, observed 10, fold ~6.667
    setB: f50..f59  -> size 10, zero hits           -> p = 1.0 (one-sided)
    setC: f0,f1 + f90..f97 -> size 10, 2 hits        -> expected 1.0, fold 2.0
    """
    return {
        "setA": [f"f{i}" for i in range(15)],
        "setB": [f"f{i}" for i in range(50, 60)],
        "setC": ["f0", "f1"] + [f"f{i}" for i in range(90, 98)],
    }


# --------------------------------------------------------------------------- #
# Known answers
# --------------------------------------------------------------------------- #
def test_overrepresented_set_is_significant(universe, hits, feature_sets):
    out = over_representation_analysis(feature_sets, hits, universe).set_index("set")
    assert out.loc["setA", "n_hits_in_set"] == 10
    assert out.loc["setA", "fold_enrichment"] == pytest.approx(10 / 1.5)
    assert out.loc["setA", "q_value"] < 0.001
    assert bool(out.loc["setA", "significant"]) is True


def test_zero_overlap_set_pvalue_is_one(universe, hits, feature_sets):
    out = over_representation_analysis(feature_sets, hits, universe).set_index("set")
    assert out.loc["setB", "n_hits_in_set"] == 0
    assert out.loc["setB", "fold_enrichment"] == pytest.approx(0.0)
    assert out.loc["setB", "p_value"] == pytest.approx(1.0)
    assert bool(out.loc["setB", "significant"]) is False


def test_fold_enrichment_value(universe, hits, feature_sets):
    out = over_representation_analysis(feature_sets, hits, universe).set_index("set")
    # setC: 2 observed vs 1.0 expected -> fold 2.0
    assert out.loc["setC", "expected"] == pytest.approx(1.0)
    assert out.loc["setC", "fold_enrichment"] == pytest.approx(2.0)


def test_results_sorted_by_qvalue(universe, hits, feature_sets):
    out = over_representation_analysis(feature_sets, hits, universe)
    assert out.iloc[0]["set"] == "setA"          # most enriched first
    assert list(out["q_value"]) == sorted(out["q_value"])


def test_overlap_features_listed(universe, hits, feature_sets):
    out = over_representation_analysis(feature_sets, hits, universe).set_index("set")
    assert out.loc["setC", "overlap_features"] == "f0, f1"
    assert out.loc["setA", "overlap_features"].startswith("f0, f1")


def test_proportional_set_has_fold_near_one(universe, hits):
    # A set whose hit fraction matches the background (10%): 1 hit in a size-10 set.
    sets = {"neutral": ["f0"] + [f"f{i}" for i in range(50, 59)]}
    out = over_representation_analysis(sets, hits, universe).set_index("set")
    assert out.loc["neutral", "fold_enrichment"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Set handling
# --------------------------------------------------------------------------- #
def test_members_outside_universe_ignored(universe, hits):
    # Set lists two real hits plus junk not in the universe; only real ones count.
    sets = {"mixed": ["f0", "f1", "not_a_feature", "ghost"]}
    out = over_representation_analysis(sets, hits, universe).set_index("set")
    assert out.loc["mixed", "set_size"] == 2
    assert out.loc["mixed", "n_hits_in_set"] == 2


def test_small_sets_filtered_out(universe, hits, feature_sets):
    sets = dict(feature_sets, tiny=["f0"])           # size 1, below default min 2
    out = over_representation_analysis(sets, hits, universe)
    assert "tiny" not in set(out["set"])


def test_max_set_size_filter(universe, hits, feature_sets):
    out = over_representation_analysis(
        feature_sets, hits, universe, max_set_size=10
    )
    assert "setA" not in set(out["set"])             # size 15, above cap
    assert "setC" in set(out["set"])


# --------------------------------------------------------------------------- #
# Guardrails (fail loudly)
# --------------------------------------------------------------------------- #
def test_empty_feature_sets_raises(universe, hits):
    with pytest.raises(ValueError, match="feature_sets.*empty"):
        over_representation_analysis({}, hits, universe)


def test_empty_universe_raises(hits, feature_sets):
    with pytest.raises(ValueError, match="universe.*empty"):
        over_representation_analysis(feature_sets, hits, [])


def test_empty_hits_raises(universe, feature_sets):
    with pytest.raises(ValueError, match="hits.*empty"):
        over_representation_analysis(feature_sets, [], universe)


def test_hit_outside_universe_raises(universe, feature_sets):
    with pytest.raises(ValueError, match="not in the universe"):
        over_representation_analysis(feature_sets, ["f0", "rogue_hit"], universe)


def test_duplicate_universe_raises(hits, feature_sets):
    with pytest.raises(ValueError, match="duplicate"):
        over_representation_analysis(feature_sets, hits, ["f0", "f0", "f1"])


def test_all_sets_filtered_raises(universe, hits):
    with pytest.raises(ValueError, match="No feature sets remain"):
        over_representation_analysis({"tiny": ["f0"]}, hits, universe)
