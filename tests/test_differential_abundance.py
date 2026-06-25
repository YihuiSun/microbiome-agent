"""Tests for the differential_abundance tool.

We build a synthetic dataset with a known answer: one taxon genuinely differs
between groups, the others are noise. A good test for an analysis function isn't
"does it run" -- it's "does it reach the conclusion we already know is correct,
and does it refuse bad input." Those two properties are exactly what the agent
will depend on later.
"""

import numpy as np
import pandas as pd
import pytest

from microbiome_agent.analysis.differential_abundance import differential_abundance


def _make_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 30  # samples per group
    samples = [f"S{i:03d}" for i in range(2 * n)]
    groups = pd.Series(["control"] * n + ["case"] * n, index=samples)

    # taxon_signal really differs between groups; the two noise taxa do not.
    signal = np.concatenate(
        [rng.normal(1.0, 0.2, n), rng.normal(3.0, 0.2, n)]
    )
    noise1 = rng.normal(2.0, 0.5, 2 * n)
    noise2 = rng.normal(5.0, 0.5, 2 * n)

    abundance = pd.DataFrame(
        {"taxon_signal": signal, "taxon_noise1": noise1, "taxon_noise2": noise2},
        index=samples,
    )
    return abundance, groups


def test_detects_true_signal():
    abundance, groups = _make_dataset()
    res = differential_abundance(abundance, groups)
    top = res.iloc[0]
    assert top["feature"] == "taxon_signal"
    assert bool(top["significant"]) is True


def test_noise_features_not_significant():
    abundance, groups = _make_dataset()
    res = differential_abundance(abundance, groups)
    noise = res[res["feature"].str.startswith("taxon_noise")]
    assert not noise["significant"].any()


def test_requires_exactly_two_groups():
    abundance, groups = _make_dataset()
    groups[:] = "only_one_group"
    with pytest.raises(ValueError):
        differential_abundance(abundance, groups)


def test_misaligned_index_raises():
    abundance, groups = _make_dataset()
    shuffled = groups.copy()
    shuffled.index = [f"WRONG{i}" for i in range(len(groups))]
    with pytest.raises(ValueError):
        differential_abundance(abundance, shuffled)
