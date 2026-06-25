"""Tests for the dataset loader.

Two kinds of checks, same philosophy as the differential-abundance tests:
1. Does it correctly load and align well-formed data (including the bundled
   example, end-to-end into the analysis tool)?
2. Does it refuse malformed input loudly, rather than returning silent garbage?
The second kind is what makes the loader safe for the agent to depend on.
"""

import pandas as pd
import pytest

from microbiome_agent.analysis.differential_abundance import differential_abundance
from microbiome_agent.datasets import example_dataset, load_dataset


def _write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def test_example_dataset_loads_and_aligns():
    ds = example_dataset()
    assert ds.abundance.shape[0] == ds.metadata.shape[0]      # same n samples
    assert list(ds.abundance.index) == list(ds.metadata.index)  # same order
    assert "study_condition" in ds.metadata.columns


def test_example_signal_recovered_end_to_end():
    # The whole point: loader output feeds straight into the analysis tool,
    # and the planted Fusobacterium signal is recovered as significant.
    ds = example_dataset()
    res = differential_abundance(ds.abundance, ds.groups("study_condition"))
    top = res.iloc[0]
    assert top["feature"] == "Fusobacterium_nucleatum"
    assert bool(top["significant"]) is True


def test_alignment_to_shared_samples(tmp_path):
    # Metadata is missing one sample that abundance has; loader keeps the overlap.
    ab = tmp_path / "ab.csv"
    md = tmp_path / "md.csv"
    _write_csv(ab, ["sample_id", "taxonA"], [["S1", 0.5], ["S2", 0.3], ["S3", 0.9]])
    _write_csv(md, ["sample_id", "cond"], [["S1", "x"], ["S2", "y"]])
    ds = load_dataset(ab, md)
    assert list(ds.abundance.index) == ["S1", "S2"]


def test_missing_id_column_raises(tmp_path):
    ab = tmp_path / "ab.csv"
    md = tmp_path / "md.csv"
    _write_csv(ab, ["wrong_id", "taxonA"], [["S1", 0.5]])
    _write_csv(md, ["sample_id", "cond"], [["S1", "x"]])
    with pytest.raises(ValueError):
        load_dataset(ab, md)


def test_non_numeric_abundance_raises(tmp_path):
    ab = tmp_path / "ab.csv"
    md = tmp_path / "md.csv"
    _write_csv(ab, ["sample_id", "taxonA"], [["S1", "oops"], ["S2", "nope"]])
    _write_csv(md, ["sample_id", "cond"], [["S1", "x"], ["S2", "y"]])
    with pytest.raises(ValueError):
        load_dataset(ab, md)


def test_groups_helper_rejects_unknown_column():
    ds = example_dataset()
    with pytest.raises(ValueError):
        ds.groups("no_such_column")
