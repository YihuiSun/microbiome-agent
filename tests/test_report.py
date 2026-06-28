"""Tests for the report assembler.

Import path matches the src layout (mirrors test_differential_abundance.py).
The assembler produces files and strings rather than numbers, so the checks are
about structure: the right files/figures appear, sections track the inputs
given, the HTML is self-contained, and bad inputs fail loudly.
"""

import pandas as pd
import pytest

from microbiome_agent.analysis.diversity import beta_diversity
from microbiome_agent.analysis.report import assemble_report, Report

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# Fixtures: realistic tool outputs
# --------------------------------------------------------------------------- #
@pytest.fixture
def differential():
    return pd.DataFrame(
        {
            "feature": ["Fusobacterium", "Bacteroides", "Prevotella",
                        "Roseburia", "Akkermansia", "Faecalibacterium"],
            "group_a": ["control"] * 6,
            "group_b": ["CRC"] * 6,
            "mean_a": [1.0, 5.0, 3.0, 2.0, 1.5, 4.0],
            "mean_b": [9.0, 5.2, 2.8, 1.9, 1.4, 4.1],
            "log2_fold_change": [3.2, 0.05, -0.1, -0.07, -0.1, 0.03],
            "p_value": [1e-6, 0.6, 0.5, 0.7, 0.65, 0.8],
            "q_value": [6e-6, 0.8, 0.8, 0.84, 0.8, 0.8],
            "significant": [True, False, False, False, False, False],
        }
    )


@pytest.fixture
def alpha():
    return pd.DataFrame(
        {
            "sample": [f"s{i}" for i in range(6)],
            "shannon": [2.1, 2.3, 2.0, 1.2, 1.0, 1.3],
            "observed_features": [40, 42, 39, 25, 22, 26],
            "group": ["control", "control", "control", "CRC", "CRC", "CRC"],
        }
    )


@pytest.fixture
def beta_and_groups():
    abundance = pd.DataFrame(
        [
            [100, 1, 1], [98, 2, 1], [102, 1, 2],
            [1, 100, 1], [2, 98, 1], [1, 102, 2],
        ],
        index=[f"c{i}" for i in range(3)] + [f"k{i}" for i in range(3)],
        columns=["t1", "t2", "t3"],
    )
    groups = pd.Series(["control"] * 3 + ["CRC"] * 3, index=abundance.index)
    return beta_diversity(abundance, groups, seed=0), groups


@pytest.fixture
def enrichment():
    return pd.DataFrame(
        {
            "set": ["Butyrate production", "Mucin degradation"],
            "set_size": [15, 10],
            "n_hits_in_set": [10, 2],
            "fold_enrichment": [6.67, 2.0],
            "q_value": [5e-10, 0.39],
            "significant": [True, False],
        }
    )


# --------------------------------------------------------------------------- #
# Full report
# --------------------------------------------------------------------------- #
def test_full_report_writes_all_artifacts(
    tmp_path, differential, alpha, beta_and_groups, enrichment
):
    beta, groups = beta_and_groups
    rep = assemble_report(
        title="CRC vs control",
        output_dir=tmp_path,
        differential=differential,
        alpha=alpha,
        beta=beta,
        enrichment=enrichment,
        groups=groups,
    )
    assert isinstance(rep, Report)
    assert rep.markdown_path.exists() and rep.html_path.exists()
    assert set(rep.figures) == {"volcano", "alpha", "pcoa", "enrichment"}
    for p in rep.figures.values():
        assert p.exists()
        assert p.read_bytes()[:8] == _PNG_MAGIC


def test_markdown_has_all_sections_and_facts(
    tmp_path, differential, alpha, beta_and_groups, enrichment
):
    beta, groups = beta_and_groups
    rep = assemble_report(
        title="CRC vs control", output_dir=tmp_path,
        differential=differential, alpha=alpha, beta=beta,
        enrichment=enrichment, groups=groups,
    )
    md = rep.markdown
    assert "# CRC vs control" in md
    for heading in ("## Summary", "## Differential abundance",
                    "## Alpha diversity", "## Beta diversity",
                    "## Functional enrichment"):
        assert heading in md
    assert "Fusobacterium" in md          # strongest differential feature
    assert "PERMANOVA" in md
    assert "figures/volcano.png" in md    # markdown links figures by rel path


def test_html_is_self_contained(tmp_path, differential, beta_and_groups):
    beta, groups = beta_and_groups
    rep = assemble_report(
        title="t", output_dir=tmp_path,
        differential=differential, beta=beta, groups=groups,
    )
    html = rep.html
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "data:image/png;base64," in html   # figures embedded, not linked
    assert "<table" in html                    # differential table rendered


# --------------------------------------------------------------------------- #
# Partial reports track the inputs given
# --------------------------------------------------------------------------- #
def test_only_alpha_section(tmp_path, alpha):
    rep = assemble_report(title="alpha only", output_dir=tmp_path, alpha=alpha)
    assert set(rep.figures) == {"alpha"}
    assert "## Alpha diversity" in rep.markdown
    assert "## Differential abundance" not in rep.markdown
    assert "## Beta diversity" not in rep.markdown


def test_alpha_without_group_column(tmp_path, alpha):
    no_group = alpha.drop(columns=["group"])
    rep = assemble_report(title="t", output_dir=tmp_path, alpha=no_group)
    assert rep.figures["alpha"].exists()       # falls back to a single boxplot


# --------------------------------------------------------------------------- #
# Guardrails (fail loudly)
# --------------------------------------------------------------------------- #
def test_no_inputs_raises(tmp_path):
    with pytest.raises(ValueError, match="Nothing to report"):
        assemble_report(title="t", output_dir=tmp_path)


def test_empty_title_raises(tmp_path, alpha):
    with pytest.raises(ValueError, match="title"):
        assemble_report(title="  ", output_dir=tmp_path, alpha=alpha)


def test_missing_column_raises(tmp_path, differential):
    bad = differential.drop(columns=["q_value"])
    with pytest.raises(ValueError, match="missing columns"):
        assemble_report(title="t", output_dir=tmp_path, differential=bad)


def test_wrong_beta_type_raises(tmp_path, differential):
    with pytest.raises(ValueError, match="BetaDiversityResult"):
        assemble_report(title="t", output_dir=tmp_path,
                        differential=differential, beta={"not": "a result"})
