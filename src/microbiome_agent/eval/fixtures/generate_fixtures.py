"""Generate synthetic, known-answer fixture datasets for the Phase 4 eval cases.

Each case gets its own ``eval/fixtures/<case_id>/{abundance.csv,metadata.csv}``,
generated once by this script and checked into the repo (per the eval design
doc's "checked-in fixtures" decision -- deterministic across machines,
inspectable/diffable in code review, never regenerated with a fresh seed).

Every fixture is verified against this project's *real* analysis functions
(``differential_abundance``, ``alpha_diversity``, ``beta_diversity``) at
generation time -- not just eyeballed -- so the planted signal is confirmed to
actually produce the intended statistical outcome before it's written to disk.
Where a specific outcome is required (e.g. "must NOT survive FDR correction"),
the script searches a small range of seeds and reports which one it picked.

Run once from the repo:

    python -m microbiome_agent.eval.fixtures.generate_fixtures

Column-naming convention: feature names use genus (or genus_species) with
underscores, e.g. ``Fusobacterium_nucleatum``, matching how a real
curatedMetagenomicData export would look.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from microbiome_agent.analysis.differential_abundance import differential_abundance
from microbiome_agent.analysis.diversity import alpha_diversity, beta_diversity

FIXTURES_DIR = Path(__file__).parent
GROUP_COL = "study_condition"


# --------------------------------------------------------------------------- #
# Generic two-group synthetic abundance table
# --------------------------------------------------------------------------- #
def _simulate_table(
    *,
    case_id: str,
    n_per_group: int,
    group_labels: tuple[str, str],
    background_features: list[str],
    planted: dict[str, tuple[float, float]],
    seed: int,
    noise_sd_frac: float = 0.35,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a samples-by-features relative-abundance table.

    ``planted`` maps feature name -> (mean_in_group_a, mean_in_group_b). Every
    other feature in ``background_features`` gets the same random mean in both
    groups (no true signal) -- these are the "universe" noise features that
    make FDR correction meaningful. All values are lognormal-ish and clipped
    at zero; each row is renormalised to sum to 1 (relative abundance).
    """
    rng = np.random.default_rng(seed)
    group_a, group_b = group_labels
    n_total = 2 * n_per_group
    sample_ids = [f"{case_id}_{i:03d}" for i in range(1, n_total + 1)]
    groups = pd.Series(
        [group_a] * n_per_group + [group_b] * n_per_group,
        index=sample_ids,
        name=GROUP_COL,
    )

    all_features = list(planted.keys()) + [
        f for f in background_features if f not in planted
    ]
    data = np.zeros((n_total, len(all_features)))

    for j, feature in enumerate(all_features):
        if feature in planted:
            mean_a, mean_b = planted[feature]
        else:
            # Background noise feature: same mean in both groups, drawn once
            # per feature so different noise features have different typical
            # abundances (more realistic than one flat baseline).
            shared_mean = float(rng.uniform(0.5, 5.0))
            mean_a = mean_b = shared_mean

        for i in range(n_total):
            mean = mean_a if groups.iloc[i] == group_a else mean_b
            sd = max(mean * noise_sd_frac, 1e-3)
            data[i, j] = max(rng.normal(mean, sd), 0.0)

    abundance = pd.DataFrame(data, index=sample_ids, columns=all_features)
    # Relative abundance: each sample's row sums to 1.
    row_sums = abundance.sum(axis=1)
    abundance = abundance.div(row_sums, axis=0)
    return abundance, groups


_BACKGROUND_TAXA = [
    "Bacteroides_uniformis", "Bacteroides_vulgatus", "Alistipes_putredinis",
    "Ruminococcus_bromii", "Akkermansia_muciniphila", "Blautia_obeum",
    "Dorea_formicigenerans", "Parabacteroides_distasonis", "Collinsella_aerofaciens",
    "Bifidobacterium_adolescentis", "Eubacterium_hallii", "Coprococcus_comes",
    "Subdoligranulum_variabile", "Ruminococcus_torques", "Odoribacter_splanchnicus",
    "Alistipes_shahii", "Bacteroides_ovatus", "Barnesiella_intestinihominis",
]


def _higher_in(row: pd.Series, target_group: str) -> bool:
    """True if a differential_abundance row's fold change means "higher in
    target_group", accounting for the function's own alphabetical group_a/
    group_b assignment (NOT the order groups were passed in) -- e.g. "RA"
    sorts before "control" since uppercase precedes lowercase in ASCII, so
    group_a is not always the "first" group a caller had in mind. This same
    logic is duplicated in eval/checks.py for the same reason: never assume
    fold-change sign maps to a fixed group without checking group_a/group_b.
    """
    if row["group_b"] == target_group:
        return bool(row["log2_fold_change"] > 0)
    if row["group_a"] == target_group:
        return bool(row["log2_fold_change"] < 0)
    raise ValueError(f"{target_group!r} is neither group_a nor group_b in this row.")


def _write_case(case_id: str, abundance: pd.DataFrame, groups: pd.Series, readme: str) -> None:
    out = FIXTURES_DIR / case_id
    out.mkdir(parents=True, exist_ok=True)

    ab = abundance.copy()
    ab.index.name = "sample_id"
    ab.to_csv(out / "abundance.csv")

    meta = pd.DataFrame({GROUP_COL: groups})
    meta.index.name = "sample_id"
    meta.to_csv(out / "metadata.csv")

    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"  wrote {out}/  ({abundance.shape[0]} samples x {abundance.shape[1]} features)")


# --------------------------------------------------------------------------- #
# Case 2 -- ra-prevotella-expansion
# --------------------------------------------------------------------------- #
def build_ra() -> None:
    print("case 2: ra-prevotella-expansion")
    planted = {
        "Prevotella_copri": (1.0, 8.0),      # up in RA
        "Bacteroides_fragilis": (6.0, 1.5),  # down in RA
    }
    seed = 2
    abundance, groups = _simulate_table(
        case_id="ra", n_per_group=15, group_labels=("control", "RA"),
        background_features=_BACKGROUND_TAXA, planted=planted, seed=seed)

    da = differential_abundance(abundance, groups)
    hit = da.set_index("feature")
    assert hit.loc["Prevotella_copri", "significant"], "Prevotella_copri should be significant"
    assert _higher_in(hit.loc["Prevotella_copri"], "RA"), "Prevotella_copri should be UP in RA"
    assert hit.loc["Bacteroides_fragilis", "significant"], "Bacteroides_fragilis should be significant"
    assert _higher_in(hit.loc["Bacteroides_fragilis"], "control"), "Bacteroides_fragilis should be UP in control (down in RA)"
    print(f"  verified: seed={seed}, "
          f"Prevotella_copri q={hit.loc['Prevotella_copri','q_value']:.2e}, "
          f"Bacteroides_fragilis q={hit.loc['Bacteroides_fragilis','q_value']:.2e}")

    _write_case("ra", abundance, groups, readme=f"""\
# ra fixture (case: ra-prevotella-expansion)

Synthetic, seed={seed}. 15 control + 15 RA samples, {abundance.shape[1]} features.

Planted signal, modeled on Scher et al., eLife 2013
(https://doi.org/10.7554/eLife.01202):
- `Prevotella_copri` expanded in RA (up).
- `Bacteroides_fragilis` reduced in RA (down).

Both verified significant (q < 0.05) via `differential_abundance` at
generation time; see `generate_fixtures.py:build_ra`.
""")


# --------------------------------------------------------------------------- #
# Case 3 -- t2d-dysbiosis-enrichment
# --------------------------------------------------------------------------- #
def build_t2d() -> None:
    print("case 3: t2d-dysbiosis-enrichment")
    planted = {
        "Roseburia_intestinalis": (7.0, 1.5),       # butyrate producer, down in T2D
        "Eubacterium_rectale": (6.0, 1.5),          # butyrate producer, down in T2D
        "Escherichia_coli": (0.5, 5.0),              # opportunistic pathogen, up in T2D
        "Enterococcus_faecalis": (0.5, 4.0),         # opportunistic pathogen, up in T2D
    }
    seed = 3
    abundance, groups = _simulate_table(
        case_id="t2d", n_per_group=15, group_labels=("control", "T2D"),
        background_features=_BACKGROUND_TAXA, planted=planted, seed=seed)

    da = differential_abundance(abundance, groups)
    hit = da.set_index("feature")
    for f in ("Roseburia_intestinalis", "Eubacterium_rectale"):
        assert hit.loc[f, "significant"] and _higher_in(hit.loc[f], "control"), f
    for f in ("Escherichia_coli", "Enterococcus_faecalis"):
        assert hit.loc[f, "significant"] and _higher_in(hit.loc[f], "T2D"), f
    print(f"  verified: seed={seed}, all 4 planted taxa significant with correct direction")

    _write_case("t2d", abundance, groups, readme=f"""\
# t2d fixture (case: t2d-dysbiosis-enrichment)

Synthetic, seed={seed}. 15 control + 15 T2D samples, {abundance.shape[1]} features.

Planted signal, modeled on Qin et al., Nature 2012
(https://doi.org/10.1038/nature11450):
- Butyrate producers (`Roseburia_intestinalis`, `Eubacterium_rectale`) down in T2D.
- Opportunistic pathogens (`Escherichia_coli`, `Enterococcus_faecalis`) up in T2D.

The eval case's `feature_sets` argument to `run_enrichment` (defined in
`eval/cases.py`, not in this CSV) maps these to `"scfa_producing_bacteria"` and
`"opportunistic_pathogens"` respectively, expecting `"opportunistic_pathogens"`
to come back significantly over-represented among the differential-abundance
hits (since both its members are planted UP, i.e. among the hits, while the
butyrate producers are DOWN and so not in the hit list for a one-sided
"greater" enrichment test).
""")


# --------------------------------------------------------------------------- #
# Case 5 -- cdiff-diversity-collapse
# --------------------------------------------------------------------------- #
def build_cdiff() -> None:
    print("case 5: cdiff-diversity-collapse")
    seed = 5
    rng = np.random.default_rng(seed)
    n_per_group = 10
    features = _BACKGROUND_TAXA + ["Proteobacteria_dominant_taxon"]
    n_total = 2 * n_per_group
    sample_ids = [f"cdiff_{i:03d}" for i in range(1, n_total + 1)]
    groups = pd.Series(
        ["healthy"] * n_per_group + ["recurrent_CDI"] * n_per_group,
        index=sample_ids, name=GROUP_COL)

    data = np.zeros((n_total, len(features)))
    for i, sid in enumerate(sample_ids):
        if groups.loc[sid] == "healthy":
            # Even community: every background taxon present at a similar level.
            vals = rng.uniform(2.0, 4.0, size=len(_BACKGROUND_TAXA))
            dominant = rng.uniform(0.1, 0.3)
        else:
            # Collapsed community: one taxon dominates, everything else near zero.
            vals = rng.uniform(0.01, 0.15, size=len(_BACKGROUND_TAXA))
            dominant = rng.uniform(15.0, 25.0)
        data[i, : len(_BACKGROUND_TAXA)] = vals
        data[i, -1] = dominant

    abundance = pd.DataFrame(data, index=sample_ids, columns=features)
    abundance = abundance.div(abundance.sum(axis=1), axis=0)

    alpha = alpha_diversity(abundance, groups)
    means = alpha.groupby("group")["shannon"].mean()
    assert means["recurrent_CDI"] < means["healthy"] * 0.6, (
        f"CDI Shannon ({means['recurrent_CDI']:.2f}) should collapse well below "
        f"healthy ({means['healthy']:.2f})")
    print(f"  verified: seed={seed}, mean Shannon healthy={means['healthy']:.2f} bits, "
          f"recurrent_CDI={means['recurrent_CDI']:.2f} bits")

    _write_case("cdiff", abundance, groups, readme=f"""\
# cdiff fixture (case: cdiff-diversity-collapse)

Synthetic, seed={seed}. 10 healthy + 10 recurrent-CDI samples,
{abundance.shape[1]} features.

Planted signal, modeled on Chang et al., J Infect Dis 2008 (PMID 18199029):
markedly reduced alpha (Shannon) diversity in recurrent C. difficile-associated
diarrhea, via one dominant taxon crowding out the rest of the community.
Verified: mean Shannon in `recurrent_CDI` < 60% of `healthy` mean via
`alpha_diversity` at generation time.
""")


# --------------------------------------------------------------------------- #
# Case 6 -- obesity-mouse-caveat (deliberately NON-significant after FDR)
# --------------------------------------------------------------------------- #
def build_obesity() -> None:
    print("case 6: obesity-mouse-caveat")
    # Small, noisy Firmicutes/Bacteroidetes-style shift -- searches seeds for
    # one where the effect is present in the means but does NOT survive FDR,
    # mirroring how the human obesity F/B signature has not reliably
    # replicated despite the mouse-model finding it's modeled on.
    planted_template = {
        "Firmicutes_like_taxon": (4.0, 5.2),      # small bump, obese group
        "Bacteroidetes_like_taxon": (5.2, 4.2),    # small dip, obese group
    }
    for seed in range(100, 160):
        abundance, groups = _simulate_table(
            case_id="obesity", n_per_group=10, group_labels=("lean", "obese"),
            background_features=_BACKGROUND_TAXA, planted=planted_template,
            seed=seed, noise_sd_frac=0.55)
        da = differential_abundance(abundance, groups)
        if int(da["significant"].sum()) == 0:
            break
    else:
        raise RuntimeError("Could not find a seed with zero significant features for 'obesity'.")

    print(f"  verified: seed={seed}, 0 of {len(da)} features significant after FDR "
          f"(smallest q = {da['q_value'].min():.3g})")

    _write_case("obesity", abundance, groups, readme=f"""\
# obesity fixture (case: obesity-mouse-caveat)

Synthetic, seed={seed}. 10 lean + 10 obese samples, {abundance.shape[1]} features.

Modeled on the Firmicutes/Bacteroidetes shift reported in Turnbaugh et al.,
Nature 2006 (https://doi.org/10.1038/nature05414) -- but that finding was in
**mice**, and the human F/B obesity signature has not reliably replicated in
later human cohorts. This fixture deliberately plants only a small, noisy
version of the shift and was seed-selected so that **zero features survive
FDR correction** (verified via `differential_abundance` at generation time).

Ground truth for this case is double-sided on purpose: the deterministic check
only requires the tool ran correctly and reports no significant hits; the
LLM-judge pass evaluates whether the agent's interpretation avoids overclaiming
by citing the mouse literature as if it straightforwardly applies here.
""")


# --------------------------------------------------------------------------- #
# Case 7 -- no-signal-null-result
# --------------------------------------------------------------------------- #
def build_null() -> None:
    print("case 7: no-signal-null-result")
    for seed in range(200, 260):
        abundance, groups = _simulate_table(
            case_id="null", n_per_group=10, group_labels=("group_a", "group_b"),
            background_features=_BACKGROUND_TAXA, planted={}, seed=seed)
        da = differential_abundance(abundance, groups)
        if int(da["significant"].sum()) == 0:
            break
    else:
        raise RuntimeError("Could not find a seed with zero significant features for 'null'.")

    print(f"  verified: seed={seed}, 0 of {len(da)} features significant (pure noise, "
          f"no planted signal)")

    _write_case("null", abundance, groups, readme=f"""\
# null fixture (case: no-signal-null-result)

Synthetic, seed={seed}. 10 + 10 samples, {abundance.shape[1]} features, labels
carry **no true signal** -- every feature drawn from the same distribution in
both groups. Verified via `differential_abundance`: 0 significant features
after FDR correction. Correct agent answer is an honest "no significant
difference found," not a cherry-picked marginal p-value.
""")


# --------------------------------------------------------------------------- #
# Case 8 -- permanova-small-n
# --------------------------------------------------------------------------- #
def build_small_n() -> None:
    print("case 8: permanova-small-n")
    planted = {
        "Fusobacterium_nucleatum": (0.5, 9.0),
        "Bacteroides_fragilis": (7.0, 1.0),
    }
    seed = 8
    abundance, groups = _simulate_table(
        case_id="smalln", n_per_group=3, group_labels=("control", "disease"),
        background_features=_BACKGROUND_TAXA, planted=planted, seed=seed,
        noise_sd_frac=0.25)

    beta = beta_diversity(abundance, groups, seed=seed)
    print(f"  verified: seed={seed}, n=3 per group, PERMANOVA pseudo-F="
          f"{beta.permanova.iloc[0]['statistic']:.3g}, "
          f"p={beta.permanova.iloc[0]['p_value']:.3g} "
          f"({int(beta.permanova.iloc[0]['num_permutations'])} permutations)")
    # Deliberately small n -- the point of this case is that
    # run_beta_diversity's own `note` field should fire (min group size < 4),
    # regardless of how the p-value lands. No assertion on significance here.

    _write_case("smalln", abundance, groups, readme=f"""\
# smalln fixture (case: permanova-small-n)

Synthetic, seed={seed}. Only 3 + 3 samples (deliberately below the `run_beta_diversity`
small-group threshold of 4), {abundance.shape[1]} features, with a clear
compositional separation planted (Fusobacterium/Bacteroides shift, same style
as the CRC signal) so the community really is well-separated. The point of
this case is the PERMANOVA small-n resolution-limit caveat: the MCP server's
`run_beta_diversity` tool emits a literal `note` field whenever a group has
fewer than 4 samples, and the agent is expected to surface that caveat to the
user rather than reporting the p-value at face value.
""")


# --------------------------------------------------------------------------- #
# Case 9 -- fdr-vs-raw-p-trap
# --------------------------------------------------------------------------- #
def build_fdr_trap() -> None:
    print("case 9: fdr-vs-raw-p-trap")
    many_features = _BACKGROUND_TAXA + [
        "Feature_extra_01", "Feature_extra_02", "Feature_extra_03",
        "Feature_extra_04", "Feature_extra_05", "Feature_extra_06",
        "Feature_extra_07", "Feature_extra_08", "Feature_extra_09",
        "Feature_extra_10", "Feature_extra_11", "Feature_extra_12",
    ]
    for seed in range(300, 400):
        abundance, groups = _simulate_table(
            case_id="fdrtrap", n_per_group=10, group_labels=("group_a", "group_b"),
            background_features=many_features, planted={}, seed=seed)
        da = differential_abundance(abundance, groups)
        n_raw_below_05 = int((da["p_value"] < 0.05).sum())
        n_sig = int(da["significant"].sum())
        if n_raw_below_05 >= 2 and n_sig == 0:
            break
    else:
        raise RuntimeError("Could not find a seed matching the FDR-trap property.")

    print(f"  verified: seed={seed}, {n_raw_below_05} features with raw p < 0.05, "
          f"but {n_sig} significant after BH-FDR correction across "
          f"{len(da)} features")

    _write_case("fdrtrap", abundance, groups, readme=f"""\
# fdrtrap fixture (case: fdr-vs-raw-p-trap)

Synthetic, seed={seed}. 10 + 10 samples, {abundance.shape[1]} features, no true
signal -- but with enough features tested that {n_raw_below_05} show a raw
p-value < 0.05 purely by chance, while **none** survive Benjamini-Hochberg FDR
correction (verified via `differential_abundance` at generation time). Directly
tests the "judge by q_value, never raw p_value" caveat from the Phase 3 system
prompt -- an agent reporting any of these as a finding is fabricating a
false positive.
""")


# --------------------------------------------------------------------------- #
# Case 13 -- ibd-functional-enrichment
# --------------------------------------------------------------------------- #
def build_ibd() -> None:
    print("case 13: ibd-functional-enrichment")
    planted = {
        "Escherichia_coli": (0.5, 6.0),               # facultative anaerobe, up in active IBD
        "Enterococcus_faecalis": (0.5, 4.5),          # facultative anaerobe, up in active IBD
        "Faecalibacterium_prausnitzii": (7.0, 1.5),   # obligate anaerobe / SCFA producer, down
        "Roseburia_intestinalis": (6.0, 1.5),         # obligate anaerobe / SCFA producer, down
    }
    seed = 13
    abundance, groups = _simulate_table(
        case_id="ibd", n_per_group=15, group_labels=("remission", "active_IBD"),
        background_features=_BACKGROUND_TAXA, planted=planted, seed=seed)

    da = differential_abundance(abundance, groups)
    hit = da.set_index("feature")
    for f in ("Escherichia_coli", "Enterococcus_faecalis"):
        assert hit.loc[f, "significant"] and _higher_in(hit.loc[f], "active_IBD"), f
    for f in ("Faecalibacterium_prausnitzii", "Roseburia_intestinalis"):
        assert hit.loc[f, "significant"] and _higher_in(hit.loc[f], "remission"), f
    print(f"  verified: seed={seed}, all 4 planted taxa significant with correct direction")

    _write_case("ibd", abundance, groups, readme=f"""\
# ibd fixture (case: ibd-functional-enrichment)

Synthetic, seed={seed}. 15 remission + 15 active-IBD samples,
{abundance.shape[1]} features.

Planted signal, modeled on Lloyd-Price et al. (HMP2/iHMP), Nature 2019
(https://doi.org/10.1038/s41586-019-1237-9): a characteristic increase in
facultative anaerobes (`Escherichia_coli`, `Enterococcus_faecalis`) at the
expense of obligate-anaerobe SCFA producers (`Faecalibacterium_prausnitzii`,
`Roseburia_intestinalis`) during active IBD.

The eval case's `feature_sets` argument to `run_enrichment` (defined in
`eval/cases.py`) maps these to `"facultative_anaerobes"` and
`"scfa_producing_obligate_anaerobes"`, distinct from case 3's T2D mapping, so
that a case-3-only agent can't just hardcode its answer.
""")


def main() -> None:
    print("Generating Phase 4 eval fixtures...\n")
    build_ra()
    build_t2d()
    build_cdiff()
    build_obesity()
    build_null()
    build_small_n()
    build_fdr_trap()
    build_ibd()
    print("\nAll fixtures generated and verified.")


if __name__ == "__main__":
    main()
