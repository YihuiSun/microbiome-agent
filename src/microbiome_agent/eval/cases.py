"""The 14 curated Phase 4 eval cases.

Mirrors `phase4_eval_design.md` (in the outputs from that design session) --
see that doc for the literature grounding and rationale behind each case. This
module is the executable version of that doc: every case here is an
`EvalCase` the runner can actually fire at the live agent.

Fixture datasets for cases that need one live under `eval/fixtures/<case_id>/`
(built and verified by `eval/fixtures/generate_fixtures.py`). Cases 1, 4, 10,
11, and 12 reuse the bundled `datasets/example/` dataset instead (via the
`load_example_dataset` tool) -- no fixture needed.
"""

from __future__ import annotations

from pathlib import Path

from microbiome_agent.eval import checks
from microbiome_agent.eval.types import EvalCase

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def fixture_paths(case_id: str) -> tuple[str, str]:
    d = FIXTURES_DIR / case_id
    return str(d / "abundance.csv"), str(d / "metadata.csv")


# --------------------------------------------------------------------------- #
# 1. crc-fuso-enrichment
# --------------------------------------------------------------------------- #
_case_1 = EvalCase(
    id="crc-fuso-enrichment",
    category="differential_abundance",
    question=(
        "Load the bundled example dataset and run differential abundance "
        "grouped by study_condition. Which taxa differ significantly "
        "between the CRC and control groups, and in which direction?"
    ),
    ground_truth=(
        "Fusobacterium nucleatum is significantly elevated in the CRC group. "
        "Grounded in Castellarin et al., Genome Research 2012 "
        "(https://doi.org/10.1101/gr.126516.111) and Kostic et al., Genome "
        "Research 2012 (https://doi.org/10.1101/gr.126573.111). (The bundled "
        "example dataset also happens to show Faecalibacterium_prausnitzii "
        "significantly lower in CRC -- a bonus incidental signal, not "
        "required for this case to pass.)"
    ),
    notes="Judge by q_value (BH-FDR), never raw p_value.",
    checks=[
        checks.expects_tool_calls(["load_example_dataset", "run_differential_abundance"]),
        checks.no_tool_errors(),
        checks.differential_feature_direction("Fusobacterium_nucleatum", "CRC"),
    ],
)


# --------------------------------------------------------------------------- #
# 2. ra-prevotella-expansion
# --------------------------------------------------------------------------- #
_ra_abundance, _ra_metadata = fixture_paths("ra")
_case_2 = EvalCase(
    id="ra-prevotella-expansion",
    category="differential_abundance",
    question=(
        f"Load the dataset at abundance file {_ra_abundance} and metadata "
        f"file {_ra_metadata}, then run differential abundance grouped by "
        "study_condition. Which taxa differ significantly between the RA "
        "and control groups, and in which direction?"
    ),
    ground_truth=(
        "Prevotella_copri is significantly expanded in RA; Bacteroides_"
        "fragilis is significantly reduced in RA. Grounded in Scher et al., "
        "eLife 2013 (https://doi.org/10.7554/eLife.01202)."
    ),
    notes="Checks both directions in one case -- catches an agent that only reports the 'interesting' direction.",
    checks=[
        checks.expects_tool_calls(["load_dataset", "run_differential_abundance"]),
        checks.no_tool_errors(),
        checks.differential_feature_direction("Prevotella_copri", "RA"),
        checks.differential_feature_direction("Bacteroides_fragilis", "control"),
    ],
)


# --------------------------------------------------------------------------- #
# 3. t2d-dysbiosis-enrichment
# --------------------------------------------------------------------------- #
_t2d_abundance, _t2d_metadata = fixture_paths("t2d")
_case_3 = EvalCase(
    id="t2d-dysbiosis-enrichment",
    category="chained",
    question=(
        f"Load the dataset at abundance file {_t2d_abundance} and metadata "
        f"file {_t2d_metadata}, run differential abundance grouped by "
        "study_condition, then test functional enrichment on the "
        "significant features using these feature sets: "
        '{"scfa_producing_bacteria": ["Roseburia_intestinalis", '
        '"Eubacterium_rectale", "Faecalibacterium_prausnitzii"], '
        '"opportunistic_pathogens": ["Escherichia_coli", '
        '"Enterococcus_faecalis", "Klebsiella_pneumoniae"]}. '
        "What's different between T2D and control, and what does it mean "
        "functionally?"
    ),
    ground_truth=(
        "Butyrate-producing taxa (Roseburia_intestinalis, Eubacterium_"
        "rectale) are decreased and opportunistic pathogens (Escherichia_"
        "coli, Enterococcus_faecalis) are increased in T2D. Grounded in Qin "
        "et al., Nature 2012 (https://doi.org/10.1038/nature11450). Because "
        "over-representation analysis here doesn't split by direction (the "
        "'hits' list is every significant feature, up or down), BOTH "
        "feature sets come back significantly enriched, verified "
        "empirically at fixture-generation time -- an agent correctly "
        "running the tool should report both as enriched, not just the "
        "'interesting-sounding' pathogen set."
    ),
    notes=(
        "Best case for testing whether the agent chains "
        "run_differential_abundance -> run_enrichment unprompted, since the "
        "question only asks what's different and what it means "
        "functionally, not to run enrichment explicitly step-by-step."
    ),
    checks=[
        checks.expects_tool_calls(
            ["load_dataset", "run_differential_abundance", "run_enrichment"]
        ),
        checks.no_tool_errors(),
        checks.differential_feature_direction("Roseburia_intestinalis", "control"),
        checks.differential_feature_direction("Escherichia_coli", "T2D"),
        checks.enrichment_set_significant("scfa_producing_bacteria", expect_significant=True),
        checks.enrichment_set_significant("opportunistic_pathogens", expect_significant=True),
    ],
)


# --------------------------------------------------------------------------- #
# 4. crc-beta-diversity-classification
# --------------------------------------------------------------------------- #
_case_4 = EvalCase(
    id="crc-beta-diversity-classification",
    category="beta_diversity",
    question=(
        "Load the bundled example dataset and test whether overall "
        "community composition (beta diversity) differs between the CRC "
        "and control groups."
    ),
    ground_truth=(
        "CRC and control fecal microbiomes separate significantly by "
        "community composition (verified: PERMANOVA pseudo-F=6.92, "
        "p=0.001 on this exact bundled dataset). Grounded in Zeller et al., "
        "Molecular Systems Biology 2014 (https://doi.org/10.15252/msb.20145645), "
        "which additionally reports a functional shift from fiber "
        "degradation toward host-carbohydrate/amino-acid utilization plus "
        "increased LPS metabolism -- not testable here since this project's "
        "enrichment tool isn't wired to a real pathway catalogue yet."
    ),
    notes="Check that the report text doesn't overclaim causation from a compositional association.",
    checks=[
        checks.expects_tool_calls(["load_example_dataset", "run_beta_diversity"]),
        checks.no_tool_errors(),
        checks.beta_diversity_significant(expect_significant=True),
    ],
)


# --------------------------------------------------------------------------- #
# 5. cdiff-diversity-collapse
# --------------------------------------------------------------------------- #
_cdiff_abundance, _cdiff_metadata = fixture_paths("cdiff")
_case_5 = EvalCase(
    id="cdiff-diversity-collapse",
    category="alpha_diversity",
    question=(
        f"Load the dataset at abundance file {_cdiff_abundance} and "
        f"metadata file {_cdiff_metadata}, then compute alpha diversity "
        "grouped by study_condition. How does diversity compare between "
        "the recurrent_CDI and healthy groups?"
    ),
    ground_truth=(
        "Fecal microbiome diversity is markedly reduced in recurrent "
        "C. difficile-associated diarrhea (verified: mean Shannon in this "
        "fixture is roughly 0.6 bits in recurrent_CDI vs ~4.2 bits in "
        "healthy). Grounded in Chang et al., J Infect Dis 2008 "
        "(PMID 18199029)."
    ),
    notes=(
        "Strong, unambiguous, near-universal effect -- a sanity case. The "
        "agent must NOT claim a group-difference significance test on this "
        "output, per the Phase 3 system prompt: alpha diversity reports "
        "values only, it does not itself test whether groups differ."
    ),
    checks=[
        checks.expects_tool_calls(["load_dataset", "run_alpha_diversity"]),
        checks.no_tool_errors(),
    ],
)


# --------------------------------------------------------------------------- #
# 6. obesity-mouse-caveat
# --------------------------------------------------------------------------- #
_obesity_abundance, _obesity_metadata = fixture_paths("obesity")
_case_6 = EvalCase(
    id="obesity-mouse-caveat",
    category="differential_abundance",
    question=(
        f"Load the dataset at abundance file {_obesity_abundance} and "
        f"metadata file {_obesity_metadata}, then run differential "
        "abundance grouped by study_condition to see whether taxa differ "
        "between the obese and lean groups. This dataset relates to the "
        "well-known finding that mouse gut microbiomes show an obesity-"
        "associated Firmicutes/Bacteroidetes shift with increased energy-"
        "harvest capacity -- does this human-style dataset confirm that?"
    ),
    ground_truth=(
        "No feature survives FDR correction in this fixture (verified at "
        "generation time: 0 of N significant, smallest q~0.6). The "
        "Firmicutes/Bacteroidetes obesity finding this question references "
        "is Turnbaugh et al., Nature 2006 (https://doi.org/10.1038/nature05414) "
        "-- a MOUSE model whose human replication has been inconsistent. "
        "Ground truth here is deliberately double-sided: the deterministic "
        "check only requires the tool ran and correctly reports no "
        "significant hits; whether the agent avoids overclaiming a mouse-to-"
        "human generalization the data doesn't support is for the LLM judge."
    ),
    notes=(
        "Tests the 'no significant difference found is a valid result; do "
        "not overstate' caveat, and specifically whether the agent flags "
        "the mouse-vs-human translation gap rather than treating the "
        "question's premise as established fact."
    ),
    checks=[
        checks.expects_tool_calls(["load_dataset", "run_differential_abundance"]),
        checks.no_tool_errors(),
        checks.no_significant_features(),
    ],
)


# --------------------------------------------------------------------------- #
# 7. no-signal-null-result
# --------------------------------------------------------------------------- #
_null_abundance, _null_metadata = fixture_paths("null")
_case_7 = EvalCase(
    id="no-signal-null-result",
    category="negative_control",
    question=(
        f"Load the dataset at abundance file {_null_abundance} and "
        f"metadata file {_null_metadata}, then run differential abundance "
        "grouped by study_condition. What differs between group_a and "
        "group_b?"
    ),
    ground_truth=(
        "Nothing -- this fixture has no true signal (labels carry no "
        "underlying difference), verified at generation time: 0 "
        "significant features after FDR correction."
    ),
    notes="Tests whether the agent honestly reports 'no significant difference' instead of cherry-picking a marginal p-value.",
    checks=[
        checks.expects_tool_calls(["load_dataset", "run_differential_abundance"]),
        checks.no_tool_errors(),
        checks.no_significant_features(),
    ],
)


# --------------------------------------------------------------------------- #
# 8. permanova-small-n
# --------------------------------------------------------------------------- #
_smalln_abundance, _smalln_metadata = fixture_paths("smalln")
_case_8 = EvalCase(
    id="permanova-small-n",
    category="beta_diversity",
    question=(
        f"Load the dataset at abundance file {_smalln_abundance} and "
        f"metadata file {_smalln_metadata}, then test whether community "
        "composition differs between the control and disease groups."
    ),
    ground_truth=(
        "Only 3 samples per group. The community really is well-separated "
        "(a Fusobacterium/Bacteroides-style shift is planted), but "
        "run_beta_diversity's own response includes a `note` field flagging "
        "that PERMANOVA's p-value is resolution-limited below 4 samples per "
        "group."
    ),
    notes=(
        "Tests whether the agent surfaces the small-n PERMANOVA resolution-"
        "limit caveat from the system prompt, rather than reporting the "
        "p-value at face value."
    ),
    checks=[
        checks.expects_tool_calls(["load_dataset", "run_beta_diversity"]),
        checks.no_tool_errors(),
        checks.beta_diversity_note_present(),
    ],
)


# --------------------------------------------------------------------------- #
# 9. fdr-vs-raw-p-trap
# --------------------------------------------------------------------------- #
_fdrtrap_abundance, _fdrtrap_metadata = fixture_paths("fdrtrap")
_case_9 = EvalCase(
    id="fdr-vs-raw-p-trap",
    category="differential_abundance",
    question=(
        f"Load the dataset at abundance file {_fdrtrap_abundance} and "
        f"metadata file {_fdrtrap_metadata}, then run differential "
        "abundance grouped by study_condition. Which features are "
        "different between group_a and group_b?"
    ),
    ground_truth=(
        "None -- several features show a raw p-value under 0.05 purely by "
        "chance (multiple testing across many features), but zero survive "
        "Benjamini-Hochberg FDR correction (verified at generation time)."
    ),
    notes=(
        "Directly tests the 'judge by q_value, never raw p_value' caveat -- "
        "an agent reporting any of these nominal hits as a finding is "
        "fabricating a false positive."
    ),
    checks=[
        checks.expects_tool_calls(["load_dataset", "run_differential_abundance"]),
        checks.no_tool_errors(),
        checks.no_significant_features(),
    ],
)


# --------------------------------------------------------------------------- #
# 10. tool-selection-summary-only
# --------------------------------------------------------------------------- #
_case_10 = EvalCase(
    id="tool-selection-summary-only",
    category="negative_control",
    question=(
        "Load the bundled example dataset and tell me how many samples and "
        "features it has, and what metadata columns are available."
    ),
    ground_truth=(
        "24 samples, 8 features, metadata columns study_condition, age, "
        "sex -- answerable from load_example_dataset's own summary plus "
        "dataset_summary, with no analysis tool required."
    ),
    notes="Negative control for tool-selection accuracy -- an agent that reflexively runs differential abundance here is over-calling tools.",
    checks=[
        checks.only_tools_called(["load_example_dataset", "dataset_summary"]),
        checks.no_tool_errors(),
    ],
)


# --------------------------------------------------------------------------- #
# 11. chained-full-pipeline
# --------------------------------------------------------------------------- #
_case_11 = EvalCase(
    id="chained-full-pipeline",
    category="chained",
    question=(
        "Analyse the example dataset. Run differential abundance and alpha "
        "and beta diversity grouped by study_condition, then generate a "
        "report."
    ),
    ground_truth=(
        "The exact CLI example from the Phase 3 handoff. Expected chain: "
        "load_example_dataset -> run_differential_abundance -> "
        "run_alpha_diversity -> run_beta_diversity -> generate_report. On "
        "this dataset: Fusobacterium_nucleatum up in CRC (q=0.0003), "
        "PERMANOVA significant (p=0.001), alpha diversity nearly identical "
        "between groups (~2.9 vs ~2.8 bits)."
    ),
    notes="End-to-end regression case mirroring documented CLI usage directly.",
    checks=[
        checks.expects_tool_calls([
            "load_example_dataset", "run_differential_abundance",
            "run_alpha_diversity", "run_beta_diversity", "generate_report",
        ]),
        checks.no_tool_errors(),
        checks.differential_feature_direction("Fusobacterium_nucleatum", "CRC"),
        checks.beta_diversity_significant(expect_significant=True),
    ],
)


# --------------------------------------------------------------------------- #
# 12. mcp-tool-error-recovery
# --------------------------------------------------------------------------- #
_case_12 = EvalCase(
    id="mcp-tool-error-recovery",
    category="edge_case",
    question=(
        "Load the bundled example dataset and run differential abundance "
        "grouped by the metadata column 'diagnosis_status'."
    ),
    ground_truth=(
        "There is no metadata column named 'diagnosis_status' in the "
        "bundled example dataset -- the available column is "
        "'study_condition' (plus age, sex). Two behaviors are equally "
        "correct here: (a) the agent calls run_differential_abundance "
        "directly, Dataset.groups() raises ValueError('No metadata "
        "column ...'), forwarded to the agent as '[tool error] ...', and "
        "it recovers gracefully; or (b) the agent first inspects "
        "dataset_summary, notices 'diagnosis_status' isn't among the "
        "available columns, and never attempts run_differential_abundance "
        "at all -- avoiding the error rather than recovering from it. "
        "Either way, the correct behavior is to surface the problem "
        "clearly, ideally naming the actual available column, rather than "
        "fabricating a result."
    ),
    notes=(
        "Mirrors the Phase 3 unit test for tool-error forwarding, but at "
        "the eval/behavioral level rather than the mocked-SDK level. A "
        "live run showed the agent taking the proactive path (b) -- "
        "checking dataset_summary first and never calling "
        "run_differential_abundance -- which is valid and arguably "
        "better than the reactive path (a) this case originally assumed "
        "was the only correct behavior."
    ),
    checks=[
        checks.expects_tool_calls(["load_example_dataset"]),
        checks.expects_graceful_bad_input_handling(
            "load_example_dataset", "run_differential_abundance"
        ),
        checks.final_answer_contains(
            ["study_condition", "diagnosis_status", "column", "available", "doesn't exist", "couldn't find"],
            mode="any",
        ),
    ],
)


# --------------------------------------------------------------------------- #
# 13. ibd-functional-enrichment
# --------------------------------------------------------------------------- #
_ibd_abundance, _ibd_metadata = fixture_paths("ibd")
_case_13 = EvalCase(
    id="ibd-functional-enrichment",
    category="enrichment",
    question=(
        f"Load the dataset at abundance file {_ibd_abundance} and metadata "
        f"file {_ibd_metadata}, run differential abundance grouped by "
        "study_condition, then test functional enrichment on the "
        "significant features using these feature sets: "
        '{"facultative_anaerobes": ["Escherichia_coli", '
        '"Enterococcus_faecalis", "Klebsiella_pneumoniae"], '
        '"scfa_producing_obligate_anaerobes": ["Faecalibacterium_prausnitzii", '
        '"Roseburia_intestinalis", "Eubacterium_rectale"]}. '
        "What's the functional picture during active IBD?"
    ),
    ground_truth=(
        "Facultative anaerobes (Escherichia_coli, Enterococcus_faecalis) "
        "increase at the expense of obligate-anaerobe SCFA producers "
        "(Faecalibacterium_prausnitzii, Roseburia_intestinalis) during "
        "active IBD. Grounded in Lloyd-Price et al. (HMP2/iHMP), Nature "
        "2019 (https://doi.org/10.1038/s41586-019-1237-9). As in case 3, "
        "both feature sets come back significantly enriched (verified at "
        "fixture-generation time), since ORA here doesn't split by "
        "direction."
    ),
    notes=(
        "Second enrichment case on a different dataset and feature-set "
        "mapping than case 3, catching an agent that hardcodes assumptions "
        "from one case rather than reading the actual hit list each time."
    ),
    checks=[
        checks.expects_tool_calls(
            ["load_dataset", "run_differential_abundance", "run_enrichment"]
        ),
        checks.no_tool_errors(),
        checks.differential_feature_direction("Escherichia_coli", "active_IBD"),
        checks.differential_feature_direction("Faecalibacterium_prausnitzii", "remission"),
        checks.enrichment_set_significant("facultative_anaerobes", expect_significant=True),
        checks.enrichment_set_significant("scfa_producing_obligate_anaerobes", expect_significant=True),
    ],
)


# --------------------------------------------------------------------------- #
# 14. ra-chained-full-pipeline
# --------------------------------------------------------------------------- #
_case_14 = EvalCase(
    id="ra-chained-full-pipeline",
    category="chained",
    question=(
        f"Load the dataset at abundance file {_ra_abundance} and metadata "
        f"file {_ra_metadata}. Run differential abundance and alpha and "
        "beta diversity grouped by study_condition, then generate a report "
        "titled 'RA Microbiome Analysis'."
    ),
    ground_truth=(
        "Second full end-to-end case, on the RA dataset from case 2 rather "
        "than the bundled example dataset. Verified: PERMANOVA highly "
        "significant (pseudo-F=12.66, p=0.001, both groups n=15); alpha "
        "diversity nearly identical between groups (~4.01 vs ~4.02 bits, no "
        "meaningful difference); Prevotella_copri up / Bacteroides_fragilis "
        "down in RA, as in case 2."
    ),
    notes=(
        "Tests that the chaining behavior from case 11 generalizes beyond "
        "the one dataset it's bundled with, and gives the LLM judge a "
        "second literature-grounded report to check for narrative accuracy, "
        "not just successful execution."
    ),
    checks=[
        checks.expects_tool_calls([
            "load_dataset", "run_differential_abundance",
            "run_alpha_diversity", "run_beta_diversity", "generate_report",
        ]),
        checks.no_tool_errors(),
        checks.differential_feature_direction("Prevotella_copri", "RA"),
        checks.beta_diversity_significant(expect_significant=True),
    ],
)


ALL_CASES: list[EvalCase] = [
    _case_1, _case_2, _case_3, _case_4, _case_5, _case_6, _case_7,
    _case_8, _case_9, _case_10, _case_11, _case_12, _case_13, _case_14,
]

CASES_BY_ID: dict[str, EvalCase] = {c.id: c for c in ALL_CASES}
