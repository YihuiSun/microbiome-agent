# Phase 4 — Evaluation Harness Design (Draft for Review)

Status: **draft, not yet implemented.** Nothing in this doc has been coded. The
goal is to agree on the test cases and scoring rubric before writing `eval.py`.

Note on grounding: source is now confirmed directly (you connected the local
`src/microbiome_agent` folder). All signatures below are read from the actual
files, not inferred. Confirmed MCP tool signatures:

- `load_dataset(abundance_path, metadata_path, sample_id_col="sample_id")`
- `load_example_dataset()` — no args; loads the bundled dataset with the
  planted Fusobacterium/CRC signal.
- `dataset_summary(dataset_id, max_levels=12)`
- `run_differential_abundance(dataset_id, group_column, fdr_alpha=0.05, top_n=50)`
  — result columns include `feature`, `log2_fold_change`, `p_value`,
  `q_value`, `significant` (`q_value < fdr_alpha`).
- `run_alpha_diversity(dataset_id, group_column=None, base=2.0)` — reports
  values only, confirmed by the docstring: "it does not itself test whether
  groups differ."
- `run_beta_diversity(dataset_id, group_column, metric="braycurtis",
  permutations=999, seed=None)` — response includes a `note` field
  (literal key, not just narrative text) when any group has fewer than 4
  samples, flagging the PERMANOVA resolution limit.
- `run_enrichment(differential_analysis_id, feature_sets, fdr_alpha=0.05,
  top_n=50)` — **important correction from the draft below:** this tool
  always pulls its hit list from a prior `run_differential_abundance` call by
  `analysis_id`. There is no way to run enrichment standalone through the MCP
  layer without a differential-abundance step first (the underlying
  `over_representation_analysis(feature_sets, hits, universe)` function does
  take hits/universe directly, but the MCP tool wrapper doesn't expose that
  path). `feature_sets` (`dict[str, list[str]]`) must be supplied by the
  caller in the question/prompt — there's no catalogue loader yet (a known,
  intentional Phase 1/2 limitation).
- `generate_report(title, analysis_ids, output_dir, top_n=20)` — takes a list
  of `analysis_id`s (any mix of differential/alpha/beta/enrichment).
- Unknown-handle errors are real `ValueError`s with message text like
  `"Unknown dataset_id 'xyz'. Call load_dataset ... first; known: [...]"` —
  useful for the deterministic check in case 12.

---

## 1. Goals

Per the handoff: curate 8–15 test cases with known answers, score each on two
levels (deterministic + LLM-as-judge), and compute three summary metrics
(tool-selection accuracy, incorrect-claim rate, run-to-run consistency). This
is the piece that turns the project into a portfolio-credible eval, so the
cases need real ground truth, not invented numbers.

## 2. Test case schema

Each case is a dict (or small dataclass) with:

- `id` — short slug, e.g. `crc-fuso-enrichment`
- `category` — one of `differential_abundance`, `alpha_diversity`,
  `beta_diversity`, `enrichment`, `chained`, `negative_control`, `edge_case`
- `dataset` — which dataset the agent should analyze: `example` (the bundled
  synthetic CRC dataset from Phase 1) or a case-specific synthetic dataset we
  construct to mimic a published study's reported effect
- `question` — the natural-language prompt fed to the agent
- `expected_tools` — ordered (or set, if order-flexible) list of MCP tool
  names the agent should call, e.g. `["load_example_dataset",
  "run_differential_abundance"]`
- `deterministic_checks` — a list of pass/fail assertions runnable without an
  LLM (see §4)
- `ground_truth` — the known answer plus its literature source (citation +
  DOI) or, for synthetic/edge cases, the planted signal
- `notes` — caveats the agent is expected to surface (FDR vs. raw p, small-n
  PERMANOVA limit, etc.), used by the LLM-judge rubric

## 3. Candidate test cases

### 3a. Literature-grounded cases (real published findings)

These need synthetic abundance tables constructed so the *direction* of the
effect matches the cited paper (we are not re-hosting the original patient
data — these are synthetic datasets with a planted signal modeled on the
published finding, same approach as the Phase 1 `example_dataset()`).

1. **`crc-fuso-enrichment`** — Differential abundance. Planted signal:
   *Fusobacterium* enriched in CRC vs. control. Grounded in Castellarin et al.,
   *Genome Research* 2012 ([DOI](https://doi.org/10.1101/gr.126516.111)) and
   the companion paper Kostic et al., *Genome Research* 2012
   ([DOI](https://doi.org/10.1101/gr.126573.111)), which also reports
   Bacteroidetes/Firmicutes depletion in tumors — gives a two-taxon check.
   Confirmed via `ROADMAP.md`/`HANDOFF_through_phase2.md`: the bundled
   `example_dataset()` already plants exactly this Fusobacterium/CRC signal, so
   this case needs **no new fixture** — it's `dataset: example`, reusing what
   Phase 1 already ships. Expected tool: `load_example_dataset` →
   `run_differential_abundance`. Check: *Fusobacterium* appears with q < 0.05
   and positive log2FC in the CRC group.

2. **`ra-prevotella-expansion`** — Differential abundance. Planted signal:
   *Prevotella copri* expanded, *Bacteroides* reduced, in new-onset untreated
   rheumatoid arthritis. Grounded in Scher et al., *eLife* 2013
   ([DOI](https://doi.org/10.7554/eLife.01202)). Checks both directions in one
   case (one taxon up, one down) — good for catching an agent that only
   reports the "interesting" direction.

3. **`t2d-dysbiosis-enrichment`** — Chained: differential abundance +
   enrichment. Planted signal: decreased butyrate-producing taxa, increased
   opportunistic pathogens, functional enrichment for sulfate reduction and
   oxidative-stress-resistance pathways in type 2 diabetes. Grounded in Qin et
   al., *Nature* 2012 ([DOI](https://doi.org/10.1038/nature11450)). This is
   the best candidate for testing whether the agent chains
   `run_differential_abundance` → `run_enrichment` unprompted, since the
   question only asks "what's different and why does it matter
   functionally."

4. **`crc-beta-diversity-classification`** — Beta diversity / PERMANOVA.
   Grounded in Zeller et al., *Molecular Systems Biology* 2014
   ([DOI](https://doi.org/10.15252/msb.20145645)): CRC and control fecal
   microbiomes separate significantly by community composition, and the
   functional shift is from fiber degradation toward host-carbohydrate/amino-
   acid utilization plus increased LPS metabolism. Check: significant
   PERMANOVA (p < 0.05) and that report text doesn't overclaim causation.

5. **`cdiff-diversity-collapse`** — Alpha diversity. Grounded in Chang et al.,
   *Journal of Infectious Diseases* 2008
   ([DOI](https://doi.org/10.1086/533096) — note: verify DOI against PMID
   18199029 when repo/network access allows re-check), reporting markedly
   reduced fecal microbiome diversity in recurrent *C. difficile*-associated
   diarrhea. This is a strong, unambiguous, near-universal effect — good
   "should be easy" sanity case. Check: Shannon diversity lower in cases,
   agent does **not** claim a group-difference significance test (per the
   Phase 3 caveat that alpha diversity reports values only).

6. **`obesity-mouse-caveat`** — Differential abundance, deliberately
   contested/mixed evidence. Grounded in Turnbaugh et al., *Nature* 2006
   ([DOI](https://doi.org/10.1038/nature05414)): Firmicutes/Bacteroidetes
   shift and increased energy-harvest capacity — but this was a **mouse**
   model, and the human Firmicutes/Bacteroidetes obesity signature has not
   reliably replicated in later human cohorts. This case exists specifically
   to test the "no significant difference found is a valid result; do not
   overstate" caveat and whether the agent flags the mouse-vs-human
   translation gap when asked to generalize. Ground truth here is deliberately
   double-sided: the deterministic check only requires the tool ran
   correctly; the LLM-judge check evaluates whether the interpretation
   overclaims.

### 3b. Synthetic / edge cases (no literature grounding needed — testing agent behavior, not biology)

7. **`no-signal-null-result`** — Differential abundance on a dataset with
   *no* planted signal (random labels). Correct answer: no taxa pass FDR
   correction. Tests whether the agent reports "no significant difference"
   honestly instead of cherry-picking a marginal p-value.

8. **`permanova-small-n`** — Beta diversity with n < 4 samples per group.
   Tests whether the agent surfaces the small-n PERMANOVA resolution-limit
   caveat from the system prompt, rather than reporting the p-value at face
   value.

9. **`fdr-vs-raw-p-trap`** — Differential abundance on a dataset engineered so
   several taxa have raw p < 0.05 but none survive BH correction. Directly
   tests the "judge by q-value, never raw p-value" caveat — an agent that
   fails this is fabricating a false-positive finding.

10. **`tool-selection-summary-only`** — Question that only requires
    `dataset_summary` (e.g., "how many samples and features are in this
    dataset?"). Tests that the agent doesn't over-call tools (e.g., doesn't
    needlessly run differential abundance when just asked for sample counts).
    Good negative control for tool-selection accuracy.

11. **`chained-full-pipeline`** — "Analyze the example dataset: differential
    abundance, alpha and beta diversity by study_condition, then generate a
    report." This is the exact CLI example from the Phase 3 handoff. Expected
    tools: `load_example_dataset` → `run_differential_abundance` →
    `run_alpha_diversity` → `run_beta_diversity` → `generate_report`. Good
    end-to-end regression case since it mirrors the documented CLI usage
    directly.

12. **`mcp-tool-error-recovery`** — Question that references a nonexistent
    `dataset_id` (confirmed: `_get_dataset` raises `ValueError` with message
    `"Unknown dataset_id 'xyz'. Call load_dataset ... first; known: [...]"`)
    or a bad `group_column` name (raises via `Dataset.groups()`: `"No
    metadata column 'xyz'. Available: [...]"`). Tests that the agent surfaces
    the `[tool error] ...` string gracefully — ideally relaying the
    "Available: [...]" list back to the user — rather than hallucinating a
    result. (Mirrors the Phase 3 unit test for tool-error forwarding, but at
    the eval/behavioral level rather than the mocked-SDK level.)

13. **`ibd-functional-enrichment`** — Second enrichment case, on a different
    dataset and pathway set than case 3. Planted signal: gut microbial
    community during active IBD shows a characteristic increase in
    facultative anaerobes at the expense of obligate anaerobes, with
    functional disruption of short-chain-fatty-acid and bile-acid-related
    pathways. Grounded in Lloyd-Price et al. (HMP2/iHMP), *Nature* 2019
    ([DOI](https://doi.org/10.1038/s41586-019-1237-9)). Correction from an
    earlier draft of this doc: `run_enrichment` always requires a prior
    `run_differential_abundance` `analysis_id` — there's no MCP path to run
    enrichment standalone, so this case can't isolate the enrichment stage
    from differential abundance the way I originally described. What it does
    still add over case 3: a different `feature_sets` mapping and a
    dataset where the *expected* differential-abundance hits are facultative-
    vs-obligate-anaerobe taxa rather than T2D's butyrate-producer/pathogen
    split, catching an agent that hardcodes assumptions from case 3 rather
    than reading the actual hit list each time. The question supplies the
    `feature_sets` dict inline in the prompt, since there's no catalogue
    loader yet.

14. **`ra-chained-full-pipeline`** — Second full end-to-end case (diff
    abundance → alpha diversity → beta diversity → report), run on the RA
    dataset from case 2 instead of the bundled example dataset. Tests that
    the chaining behavior from case 11 generalizes beyond the one dataset
    it's bundled with, and gives a second literature-grounded report to
    sanity-check with the LLM-judge (does the generated report correctly
    narrate the Prevotella/Bacteroides finding, not just run without
    crashing).

14 cases total: 7 literature-grounded (cases 1–6, 13) + 7 synthetic/behavioral
(cases 7–12, 14).

## 4. Deterministic checks (no LLM needed)

Per case, a small set of assertions runnable directly against the agent's
tool-call trace and returned report, e.g.:

- **Right tool(s) called** — set/order match against `expected_tools`.
- **FDR applied, not raw p** — parse the returned differential-abundance
  table and confirm ranking/filtering used the `q_value` column.
- **Right taxa flagged** — the planted-signal taxon appears in the
  significant-hits table (or is absent, for the null case).
- **Caveat present when required** — for the small-n PERMANOVA case, this is
  no longer a fuzzy text check: `run_beta_diversity` emits a literal `note`
  key in its response whenever a group has fewer than 4 samples, so the
  deterministic check is just "does the tool response contain a `note` field"
  — then the LLM-judge pass (§5) checks whether the agent's final answer
  actually *repeats* that caveat to the user rather than silently dropping
  it. For cases without a structured field (e.g. the mouse-vs-human
  translation gap in case 6), fall back to a keyword check on the final
  answer text as a coarse pre-filter.

## 5. LLM-as-judge check

A second Claude call (separate from the agent being evaluated) scores the
agent's final natural-language answer against the `ground_truth` and `notes`
fields, on a rubric such as:

- Biological interpretation soundness (1–5): does the stated conclusion match
  the planted/literature signal without overclaiming causality or ignoring
  effect size?
- Overstatement flag (yes/no): does the agent claim significance/certainty
  the data doesn't support?
- Caveat correctness (1–5): for cases with `notes`, did the agent surface the
  *right* caveat, not just any caveat?

Judge prompt should include the case's `ground_truth`, `notes`, and the
agent's full final answer — not the intermediate tool-call trace, since we
want to judge the answer a user would actually read.

## 6. Metrics (aggregate scorecard)

- **Tool-selection accuracy** — fraction of cases where `expected_tools`
  matches the actual call trace (exact-set match; order matters only for
  `chained` category cases).
- **Incorrect-claim rate** — fraction of cases where the LLM-judge
  overstatement flag is "yes," or a deterministic check fails.
- **Run-to-run consistency** — run each case N times (e.g., N=3) at
  temperature > 0, and report the fraction of cases where the deterministic
  checks pass on all N runs. Flags flaky tool selection or non-reproducible
  interpretation.

`eval.py` output: one row per case (pass/fail per deterministic check, judge
scores) plus the three aggregate metrics printed as a scorecard.

## 7. Decisions (locked in)

- **Case count: 14** (7 literature-grounded + 7 synthetic/behavioral, per §3).
- **Fixture strategy: checked-in fixture CSVs.** Each case's synthetic
  abundance/metadata tables are generated once (planted signal + noise, same
  method as Phase 1's `example_dataset()`) and committed under something like
  `tests/eval_fixtures/<case_id>/{abundance.csv,metadata.csv}`, so the eval is
  deterministic across machines and the planted signal for each case is
  inspectable/diffable in code review — not regenerated with a fresh random
  seed on every run.
- **Judge model: Haiku.** The agent-under-test keeps whatever model Phase 3
  configured (`claude-opus-4-5` per the handoff); the LLM-judge pass (§5) uses
  a cheaper Haiku call, since judging is a simpler classification/rubric task
  than the analysis itself and this keeps repeated eval runs (esp. the N=3
  consistency runs in §6) cheap.

## 8. Status: source-confirmed, ready to code

With the repo folder connected and all five source files read directly
(`differential_abundance.py`, `diversity.py`, `enrichment.py`, `report.py`,
`mcp_server/server.py`, `datasets/loaders.py`), every signature this eval
needs is now confirmed rather than inferred — see the note at the top of this
doc. The one real design correction that came out of reading source: case 13
can't isolate the enrichment tool the way originally described, since
`run_enrichment` always requires a prior differential-abundance
`analysis_id` (fixed in §3b above). No remaining open questions before
writing `eval.py`.

Sources: all literature citations above are from PubMed; DOIs are linked
inline per citation.
