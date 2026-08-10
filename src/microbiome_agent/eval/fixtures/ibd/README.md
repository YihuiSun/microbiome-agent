# ibd fixture (case: ibd-functional-enrichment)

Synthetic, seed=13. 15 remission + 15 active-IBD samples,
22 features.

Planted signal, modeled on Lloyd-Price et al. (HMP2/iHMP), Nature 2019
(https://doi.org/10.1038/s41586-019-1237-9): a characteristic increase in
facultative anaerobes (`Escherichia_coli`, `Enterococcus_faecalis`) at the
expense of obligate-anaerobe SCFA producers (`Faecalibacterium_prausnitzii`,
`Roseburia_intestinalis`) during active IBD.

The eval case's `feature_sets` argument to `run_enrichment` (defined in
`eval/cases.py`) maps these to `"facultative_anaerobes"` and
`"scfa_producing_obligate_anaerobes"`, distinct from case 3's T2D mapping, so
that a case-3-only agent can't just hardcode its answer.
