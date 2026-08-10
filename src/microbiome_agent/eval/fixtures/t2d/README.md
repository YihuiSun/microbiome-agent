# t2d fixture (case: t2d-dysbiosis-enrichment)

Synthetic, seed=3. 15 control + 15 T2D samples, 22 features.

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
