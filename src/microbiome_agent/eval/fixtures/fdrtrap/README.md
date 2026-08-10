# fdrtrap fixture (case: fdr-vs-raw-p-trap)

Synthetic, seed=300. 10 + 10 samples, 30 features, no true
signal -- but with enough features tested that 3 show a raw
p-value < 0.05 purely by chance, while **none** survive Benjamini-Hochberg FDR
correction (verified via `differential_abundance` at generation time). Directly
tests the "judge by q_value, never raw p_value" caveat from the Phase 3 system
prompt -- an agent reporting any of these as a finding is fabricating a
false positive.
