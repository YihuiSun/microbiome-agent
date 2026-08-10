# obesity fixture (case: obesity-mouse-caveat)

Synthetic, seed=100. 10 lean + 10 obese samples, 20 features.

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
