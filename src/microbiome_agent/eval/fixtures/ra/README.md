# ra fixture (case: ra-prevotella-expansion)

Synthetic, seed=2. 15 control + 15 RA samples, 20 features.

Planted signal, modeled on Scher et al., eLife 2013
(https://doi.org/10.7554/eLife.01202):
- `Prevotella_copri` expanded in RA (up).
- `Bacteroides_fragilis` reduced in RA (down).

Both verified significant (q < 0.05) via `differential_abundance` at
generation time; see `generate_fixtures.py:build_ra`.
