# Synthetic example dataset

These two files (`abundance.csv`, `metadata.csv`) are **synthetic** — generated
by a fixed-seed script for testing the pipeline. They are **not** from any real
study and should never be cited as real data.

They contain a deliberately planted signal: *Fusobacterium nucleatum* is elevated
in the `CRC` group relative to `control`, echoing a well-known published
association. That gives the analysis tools (and later the agent and its eval
harness) a known-correct answer to hit.

For real data, run `scripts/export_curatedMetagenomicData.R` and point
`load_dataset()` at the exported CSVs.
