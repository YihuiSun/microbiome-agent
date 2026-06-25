#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# ONE-TIME data prep: export a curatedMetagenomicData study to the two CSVs
# that this project's Python loader expects.
#
# You run this ONCE per study you want. After that, your Python project never
# touches R again -- it just reads the CSVs. That's the whole point: R is a
# data faucet here, not a runtime dependency.
#
# First-time setup (in R, takes a while -- Bioconductor is heavy):
#   install.packages("BiocManager")
#   BiocManager::install("curatedMetagenomicData")
#
# Then run from your shell:
#   Rscript scripts/export_curatedMetagenomicData.R
#
# Output:
#   data/<study>_abundance.csv   (sample_id + one column per species)
#   data/<study>_metadata.csv    (sample_id + clinical variables)
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(curatedMetagenomicData)
  library(dplyr)
  library(tibble)
})

# --- pick a study here -------------------------------------------------------
# This colorectal-cancer cohort is a good first choice: it's a published
# case-control study, so the literature tells you which taxa SHOULD differ --
# exactly what you want as ground truth for the eval harness later.
study_pattern <- "ZellerG_2014.relative_abundance"
out_prefix    <- "ZellerG_2014"
group_var     <- "study_condition"   # CRC vs control in this cohort
# -----------------------------------------------------------------------------

dir.create("data", showWarnings = FALSE)

message("Fetching ", study_pattern, " ...")
tse <- curatedMetagenomicData(study_pattern, dryrun = FALSE, rownames = "short")[[1]]

# Abundance: rows = species, cols = samples in the object; we want the transpose
# (samples as rows) to match the Python loader's expected orientation.
abund <- assay(tse) |>
  t() |>
  as.data.frame() |>
  rownames_to_column("sample_id")

# Metadata: keep the sample id + a few useful clinical columns if present.
md <- as.data.frame(colData(tse)) |>
  rownames_to_column("sample_id")
keep <- intersect(c("sample_id", group_var, "age", "gender", "country"),
                  colnames(md))
md <- md[, keep, drop = FALSE]

abund_path <- file.path("data", paste0(out_prefix, "_abundance.csv"))
md_path    <- file.path("data", paste0(out_prefix, "_metadata.csv"))
write.csv(abund, abund_path, row.names = FALSE)
write.csv(md,    md_path,    row.names = FALSE)

message("Wrote:\n  ", abund_path, "\n  ", md_path)
message("In Python:\n  load_dataset('", abund_path, "', '", md_path, "')")
