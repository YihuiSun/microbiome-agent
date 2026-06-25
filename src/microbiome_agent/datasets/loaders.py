"""Loading microbiome abundance tables and their sample metadata.

This is the data-entry tool for the project. Everything downstream -- the
differential abundance test, diversity metrics, the agent itself -- starts from
a clean, aligned (abundance, metadata) pair. So this module's whole job is to
turn two CSV files into trustworthy, aligned pandas objects, and to fail loudly
if they don't line up. That guarantee is what lets every later tool assume its
input is sane.

The expected on-disk format (the same one the R export script in scripts/
produces, and the same one the bundled example/ files use):

    abundance.csv   first column = sample_id, remaining columns = features
                    (taxa). Values are abundances (relative or counts).
    metadata.csv    first column = sample_id, remaining columns = sample
                    variables (e.g. study_condition, age, sex).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Dataset:
    """An aligned microbiome dataset.

    Attributes
    ----------
    abundance:
        Samples-by-features table (rows = samples, columns = taxa).
    metadata:
        Samples-by-variables table, indexed by the same sample IDs.

    The two are guaranteed to share the exact same index in the same order,
    which is the precondition every analysis tool relies on.
    """

    abundance: pd.DataFrame
    metadata: pd.DataFrame

    def groups(self, column: str) -> pd.Series:
        """Return one metadata column as a grouping vector.

        Convenience for feeding straight into ``differential_abundance``:

            ds = load_dataset(...)
            differential_abundance(ds.abundance, ds.groups("study_condition"))
        """
        if column not in self.metadata.columns:
            raise ValueError(
                f"No metadata column {column!r}. "
                f"Available: {list(self.metadata.columns)}"
            )
        return self.metadata[column]


def load_dataset(
    abundance_path: str | Path,
    metadata_path: str | Path,
    *,
    sample_id_col: str = "sample_id",
    drop_all_zero_features: bool = True,
) -> Dataset:
    """Load and align an abundance table with its sample metadata.

    Parameters
    ----------
    abundance_path, metadata_path:
        Paths to the two CSV files described in the module docstring.
    sample_id_col:
        Name of the sample-identifier column present in both files.
    drop_all_zero_features:
        If True, features (taxa) that are zero in every sample are removed --
        they carry no signal and only inflate multiple-testing correction.

    Returns
    -------
    Dataset
        Aligned abundance and metadata, restricted to the samples present in
        *both* files, in a single consistent order.

    Raises
    ------
    FileNotFoundError
        If either path does not exist.
    ValueError
        If the id column is missing, ids are duplicated, abundances are
        non-numeric, or the two files share no samples.
    """
    abundance_path = Path(abundance_path)
    metadata_path = Path(metadata_path)
    for p in (abundance_path, metadata_path):
        if not p.exists():
            raise FileNotFoundError(f"No such file: {p}")

    abundance = pd.read_csv(abundance_path)
    metadata = pd.read_csv(metadata_path)

    for name, df in (("abundance", abundance), ("metadata", metadata)):
        if sample_id_col not in df.columns:
            raise ValueError(
                f"{name} file is missing the id column {sample_id_col!r}."
            )
        if df[sample_id_col].duplicated().any():
            raise ValueError(f"{name} file has duplicate sample ids.")

    abundance = abundance.set_index(sample_id_col)
    metadata = metadata.set_index(sample_id_col)

    # Abundance values must be numeric; a stray text cell is a common CSV bug.
    non_numeric = abundance.columns[
        ~abundance.apply(lambda c: pd.api.types.is_numeric_dtype(c))
    ]
    if len(non_numeric) > 0:
        raise ValueError(
            f"Non-numeric abundance columns: {list(non_numeric)}. "
            "Check for stray text or wrong delimiter."
        )

    # Align to the samples present in both files, preserving abundance order.
    shared = [s for s in abundance.index if s in set(metadata.index)]
    if not shared:
        raise ValueError(
            "Abundance and metadata share no sample ids -- are the id columns "
            "really the same identifiers?"
        )
    abundance = abundance.loc[shared]
    metadata = metadata.loc[shared]

    if drop_all_zero_features:
        keep = abundance.columns[(abundance != 0).any(axis=0)]
        abundance = abundance[keep]

    return Dataset(abundance=abundance, metadata=metadata)


def example_dataset() -> Dataset:
    """Load the bundled synthetic example dataset.

    NOTE: this data is *synthetic* -- generated for testing the pipeline, not
    drawn from any real study. It contains a deliberately planted signal
    (Fusobacterium nucleatum elevated in the CRC group, echoing a known
    published association) so that downstream tools have a known-correct answer
    to hit. For real analyses, export a study with scripts/
    export_curatedMetagenomicData.R and point load_dataset at the result.
    """
    here = Path(__file__).parent / "example"
    return load_dataset(here / "abundance.csv", here / "metadata.csv")
