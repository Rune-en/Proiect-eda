"""
Infrastructure layer – dataset loader.

Responsibility: read raw data from disk (an external resource) and return a
clean DataFrame that the domain layer can consume.  All file I/O lives here so
that domain services stay pure and testable without touching the filesystem.

DDD: this is an *infrastructure* concern – an adapter between the external
CSV file (a backing service in 12-factor terms) and the in-process domain
model.
"""

import logging
import os

import numpy as np
import pandas as pd

# Module-level logger – Factor XI: treat logs as event streams.
logger = logging.getLogger(__name__)

# Mapping from the raw string labels in the CSV to integer Gleason groups.
# Defined here rather than in domain so the infrastructure layer owns the
# translation from raw storage format → domain value object.
_GLEASON_LABEL_MAP: dict[str, int] = {
    "Normal": 0,
    "Group 1": 1,
    "Group 2": 2,
    "Group 3": 3,
    "Group 4": 4,
    "Group 5": 5,
}


def load_dataset(path: str, normalization_type: str = "log") -> pd.DataFrame:
    """Load and preprocess the raw gene-expression CSV.

    The raw CSV has genes as rows and samples as columns plus two metadata
    columns (``gene_id``, ``gene_type``).  This function:

    1. Drops the two metadata columns.
    2. Transposes the matrix so that *samples* become rows and *genes* become
       columns.
    3. Promotes the first row (gene names) to column headers.
    4. Drops the non-numeric metadata columns that appear after transposition
       (columns 1–28 in the transposed form).
    5. Encodes the ``Gleason Group`` target column as integers.
    6. Optionally normalises gene expression values (controlled by
       ``normalization_type``).

    Args:
        path: File-system path to the raw ``DB.csv``.
        normalization_type: Feature normalisation strategy to apply after
            encoding the target.  Accepted values:

            * ``"log"``  – apply a log₂(x + 1) transform to all gene
              expression columns (default).
            * ``"none"`` – skip normalisation; raw expression values are
              returned as-is.

    Returns:
        A :class:`pandas.DataFrame` indexed by ``sample_id`` where every
        column except ``Gleason Group`` is a numeric gene-expression feature
        and ``Gleason Group`` is the integer-encoded classification target.
    """
    logger.info("Loading raw dataset from '%s'", path)

    # Read the raw CSV; genes are rows, samples are columns.
    # low_memory=False forces a single dtype-inference pass so pandas does not
    # emit DtypeWarning for the wide mixed-type sample columns.
    df = pd.read_csv(path, low_memory=False)

    # Drop non-expression metadata columns that are not needed for modelling.
    df = df.drop(columns=["gene_id", "gene_type"])

    # Transpose: rows become samples, columns become genes.
    df = df.transpose()

    # The first row after transposition contains gene names – promote to header.
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    # Drop auxiliary metadata columns that end up in columns 1–28 post-transpose.
    df = df.drop(columns=df.columns[1:29])

    # Name the index to reflect domain semantics.
    df.index.names = ["sample_id"]

    # Encode the target label as an ordinal integer (0 = Normal, 1–5 = groups)
    # and explicitly cast to int64 so the column has a numeric dtype.
    df["Gleason Group"] = (
        df["Gleason Group"].replace(_GLEASON_LABEL_MAP).astype(int)
    )

    if normalization_type == "log":
        # Apply log2(x + 1) transform to normalise right-skewed gene expression counts.
        # The +1 pseudocount avoids log(0) for zero-expression values.
        # Column 0 is "Gleason Group" (target); all other columns are gene features.
        # We rebuild the DataFrame from numpy arrays to bypass pandas 3.x strict
        # dtype enforcement that rejects assigning floats into raw CSV object columns.
        _gleason_values = df.iloc[:, 0].values
        _gene_matrix = np.log2(df.iloc[:, 1:].astype(float).values + 1)
        df = pd.DataFrame(
            np.column_stack([_gleason_values, _gene_matrix]),
            index=df.index,
            columns=df.columns,
        )
        df["Gleason Group"] = df["Gleason Group"].astype(int)

    logger.info("Dataset loaded – shape: %s", df.shape)
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Persist a transposed DataFrame to CSV.

    Args:
        df: The processed :class:`pandas.DataFrame` to save.
        path: Destination file path.
    """
    logger.info("Saving processed dataset to '%s'", path)
    df.T.to_csv(path, index=False)
    logger.info("Dataset saved successfully.")


def save_model_results(results: dict[str, list[dict]], results_dir: str) -> None:
    """Persist each model's grid-search results to its own CSV file.

    The ``scores`` sub-dict present in every result record is flattened into
    top-level columns so that the CSV is immediately usable in spreadsheet
    tools or for further analysis.

    Args:
        results: Mapping of model name (e.g. ``"decision_tree"``) to the list
            of result records returned by the corresponding grid-search
            function.
        results_dir: Directory path where the CSV files will be written.  The
            directory is created automatically if it does not exist.
    """
    os.makedirs(results_dir, exist_ok=True)

    for model_name, records in results.items():
        # Flatten each record: hoist the nested ``scores`` dict to top level.
        flat_records = []
        for record in records:
            flat = {k: v for k, v in record.items() if k != "scores"}
            flat.update(record.get("scores", {}))
            flat_records.append(flat)

        output_path = os.path.join(results_dir, f"{model_name}.csv")
        pd.DataFrame(flat_records).to_csv(output_path, index=False)
        logger.info("Saved %d rows to '%s'", len(flat_records), output_path)


def save_reduction(
    label: str,
    train_predictors: "np.ndarray",
    test_predictors: "np.ndarray",
    train_target: "np.ndarray",
    test_target: "np.ndarray",
    reductions_dir: str,
) -> None:
    """Persist the train and test arrays of one dimension-reduction step to CSV.

    Two files are written per call::

        {reductions_dir}/{label}_train.csv
        {reductions_dir}/{label}_test.csv

    Each file includes the reduced feature columns (``feature_0``,
    ``feature_1``, …) followed by a ``target`` column containing the
    integer-encoded Gleason Group labels, making the files self-contained
    for downstream analysis.

    Args:
        label: Short identifier used in the file names (e.g. ``"lasso"``).
        train_predictors: Reduced training feature matrix.
        test_predictors: Reduced test feature matrix.
        train_target: Training labels (1-D array).
        test_target: Test labels (1-D array).
        reductions_dir: Output directory; created automatically if absent.
    """
    import numpy as np  # local import keeps top-level imports clean

    os.makedirs(reductions_dir, exist_ok=True)

    # Build uniform column names regardless of reduction method.
    n_features = train_predictors.shape[1]
    feature_cols = [f"feature_{i}" for i in range(n_features)]

    for split_name, X, y in (
        ("train", train_predictors, train_target),
        ("test",  test_predictors,  test_target),
    ):
        df = pd.DataFrame(X, columns=feature_cols)
        df["target"] = y
        output_path = os.path.join(reductions_dir, f"{label}_{split_name}.csv")
        df.to_csv(output_path, index=False)
        logger.info(
            "Saved %s reduction (%s): %d samples × %d features → '%s'",
            label, split_name, X.shape[0], n_features, output_path,
        )
