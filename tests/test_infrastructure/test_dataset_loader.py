"""
Unit tests for the dataset-loader infrastructure module.

The tests use ``tmp_path`` (pytest's built-in temporary directory fixture)
to create small synthetic CSV files so no real ``DB.csv`` is required.

Raw CSV structure (mirrors DB.csv):
    - Columns: gene_name, gene_id, gene_type, <sample_1>, ..., <sample_N>
    - Row 0:   gene_name="Gleason Group" – sample values are clinical label strings.
    - Rows 1–28: other clinical/metadata features (dropped after transposition).
    - Rows 29+:  numeric gene-expression features.

After ``load_dataset`` runs:
    - Rows  → samples (index named ``sample_id``)
    - Col 0 → "Gleason Group" (integer-encoded target)
    - Cols 29+ become feature columns; cols 1–28 are discarded.
"""

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.dataset_loader import load_dataset, save_dataset


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _build_minimal_csv(path, n_samples: int = 4) -> None:
    """Write the smallest valid CSV that ``load_dataset`` can handle.

    The function creates a gene × sample CSV with:
        - gene_name / gene_id / gene_type metadata columns.
        - Row 0: "Gleason Group" with sample label values.
        - Rows 1–28: placeholder metadata rows (28 rows, dropped post-transpose).
        - Row 29: a single numeric gene-expression feature row.

    Args:
        path: ``pathlib.Path`` target location.
        n_samples: Number of sample columns to generate (≥ 2).
    """
    gleason_labels = ["Normal", "Group 1", "Group 2", "Group 3", "Group 4", "Group 5"]
    sample_ids = [f"TCGA-{i:03d}" for i in range(n_samples)]

    rows = []

    # Row 0: Gleason Group target labels.
    row0 = {"gene_name": "Gleason Group", "gene_id": "", "gene_type": ""}
    for i, sid in enumerate(sample_ids):
        row0[sid] = gleason_labels[i % len(gleason_labels)]
    rows.append(row0)

    # Rows 1–28: placeholder metadata (will be dropped by load_dataset).
    for meta_idx in range(1, 29):
        row = {"gene_name": f"metadata_{meta_idx}", "gene_id": "", "gene_type": ""}
        for sid in sample_ids:
            row[sid] = 0.0
        rows.append(row)

    # Row 29: a single gene-expression feature.
    row29 = {"gene_name": "ENSG00000001", "gene_id": "ENSG00000001", "gene_type": "protein_coding"}
    for i, sid in enumerate(sample_ids):
        row29[sid] = float(i + 1)
    rows.append(row29)

    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadDataset:
    """Test suite for ``load_dataset``."""

    def test_returns_dataframe(self, tmp_path) -> None:
        """``load_dataset`` must return a pandas DataFrame."""
        csv_path = tmp_path / "test.csv"
        _build_minimal_csv(csv_path, n_samples=4)

        result = load_dataset(str(csv_path))

        assert isinstance(result, pd.DataFrame)

    def test_index_is_named_sample_id(self, tmp_path) -> None:
        """The DataFrame index must be named ``sample_id``."""
        csv_path = tmp_path / "test.csv"
        _build_minimal_csv(csv_path, n_samples=4)

        result = load_dataset(str(csv_path))

        assert result.index.name == "sample_id"

    def test_gleason_group_is_numeric(self, tmp_path) -> None:
        """The ``Gleason Group`` column must contain integer values after loading."""
        csv_path = tmp_path / "test.csv"
        _build_minimal_csv(csv_path, n_samples=4)

        result = load_dataset(str(csv_path))

        assert pd.api.types.is_numeric_dtype(result["Gleason Group"])

    def test_gleason_group_values_in_expected_range(self, tmp_path) -> None:
        """Encoded Gleason Group values must be integers in [0, 5]."""
        csv_path = tmp_path / "test.csv"
        _build_minimal_csv(csv_path, n_samples=6)  # 6 samples → all 6 labels

        result = load_dataset(str(csv_path))
        values = result["Gleason Group"].unique()

        assert all(0 <= v <= 5 for v in values)

    def test_missing_file_raises_file_not_found(self) -> None:
        """Loading a non-existent path must raise ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/to/DB.csv")


class TestSaveDataset:
    """Test suite for ``save_dataset``."""

    def test_creates_csv_file(self, tmp_path) -> None:
        """``save_dataset`` must create the target CSV file."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out_path = tmp_path / "output.csv"

        save_dataset(df, str(out_path))

        assert out_path.exists()

    def test_saved_file_is_readable(self, tmp_path) -> None:
        """The CSV written by ``save_dataset`` must be readable by pandas."""
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        out_path = tmp_path / "output.csv"

        save_dataset(df, str(out_path))

        loaded = pd.read_csv(out_path)
        assert not loaded.empty
