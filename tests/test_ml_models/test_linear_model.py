"""
Unit tests for the Linear Regression grid-search domain service.

Linear Regression has no hyper-parameters, so the expected result count
equals simply len(data_list) – one record per split.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.domain.services.ml_models.linear_model import grid_search_linear_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_split(name: str = "test") -> dict:
    """Return a minimal dataset split dict."""
    X, y = make_classification(
        n_samples=60,
        n_features=10,
        n_classes=3,
        n_informative=5,
        random_state=1,
    )
    y = y.astype(float)
    mid = len(X) // 2
    return {
        "name": name,
        "train_predictors": X[:mid],
        "train_target": y[:mid],
        "test_predictors": X[mid:],
        "test_target": y[mid:],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGridSearchLinearModel:
    """Test suite for ``grid_search_linear_model``."""

    def test_returns_one_record_per_split(self) -> None:
        """Exactly one record per split (no hyper-parameter sweep)."""
        n_splits = 3
        results = grid_search_linear_model([_make_split(f"s{i}") for i in range(n_splits)])
        assert len(results) == n_splits

    def test_each_record_has_required_keys(self) -> None:
        """All mandatory keys must be present in every result record."""
        required_keys = {"dimension_reduction_type", "model", "scores"}
        results = grid_search_linear_model([_make_split()])
        for record in results:
            assert required_keys.issubset(record.keys())

    def test_scores_sub_dict_has_all_metrics(self) -> None:
        """All regression metrics must appear in the ``scores`` dict."""
        required_metrics = {
            "rmse",
            "mae",
            "r2",
            "spearman_r",
            "within_1_acc",
        }
        results = grid_search_linear_model([_make_split()])
        for record in results:
            assert required_metrics.issubset(record["scores"].keys())

    def test_model_label(self) -> None:
        """``model`` field must equal 'Linear Regression'."""
        results = grid_search_linear_model([_make_split()])
        assert all(r["model"] == "Linear Regression" for r in results)

    def test_rmse_is_non_negative(self) -> None:
        """RMSE must be ≥ 0."""
        results = grid_search_linear_model([_make_split()])
        for record in results:
            assert record["scores"]["rmse"] >= 0.0
