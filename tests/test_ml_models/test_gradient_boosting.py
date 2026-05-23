"""
Unit tests for the Gradient Boosting grid-search domain service.

Tests verify record counts, required keys, hyper-parameter values, and basic
metric sanity checks – all using tiny in-memory synthetic data.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.domain.services.ml_models.gradient_boosting import (
    GRADIENT_BOOSTING_LEARNING_RATES,
    GRADIENT_BOOSTING_MAX_DEPTHS,
    GRADIENT_BOOSTING_N_ESTIMATORS,
    grid_search_gradient_boosting,
)


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
        random_state=2,
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

class TestGridSearchGradientBoosting:
    """Test suite for ``grid_search_gradient_boosting``."""

    def test_result_count_equals_full_grid(self) -> None:
        """Record count = n_splits × n_estimators × n_depths × n_learning_rates."""
        expected = (
            1
            * len(GRADIENT_BOOSTING_N_ESTIMATORS)
            * len(GRADIENT_BOOSTING_MAX_DEPTHS)
            * len(GRADIENT_BOOSTING_LEARNING_RATES)
        )
        results = grid_search_gradient_boosting([_make_split()])
        assert len(results) == expected

    def test_each_record_has_required_keys(self) -> None:
        """All mandatory keys must be present in every result record."""
        required_keys = {
            "dimension_reduction_type",
            "model",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "scores",
        }
        results = grid_search_gradient_boosting([_make_split()])
        for record in results:
            assert required_keys.issubset(record.keys())

    def test_n_estimators_values_match_grid(self) -> None:
        """n_estimators values in results must all come from GRADIENT_BOOSTING_N_ESTIMATORS."""
        results = grid_search_gradient_boosting([_make_split()])
        for record in results:
            assert record["n_estimators"] in GRADIENT_BOOSTING_N_ESTIMATORS

    def test_max_depth_values_match_grid(self) -> None:
        """max_depth values in results must all come from GRADIENT_BOOSTING_MAX_DEPTHS."""
        results = grid_search_gradient_boosting([_make_split()])
        for record in results:
            assert record["max_depth"] in GRADIENT_BOOSTING_MAX_DEPTHS

    def test_learning_rate_values_match_grid(self) -> None:
        """learning_rate values must all come from GRADIENT_BOOSTING_LEARNING_RATES."""
        results = grid_search_gradient_boosting([_make_split()])
        for record in results:
            assert record["learning_rate"] in GRADIENT_BOOSTING_LEARNING_RATES

    def test_model_label(self) -> None:
        """``model`` field must equal 'Gradient Boosting' for all records."""
        results = grid_search_gradient_boosting([_make_split()])
        assert all(r["model"] == "Gradient Boosting" for r in results)

    def test_within_1_acc_in_valid_range(self) -> None:
        """Within-±1-group tolerance accuracy must lie within [0, 1]."""
        results = grid_search_gradient_boosting([_make_split()])
        for record in results:
            acc = record["scores"]["within_1_acc"]
            assert 0.0 <= acc <= 1.0, f"within_1_acc out of range: {acc}"

    def test_rmse_is_non_negative(self) -> None:
        """RMSE must be non-negative for all records."""
        results = grid_search_gradient_boosting([_make_split()])
        for record in results:
            assert record["scores"]["rmse"] >= 0.0

    def test_scores_sub_dict_has_all_metrics(self) -> None:
        """All regression metrics must be present in every ``scores`` dict."""
        required_metrics = {
            "rmse",
            "mae",
            "r2",
            "spearman_r",
            "within_1_acc",
        }
        results = grid_search_gradient_boosting([_make_split()])
        for record in results:
            assert required_metrics.issubset(record["scores"].keys())
