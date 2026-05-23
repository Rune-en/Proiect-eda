"""
Unit tests for the Poisson Regression grid-search domain service.

Poisson regression requires non-negative target values.  The synthetic target
here uses integers 0–5 (matching the Gleason Group domain).
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression

from src.domain.services.ml_models.poisson_model import (
    POISSON_ALPHAS,
    grid_search_poisson_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_split(name: str = "test") -> dict:
    """Return a minimal split dict with non-negative integer targets."""
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((60, 10))
    # Non-negative integer targets in [0, 5] – like Gleason Groups.
    y = rng.integers(0, 6, size=60).astype(float)
    mid = 30
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

class TestGridSearchPoissonModel:
    """Test suite for ``grid_search_poisson_model``."""

    def test_result_count_equals_splits_times_alphas(self) -> None:
        """Record count must equal n_splits × len(POISSON_ALPHAS)."""
        n_splits = 2
        results = grid_search_poisson_model(
            [_make_split(f"s{i}") for i in range(n_splits)]
        )
        assert len(results) == n_splits * len(POISSON_ALPHAS)

    def test_each_record_has_required_keys(self) -> None:
        """All mandatory keys must appear in every record."""
        required_keys = {"dimension_reduction_type", "model", "alpha", "scores"}
        results = grid_search_poisson_model([_make_split()])
        for record in results:
            assert required_keys.issubset(record.keys())

    def test_alpha_values_match_grid(self) -> None:
        """The ``alpha`` values in results must come from POISSON_ALPHAS."""
        results = grid_search_poisson_model([_make_split()])
        returned_alphas = sorted({r["alpha"] for r in results})
        assert returned_alphas == sorted(POISSON_ALPHAS)

    def test_model_label(self) -> None:
        """``model`` field must equal 'Poisson Regression'."""
        results = grid_search_poisson_model([_make_split()])
        assert all(r["model"] == "Poisson Regression" for r in results)

    def test_rmse_is_non_negative(self) -> None:
        """RMSE must always be ≥ 0."""
        results = grid_search_poisson_model([_make_split()])
        for record in results:
            assert record["scores"]["rmse"] >= 0.0

    def test_scores_sub_dict_has_all_metrics(self) -> None:
        """All regression metrics must appear in every ``scores`` dict."""
        required_metrics = {
            "rmse",
            "mae",
            "r2",
            "spearman_r",
            "within_1_acc",
        }
        results = grid_search_poisson_model([_make_split()])
        for record in results:
            assert required_metrics.issubset(record["scores"].keys())
