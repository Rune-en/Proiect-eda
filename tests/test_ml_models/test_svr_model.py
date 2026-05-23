"""
Unit tests for the SVR (RBF kernel) grid-search domain service.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.domain.services.ml_models.svr_model import (
    SVR_C_VALUES,
    SVR_GAMMA_VALUES,
    grid_search_svr,
)


def _make_split(name: str = "test") -> dict:
    X, y = make_classification(
        n_samples=60, n_features=10, n_classes=3, n_informative=5, random_state=2,
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


class TestGridSearchSVR:

    def test_result_count_equals_full_grid(self) -> None:
        expected = 1 * len(SVR_C_VALUES) * len(SVR_GAMMA_VALUES)
        assert len(grid_search_svr([_make_split()])) == expected

    def test_each_record_has_required_keys(self) -> None:
        required = {"dimension_reduction_type", "model", "C", "gamma", "scores"}
        for record in grid_search_svr([_make_split()]):
            assert required.issubset(record.keys())

    def test_c_values_match_grid(self) -> None:
        for record in grid_search_svr([_make_split()]):
            assert record["C"] in SVR_C_VALUES

    def test_gamma_values_match_grid(self) -> None:
        for record in grid_search_svr([_make_split()]):
            assert record["gamma"] in SVR_GAMMA_VALUES

    def test_model_label(self) -> None:
        assert all(r["model"] == "SVR (rbf)" for r in grid_search_svr([_make_split()]))

    def test_within_1_acc_in_valid_range(self) -> None:
        for record in grid_search_svr([_make_split()]):
            assert 0.0 <= record["scores"]["within_1_acc"] <= 1.0

    def test_rmse_is_non_negative(self) -> None:
        for record in grid_search_svr([_make_split()]):
            assert record["scores"]["rmse"] >= 0.0

    def test_scores_sub_dict_has_all_metrics(self) -> None:
        required = {"rmse", "mae", "r2", "spearman_r", "within_1_acc"}
        for record in grid_search_svr([_make_split()]):
            assert required.issubset(record["scores"].keys())
