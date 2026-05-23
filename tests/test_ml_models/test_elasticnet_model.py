"""
Unit tests for the ElasticNetCV regressor domain service.
"""

import numpy as np
from sklearn.datasets import make_classification

from src.domain.services.ml_models.elasticnet_model import grid_search_elasticnet_model


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


class TestGridSearchElasticNetModel:

    def test_one_record_per_split(self) -> None:
        assert len(grid_search_elasticnet_model([_make_split()])) == 1
        assert len(grid_search_elasticnet_model([_make_split(), _make_split("b")])) == 2

    def test_each_record_has_required_keys(self) -> None:
        required = {"dimension_reduction_type", "model", "alpha", "l1_ratio", "scores"}
        for record in grid_search_elasticnet_model([_make_split()]):
            assert required.issubset(record.keys())

    def test_model_label(self) -> None:
        assert all(r["model"] == "ElasticNetCV" for r in grid_search_elasticnet_model([_make_split()]))

    def test_alpha_is_positive(self) -> None:
        for record in grid_search_elasticnet_model([_make_split()]):
            assert record["alpha"] > 0.0

    def test_l1_ratio_in_valid_range(self) -> None:
        for record in grid_search_elasticnet_model([_make_split()]):
            assert 0.0 <= record["l1_ratio"] <= 1.0

    def test_within_1_acc_in_valid_range(self) -> None:
        for record in grid_search_elasticnet_model([_make_split()]):
            assert 0.0 <= record["scores"]["within_1_acc"] <= 1.0

    def test_rmse_is_non_negative(self) -> None:
        for record in grid_search_elasticnet_model([_make_split()]):
            assert record["scores"]["rmse"] >= 0.0

    def test_scores_sub_dict_has_all_metrics(self) -> None:
        required = {"rmse", "mae", "r2", "spearman_r", "within_1_acc"}
        for record in grid_search_elasticnet_model([_make_split()]):
            assert required.issubset(record["scores"].keys())
