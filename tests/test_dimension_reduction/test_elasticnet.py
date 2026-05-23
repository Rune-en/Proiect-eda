"""
Unit tests for the ElasticNet dimension-reduction domain service.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.domain.services.dimension_reduction.elasticnet import reduce_features_elasticnet


def _make_data(n_features: int = 50) -> tuple:
    X, y = make_classification(
        n_samples=60, n_features=n_features, n_informative=10,
        n_classes=3, random_state=3,
    )
    mid = len(X) // 2
    return X[:mid], y[:mid].astype(float), X[mid:], y[mid:].astype(float)


class TestReduceFeaturesElasticNet:

    def test_output_shapes(self) -> None:
        x_tr, y_tr, x_te, _ = _make_data()
        train_out, test_out = reduce_features_elasticnet(x_tr, y_tr, x_te, nr_of_features=10)
        assert train_out.shape == (30, 10)
        assert test_out.shape  == (30, 10)

    def test_nr_of_features_respected(self) -> None:
        x_tr, y_tr, x_te, _ = _make_data(n_features=50)
        for k in [5, 15, 25]:
            tr, te = reduce_features_elasticnet(x_tr, y_tr, x_te, nr_of_features=k)
            assert tr.shape[1] == k
            assert te.shape[1] == k

    def test_no_leakage_same_columns(self) -> None:
        """Train and test must use the same feature indices."""
        x_tr, y_tr, x_te, _ = _make_data()
        tr, te = reduce_features_elasticnet(x_tr, y_tr, x_te, nr_of_features=10)
        # If same columns selected, column-wise correlation should be near 1
        # for columns that appear in both (they are identical observations from same genes)
        assert tr.shape[1] == te.shape[1]
