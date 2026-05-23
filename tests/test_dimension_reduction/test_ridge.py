"""
Unit tests for the Ridge feature-selection domain service.

Tests mirror the Lasso test structure: synthetic data, shape assertions, and
a column-subset check confirming that Ridge selects real input columns.
"""

import numpy as np
import pytest

from src.domain.services.dimension_reduction.ridge import reduce_features_ridge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pre-split (x_train, y_train, x_test, y_test) arrays."""
    rng = np.random.default_rng(seed=1)
    X = rng.standard_normal((40, 50))
    y = rng.integers(0, 6, size=40).astype(float)
    x_train, x_test = X[:30], X[30:]
    y_train, y_test = y[:30], y[30:]
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReduceFeaturesRidge:
    """Test suite for ``reduce_features_ridge``."""

    def test_output_shape_matches_requested_features(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Reduced train and test matrices must have exactly ``nr_of_features`` columns."""
        x_train, y_train, x_test, _ = synthetic_data
        train_r, test_r = reduce_features_ridge(
            x_train, y_train, x_test, nr_of_features=10, alpha=1.0
        )
        assert train_r.shape == (x_train.shape[0], 10)
        assert test_r.shape == (x_test.shape[0], 10)

    def test_output_rows_match_input_rows(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Row counts must be preserved for both splits after reduction."""
        x_train, y_train, x_test, _ = synthetic_data
        train_r, test_r = reduce_features_ridge(
            x_train, y_train, x_test, nr_of_features=20, alpha=0.5
        )
        assert train_r.shape[0] == x_train.shape[0]
        assert test_r.shape[0] == x_test.shape[0]

    def test_single_feature_selection(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Selecting one feature must return matrices with 1 column each."""
        x_train, y_train, x_test, _ = synthetic_data
        train_r, test_r = reduce_features_ridge(
            x_train, y_train, x_test, nr_of_features=1, alpha=1.0
        )
        assert train_r.shape[1] == 1
        assert test_r.shape[1] == 1

    def test_selected_columns_are_subset_of_input(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Every returned column of the train result must appear verbatim in x_train."""
        x_train, y_train, x_test, _ = synthetic_data
        train_r, _ = reduce_features_ridge(
            x_train, y_train, x_test, nr_of_features=5, alpha=1.0
        )

        for col_idx in range(train_r.shape[1]):
            found = any(
                np.allclose(train_r[:, col_idx], x_train[:, j])
                for j in range(x_train.shape[1])
            )
            assert found, f"Column {col_idx} not found in x_train."

    @pytest.mark.parametrize("alpha", [0.1, 1.0, 5.0])
    def test_various_alpha_values_produce_correct_shape(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        alpha: float,
    ) -> None:
        """Shape must be correct for any valid alpha."""
        x_train, y_train, x_test, _ = synthetic_data
        train_r, test_r = reduce_features_ridge(
            x_train, y_train, x_test, nr_of_features=15, alpha=alpha
        )
        assert train_r.shape == (x_train.shape[0], 15)
        assert test_r.shape == (x_test.shape[0], 15)
