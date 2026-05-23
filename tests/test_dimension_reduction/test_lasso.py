"""
Unit tests for the Lasso feature-selection domain service.

All tests use small synthetic numpy arrays so no CSV or sklearn model
persistence is required.  The tests verify:

    - Output shape matches the requested number of features.
    - The returned columns are a genuine subset of the input columns.
    - Edge cases: nr_of_features == 1, and nr_of_features == n_total_features.
"""

import numpy as np
import pytest

from src.domain.services.dimension_reduction.lasso import reduce_features_lasso


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pre-split (x_train, y_train, x_test, y_test) arrays.

    Keeping train and test separate mirrors the leakage-free API where the
    service is fitted on train only.
    """
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal((40, 50))
    y = rng.integers(0, 6, size=40).astype(float)
    # Fixed 75/25 split for reproducibility.
    x_train, x_test = X[:30], X[30:]
    y_train, y_test = y[:30], y[30:]
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReduceFeaturesLasso:
    """Test suite for ``reduce_features_lasso``."""

    def test_output_shape_matches_requested_features(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Reduced train and test matrices must have exactly ``nr_of_features`` columns."""
        x_train, y_train, x_test, _ = synthetic_data
        nr_of_features = 10

        train_r, test_r = reduce_features_lasso(
            x_train, y_train, x_test, nr_of_features=nr_of_features, alpha=1.0
        )

        assert train_r.shape == (x_train.shape[0], nr_of_features)
        assert test_r.shape == (x_test.shape[0], nr_of_features)

    def test_output_rows_match_input_rows(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Row counts must be preserved for both splits after reduction."""
        x_train, y_train, x_test, _ = synthetic_data
        train_r, test_r = reduce_features_lasso(
            x_train, y_train, x_test, nr_of_features=20, alpha=0.5
        )
        assert train_r.shape[0] == x_train.shape[0]
        assert test_r.shape[0] == x_test.shape[0]

    def test_single_feature_selection(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Selecting a single feature must return matrices with 1 column each."""
        x_train, y_train, x_test, _ = synthetic_data
        train_r, test_r = reduce_features_lasso(
            x_train, y_train, x_test, nr_of_features=1, alpha=1.0
        )
        assert train_r.shape[1] == 1
        assert test_r.shape[1] == 1

    def test_all_features_returns_same_column_count(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Requesting all features must keep every column."""
        x_train, y_train, x_test, _ = synthetic_data
        n_features = x_train.shape[1]
        train_r, test_r = reduce_features_lasso(
            x_train, y_train, x_test, nr_of_features=n_features, alpha=0.1
        )
        assert train_r.shape[1] == n_features
        assert test_r.shape[1] == n_features

    def test_selected_columns_are_subset_of_input(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Every column in the train result must appear verbatim in x_train."""
        x_train, y_train, x_test, _ = synthetic_data
        train_r, _ = reduce_features_lasso(
            x_train, y_train, x_test, nr_of_features=5, alpha=1.0
        )

        # Each column of the reduced train matrix must come from x_train.
        for col_idx in range(train_r.shape[1]):
            found = any(
                np.allclose(train_r[:, col_idx], x_train[:, j])
                for j in range(x_train.shape[1])
            )
            assert found, f"Column {col_idx} of train result not found in x_train."

    @pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
    def test_various_alpha_values_produce_correct_shape(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        alpha: float,
    ) -> None:
        """Shape must be correct regardless of the alpha value used."""
        x_train, y_train, x_test, _ = synthetic_data
        train_r, test_r = reduce_features_lasso(
            x_train, y_train, x_test, nr_of_features=15, alpha=alpha
        )
        assert train_r.shape == (x_train.shape[0], 15)
        assert test_r.shape == (x_test.shape[0], 15)
