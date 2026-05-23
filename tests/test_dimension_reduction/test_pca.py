"""
Unit tests for the PCA dimensionality-reduction domain service.

PCA *extracts* new axes rather than selecting original columns, so tests
focus on shape and variance-retention contracts rather than column-subset
identity.
"""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from src.domain.services.dimension_reduction.pca import reduce_features_pca


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Return pre-split (x_train, x_test) predictor matrices.

    PCA requires no target, so only the feature matrices are needed.
    """
    rng = np.random.default_rng(seed=2)
    X = rng.standard_normal((50, 60))
    return X[:40], X[40:]  # 40 train, 10 test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReduceFeaturesPca:
    """Test suite for ``reduce_features_pca``."""

    def test_output_has_fewer_columns_than_input(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PCA at 95 % variance should not grow the feature space."""
        x_train, x_test = synthetic_data
        train_r, test_r = reduce_features_pca(x_train, x_test, variance_threshold=0.95)
        assert train_r.shape[1] <= x_train.shape[1]
        assert test_r.shape[1] == train_r.shape[1]  # same projection applied to test

    def test_output_row_count_preserved(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Row counts must be preserved for both splits after PCA."""
        x_train, x_test = synthetic_data
        train_r, test_r = reduce_features_pca(x_train, x_test, variance_threshold=0.95)
        assert train_r.shape[0] == x_train.shape[0]
        assert test_r.shape[0] == x_test.shape[0]

    def test_explained_variance_meets_threshold(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Cumulative explained variance of the returned components must be ≥ threshold."""
        x_train, x_test = synthetic_data
        threshold = 0.90
        train_r, _ = reduce_features_pca(x_train, x_test, variance_threshold=threshold)

        # Independently verify using the TRAINING data only (matching what the
        # service does internally).
        pca = PCA(n_components=train_r.shape[1])
        pca.fit(x_train)
        cumulative_variance = pca.explained_variance_ratio_.sum()

        assert cumulative_variance >= threshold, (
            f"Explained variance {cumulative_variance:.3f} < threshold {threshold}"
        )

    @pytest.mark.parametrize("threshold", [0.80, 0.90, 0.99])
    def test_higher_threshold_retains_more_components(
        self, synthetic_data: tuple[np.ndarray, np.ndarray], threshold: float
    ) -> None:
        """A higher variance threshold must never return fewer components."""
        x_train, x_test = synthetic_data
        result_low, _ = reduce_features_pca(x_train, x_test, variance_threshold=0.80)
        result_high, _ = reduce_features_pca(x_train, x_test, variance_threshold=threshold)
        assert result_high.shape[1] >= result_low.shape[1]

    def test_output_is_float_array(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PCA output must be floating-point numpy arrays."""
        x_train, x_test = synthetic_data
        train_r, test_r = reduce_features_pca(x_train, x_test, variance_threshold=0.95)
        assert np.issubdtype(train_r.dtype, np.floating)
        assert np.issubdtype(test_r.dtype, np.floating)
