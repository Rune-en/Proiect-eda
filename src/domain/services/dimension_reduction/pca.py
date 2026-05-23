"""
Domain service – PCA-based dimensionality reduction.

DDD: pure domain service – no I/O, no side effects.  PCA projects the
high-dimensional gene-expression space onto a lower-dimensional subspace
that retains a specified fraction of the total variance.

Unlike Lasso/Ridge (feature *selection*), PCA performs feature *extraction*:
the returned columns are principal components, not original gene columns.
"""

import logging

import numpy as np
from sklearn.decomposition import PCA

# Module-level logger – follows 12-factor Factor XI (logs as event streams).
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

# Fraction of total variance that the selected principal components must
# explain.  Used as the default when no explicit value is passed.
DEFAULT_VARIANCE_THRESHOLD: float = 0.95


# ---------------------------------------------------------------------------
# Domain service function
# ---------------------------------------------------------------------------

def reduce_features_pca(
    train_predictors: np.ndarray,
    test_predictors: np.ndarray,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Project gene-expression data onto the leading principal components.

    The PCA model is fitted **on training data only** to avoid data leakage.
    The covariance structure (and therefore the principal components) is
    derived solely from training samples; the same fitted projection is then
    applied to *test_predictors*.

    Args:
        train_predictors: Training feature matrix of shape
            ``(n_train, n_features)``.
        test_predictors: Test feature matrix of shape ``(n_test, n_features)``.
            Must have the same number of columns as *train_predictors*.
        variance_threshold: Minimum cumulative explained-variance ratio
            (between 0 and 1).  E.g. ``0.95`` retains enough components to
            explain 95 % of training-set variance.

    Returns:
        A tuple ``(train_reduced, test_reduced)`` in the PCA space.
    """
    # Instantiate PCA with an automatic component count based on variance.
    # ``svd_solver="full"`` supports fractional n_components.
    pca_model = PCA(n_components=variance_threshold)

    # Fit on training data only; no target is needed for PCA.
    train_reduced = pca_model.fit_transform(train_predictors)

    # Apply the same projection to test data without refitting.
    test_reduced = pca_model.transform(test_predictors)

    n_total      = train_predictors.shape[1]
    n_components = pca_model.n_components_
    n_dropped    = n_total - n_components
    var_retained = float(pca_model.explained_variance_ratio_.sum())
    var_dropped  = 1.0 - var_retained

    logger.info(
        "[PCA] Summary\n"
        "  Method          : PCA dimensionality reduction\n"
        "  Variance thresh : %.4g  (retain components until cumvar ≥ threshold)\n"
        "  ── Input ────────────────────────────────────────────────────\n"
        "  Train input     : %d samples × %d features\n"
        "  Test  input     : %d samples × %d features\n"
        "  ── Reduction ────────────────────────────────────────────────\n"
        "  Components kept : %d  (covering %.4f = %.2f%% of train variance)\n"
        "  Dropped         : %d components (discarded %.4f = %.2f%% of variance)\n"
        "  ── Output ───────────────────────────────────────────────────\n"
        "  Train output    : %d samples × %d components\n"
        "  Test  output    : %d samples × %d components",
        variance_threshold,
        train_predictors.shape[0], n_total,
        test_predictors.shape[0],  n_total,
        n_components, var_retained, var_retained * 100,
        n_dropped,    var_dropped,  var_dropped  * 100,
        train_reduced.shape[0], n_components,
        test_predictors.shape[0], n_components,
    )

    return train_reduced, test_reduced
