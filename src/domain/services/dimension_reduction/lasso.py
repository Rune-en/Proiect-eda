"""
Domain service – Lasso-based feature selection.

DDD: a *domain service* encapsulates a specific piece of business / ML logic
that does not naturally belong to a single entity.  This service selects the
most informative gene features using Lasso regularisation, which shrinks
unimportant coefficients to zero.

12-factor: the hyper-parameter grid is stored here as a plain dict rather
than hidden in a database so it is versioned alongside the code (Factor V –
Build/release/run).
"""

import logging

import numpy as np
from sklearn.linear_model import Lasso

# Module-level logger – follows 12-factor Factor XI (logs as event streams).
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameter search space
# ---------------------------------------------------------------------------

# Number of top-ranked features to retain in each experiment.
LASSO_NR_OF_FEATURES: list[int] = [10, 20, 30, 40, 50]

# Regularisation strengths to sweep over; higher values produce sparser models.
LASSO_ALPHAS: list[float] = [0.1, 0.5, 1.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Domain service function
# ---------------------------------------------------------------------------

def reduce_features_lasso(
    train_predictors: np.ndarray,
    train_target: np.ndarray,
    test_predictors: np.ndarray,
    nr_of_features: int = 20,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the top features using Lasso regression coefficients.

    The model is fitted **exclusively on the training data** to avoid data
    leakage: the same feature indices (determined from train only) are then
    applied to *test_predictors*, so no information from the test set
    influences feature selection.

    Args:
        train_predictors: Training feature matrix of shape
            ``(n_train, n_features)``.
        train_target: Training labels of shape ``(n_train,)`` – integer-encoded
            Gleason Group values.
        test_predictors: Test feature matrix of shape ``(n_test, n_features)``.
            Must have the same number of columns as *train_predictors*.
        nr_of_features: How many features to keep.  Must be ≤
            ``train_predictors.shape[1]``.
        alpha: Lasso regularisation strength.  Larger values → sparser model.

    Returns:
        A tuple ``(train_reduced, test_reduced)`` where each array contains
        only the selected feature columns applied to the respective split.
    """
    # Fit a Lasso model on training data ONLY to learn feature importances.
    model = Lasso(alpha=alpha, max_iter=10_000)
    model.fit(train_predictors, train_target)

    # Use the absolute value of coefficients as a proxy for feature importance.
    importance = np.abs(model.coef_)

    # Retrieve the indices of the top `nr_of_features` most important features.
    selected_indices = np.argsort(importance)[-nr_of_features:]

    n_total   = train_predictors.shape[1]
    n_dropped = n_total - nr_of_features
    zeroed    = int(np.sum(model.coef_ == 0.0))

    logger.info(
        "[Lasso] Summary\n"
        "  Method          : Lasso feature selection (L1 regularisation)\n"
        "  Alpha           : %.4g\n"
        "  ── Input ────────────────────────────────────────────────────\n"
        "  Train input     : %d samples × %d features\n"
        "  Test  input     : %d samples × %d features\n"
        "  ── Selection ────────────────────────────────────────────────\n"
        "  Zeroed coefs    : %d features (coefficient == 0 after L1)\n"
        "  Requested top-k : %d features\n"
        "  Dropped         : %d features (lowest |coefficient| rank)\n"
        "  ── Output ───────────────────────────────────────────────────\n"
        "  Train output    : %d samples × %d features\n"
        "  Test  output    : %d samples × %d features",
        alpha,
        train_predictors.shape[0], n_total,
        test_predictors.shape[0],  n_total,
        zeroed,
        nr_of_features,
        n_dropped,
        train_predictors.shape[0], nr_of_features,
        test_predictors.shape[0],  nr_of_features,
    )

    # Apply the same indices to both splits; test data is never used in fitting.
    return train_predictors[:, selected_indices], test_predictors[:, selected_indices]
