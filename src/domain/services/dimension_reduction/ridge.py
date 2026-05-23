"""
Domain service – Ridge-based feature selection.

DDD: same pattern as the Lasso service – pure ML logic, no I/O.  Ridge
(L2 regularisation) never zeroes out coefficients but shrinks them uniformly,
so the top-k selection by absolute coefficient magnitude is used instead.

12-factor: hyper-parameter grid versioned alongside the code.
"""

import logging

import numpy as np
from sklearn.linear_model import Ridge

# Module-level logger – follows 12-factor Factor XI (logs as event streams).
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameter search space
# ---------------------------------------------------------------------------

# Number of top-ranked features to retain in each experiment.
RIDGE_NR_OF_FEATURES: list[int] = [10, 20, 30, 40, 50]

# Regularisation strengths to sweep over.
RIDGE_ALPHAS: list[float] = [0.1, 0.5, 1.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Domain service function
# ---------------------------------------------------------------------------

def reduce_features_ridge(
    train_predictors: np.ndarray,
    train_target: np.ndarray,
    test_predictors: np.ndarray,
    nr_of_features: int = 20,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the top features using Ridge regression coefficients.

    The model is fitted **exclusively on the training data** to avoid data
    leakage: the same feature indices (determined from train only) are then
    applied to *test_predictors*, so no information from the test set
    influences feature selection.

    Args:
        train_predictors: Training feature matrix of shape
            ``(n_train, n_features)``.
        train_target: Training labels of shape ``(n_train,)``.
        test_predictors: Test feature matrix of shape ``(n_test, n_features)``.
            Must have the same number of columns as *train_predictors*.
        nr_of_features: Number of features to retain.
        alpha: Ridge regularisation strength.

    Returns:
        A tuple ``(train_reduced, test_reduced)`` where each array contains
        only the selected feature columns applied to the respective split.
    """
    # Fit a Ridge model on training data ONLY to score features.
    model = Ridge(alpha=alpha)
    model.fit(train_predictors, train_target)

    # Rank features by the magnitude of their Ridge coefficients.
    importance = np.abs(model.coef_)

    # Select the indices of the highest-ranking features.
    selected_indices = np.argsort(importance)[-nr_of_features:]

    n_total   = train_predictors.shape[1]
    n_dropped = n_total - nr_of_features
    coef_min  = float(importance[np.argsort(importance)[-nr_of_features]])
    coef_max  = float(importance.max())

    logger.info(
        "[Ridge] Summary\n"
        "  Method          : Ridge feature selection (L2 regularisation)\n"
        "  Alpha           : %.4g\n"
        "  ── Input ────────────────────────────────────────────────────\n"
        "  Train input     : %d samples × %d features\n"
        "  Test  input     : %d samples × %d features\n"
        "  ── Selection ────────────────────────────────────────────────\n"
        "  |coef| range    : min selected = %.6g  |  global max = %.6g\n"
        "  Requested top-k : %d features\n"
        "  Dropped         : %d features (lowest |coefficient| rank)\n"
        "  ── Output ───────────────────────────────────────────────────\n"
        "  Train output    : %d samples × %d features\n"
        "  Test  output    : %d samples × %d features",
        alpha,
        train_predictors.shape[0], n_total,
        test_predictors.shape[0],  n_total,
        coef_min, coef_max,
        nr_of_features,
        n_dropped,
        train_predictors.shape[0], nr_of_features,
        test_predictors.shape[0],  nr_of_features,
    )

    # Apply the same indices to both splits; test data is never used in fitting.
    return train_predictors[:, selected_indices], test_predictors[:, selected_indices]
