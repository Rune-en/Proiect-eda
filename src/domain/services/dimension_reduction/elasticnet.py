"""
Domain service – ElasticNet-based feature selection.

DDD: pure domain service – no I/O, no side effects.  ElasticNet (L1 + L2
regularisation) selects features while grouping correlated predictors, making
it more stable than pure Lasso when several genes express the same protein.

12-factor: hyper-parameters are defined here as module constants (Factor V).
"""

import logging

import numpy as np
from sklearn.linear_model import ElasticNet

logger = logging.getLogger(__name__)


def reduce_features_elasticnet(
    train_predictors: np.ndarray,
    train_target: np.ndarray,
    test_predictors: np.ndarray,
    nr_of_features: int = 200,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top features by ElasticNet coefficient magnitude.

    Fitted exclusively on training data.  The same feature indices are then
    applied to *test_predictors* (no leakage).

    Args:
        train_predictors: Training feature matrix ``(n_train, n_features)``.
        train_target: Training labels ``(n_train,)``.
        test_predictors: Test feature matrix ``(n_test, n_features)``.
        nr_of_features: Number of top-|coefficient| features to keep.
        alpha: Overall regularisation strength.
        l1_ratio: Mix between L1 (1.0) and L2 (0.0).  Default 0.5 gives equal
            weight to sparsity and grouping.

    Returns:
        ``(train_reduced, test_reduced)`` both with ``nr_of_features`` columns.
    """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(train_predictors, train_target)

    abs_coefs = np.abs(model.coef_)
    top_indices = np.argsort(abs_coefs)[-nr_of_features:]

    train_reduced = train_predictors[:, top_indices]
    test_reduced  = test_predictors[:, top_indices]

    n_nonzero = int(np.sum(abs_coefs > 0))

    logger.info(
        "[ElasticNet] Summary\n"
        "  Method          : ElasticNet feature selection (L1+L2)\n"
        "  Alpha           : %g  |  l1_ratio : %g\n"
        "  ── Input ────────────────────────────────────────────────────\n"
        "  Train input     : %d samples × %d features\n"
        "  Test  input     : %d samples × %d features\n"
        "  ── Selection ────────────────────────────────────────────────\n"
        "  Non-zero coefs  : %d features\n"
        "  Requested top-k : %d features\n"
        "  ── Output ───────────────────────────────────────────────────\n"
        "  Train output    : %d samples × %d features\n"
        "  Test  output    : %d samples × %d features",
        alpha, l1_ratio,
        train_predictors.shape[0], train_predictors.shape[1],
        test_predictors.shape[0],  test_predictors.shape[1],
        n_nonzero, nr_of_features,
        train_reduced.shape[0], nr_of_features,
        test_reduced.shape[0],  nr_of_features,
    )

    return train_reduced, test_reduced
