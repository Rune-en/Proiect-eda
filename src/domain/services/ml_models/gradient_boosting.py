"""
Domain service – Gradient Boosting Regressor grid search.

DDD: pure domain service that sweeps the Gradient Boosting hyper-parameter
space across a list of pre-processed dataset splits.

Each element of ``data_list`` must be a dict with keys:
    ``name``              – label for this dataset variant.
    ``train_predictors``  – 2-D float array for training.
    ``train_target``      – 1-D float array of training labels.
    ``test_predictors``   – 2-D float array for evaluation.
    ``test_target``       – 1-D float array of evaluation labels.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Hyper-parameter grid
# ---------------------------------------------------------------------------

# Number of boosting stages (trees).
GRADIENT_BOOSTING_N_ESTIMATORS: list[int] = [50, 100, 200]

# Maximum depth of individual regression estimators.
GRADIENT_BOOSTING_MAX_DEPTHS: list[int] = [2, 3, 5]

# Shrinkage (step size) applied to each tree's contribution.
GRADIENT_BOOSTING_LEARNING_RATES: list[float] = [0.05, 0.1]


# ---------------------------------------------------------------------------
# Domain service function
# ---------------------------------------------------------------------------

def grid_search_gradient_boosting(data_list: list[dict]) -> list[dict]:
    """Run a Gradient Boosting grid search over all dataset splits.

    For every combination of
    (dataset split × n_estimators × max_depth × learning_rate)
    the function fits a
    :class:`~sklearn.ensemble.GradientBoostingRegressor`,
    evaluates it on the test set and appends a result record.

    Args:
        data_list: List of dataset-split dicts (see module docstring).

    Returns:
        A list of result dicts with reduction type, model name,
        hyper-parameter values, and a ``scores`` sub-dict.
    """
    rows: list[dict] = []

    for split in data_list:
        for n_estimators in GRADIENT_BOOSTING_N_ESTIMATORS:
            for max_depth in GRADIENT_BOOSTING_MAX_DEPTHS:
                for learning_rate in GRADIENT_BOOSTING_LEARNING_RATES:

                    model = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42,
                    )

                    # Compute per-sample weights to counteract class imbalance.
                    sample_weight = compute_sample_weight(
                        "balanced", split["train_target"]
                    )

                    model.fit(
                        split["train_predictors"],
                        split["train_target"],
                        sample_weight=sample_weight,
                    )

                    predictions = model.predict(split["test_predictors"])

                    # Round to nearest int for the within-±1-group tolerance metric.
                    rounded_predictions = predictions.round().astype(int)

                    rows.append(
                        {
                            "dimension_reduction_type": split["name"],
                            "model": "Gradient Boosting",
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "scores": {
                                "rmse": float(np.sqrt(mean_squared_error(
                                    split["test_target"], predictions
                                ))),
                                "mae": mean_absolute_error(
                                    split["test_target"], predictions
                                ),
                                "r2": r2_score(
                                    split["test_target"], predictions
                                ),
                                "spearman_r": float(spearmanr(
                                    split["test_target"], predictions
                                )[0]),
                                "within_1_acc": float(np.mean(
                                    np.abs(rounded_predictions - split["test_target"]) <= 1
                                )),
                                "qwk": float(cohen_kappa_score(
                                    split["test_target"],
                                    np.clip(rounded_predictions, 1, 5),
                                    weights="quadratic",
                                    labels=[1, 2, 3, 4, 5],
                                )),
                            },
                        }
                    )

    return rows
