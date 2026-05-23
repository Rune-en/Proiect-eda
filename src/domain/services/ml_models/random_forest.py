"""
Domain service – Random Forest grid search.

DDD: pure domain service that sweeps the Random Forest hyper-parameter space
across a list of pre-processed dataset splits.

Each element of ``data_list`` must be a dict with keys:
    ``name``              – label for this dataset variant.
    ``train_predictors``  – 2-D float array for training.
    ``train_target``      – 1-D int array of training labels.
    ``test_predictors``   – 2-D float array for evaluation.
    ``test_target``       – 1-D int array of evaluation labels.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Hyper-parameter grid
# ---------------------------------------------------------------------------

# Number of trees in the forest.
RANDOM_FOREST_N_ESTIMATORS: list[int] = [10, 20, 50, 100]

# Maximum depth of each tree; ``None`` means unconstrained growth.
RANDOM_FOREST_MAX_DEPTHS: list[int | None] = [10, 20, 30, None]


# ---------------------------------------------------------------------------
# Domain service function
# ---------------------------------------------------------------------------

def grid_search_random_forest(data_list: list[dict]) -> list[dict]:
    """Run a Random Forest grid search over all dataset splits.

    For every combination of (dataset split × n_estimators × max_depth) the
    function fits a :class:`~sklearn.ensemble.RandomForestRegressor`,
    evaluates it on the test set and appends a result record.

    Args:
        data_list: List of dataset-split dicts (see module docstring).

    Returns:
        A list of result dicts with reduction type, model name,
        hyper-parameter values, and a ``scores`` sub-dict.
    """
    rows: list[dict] = []

    for split in data_list:
        for n_estimators in RANDOM_FOREST_N_ESTIMATORS:
            for max_depth in RANDOM_FOREST_MAX_DEPTHS:

                # Build a new ensemble for each hyper-parameter combination.
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                )

                # Compute per-sample weights to counteract class imbalance.
                sample_weight = compute_sample_weight("balanced", split["train_target"])

                # Fit on the training data.
                model.fit(split["train_predictors"], split["train_target"], sample_weight=sample_weight)

                # Predict on the held-out test set.
                predictions = model.predict(split["test_predictors"])

                # Round to nearest int for the within-±1-group tolerance metric.
                rounded_predictions = predictions.round().astype(int)

                rows.append(
                    {
                        "dimension_reduction_type": split["name"],
                        "model": "Random Forest",
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
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
                        },
                    }
                )

    return rows
