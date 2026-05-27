"""
Domain service – Linear Regression grid search.

DDD: pure domain service that fits an OLS Linear Regression for each
dataset split in ``data_list`` and returns evaluation metrics.

Note: Linear Regression has no hyper-parameters to sweep, so a single
fit-and-evaluate pass is performed per split.

Each element of ``data_list`` must be a dict with keys:
    ``name``              – label for this dataset variant.
    ``train_predictors``  – 2-D float array for training.
    ``train_target``      – 1-D int array of training labels.
    ``test_predictors``   – 2-D float array for evaluation.
    ``test_target``       – 1-D int array of evaluation labels.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Domain service function
# ---------------------------------------------------------------------------

def grid_search_linear_model(data_list: list[dict]) -> list[dict]:
    """Fit and evaluate a Linear Regression model for each dataset split.

    Args:
        data_list: List of dataset-split dicts (see module docstring).

    Returns:
        A list of result dicts, each containing the reduction type, model
        name, and a ``scores`` sub-dict with MSE, MAE, F1, and accuracy.
    """
    rows: list[dict] = []

    for split in data_list:

        # Ordinary least-squares regression; no hyper-parameters to tune.
        model = LinearRegression()

        # Compute per-sample weights to counteract class imbalance.
        sample_weight = compute_sample_weight("balanced", split["train_target"])

        # Fit on the training split.
        model.fit(split["train_predictors"], split["train_target"], sample_weight=sample_weight)

        # Predict on the held-out test set.
        predictions = model.predict(split["test_predictors"])

        # Round predictions to the nearest integer for the within-±1-group tolerance metric.
        rounded_predictions = predictions.round().astype(int)

        # Collect evaluation metrics for this split.
        rows.append(
            {
                "dimension_reduction_type": split["name"],
                "model": "Linear Regression",
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
