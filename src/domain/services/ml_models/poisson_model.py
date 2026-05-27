"""
Domain service – Poisson Regression grid search.

DDD: pure domain service.  Poisson regression is appropriate when the target
is a non-negative count variable – the integer Gleason Group labels (0–5)
satisfy this constraint.

Each element of ``data_list`` must be a dict with keys:
    ``name``              – label for this dataset variant.
    ``train_predictors``  – 2-D float array for training.
    ``train_target``      – 1-D int array of training labels (≥ 0).
    ``test_predictors``   – 2-D float array for evaluation.
    ``test_target``       – 1-D int array of evaluation labels.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Hyper-parameter grid
# ---------------------------------------------------------------------------

# L2 regularisation strengths to sweep over.
POISSON_ALPHAS: list[float] = [0.1, 0.5, 1.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Domain service function
# ---------------------------------------------------------------------------

def grid_search_poisson_model(data_list: list[dict]) -> list[dict]:
    """Run a Poisson Regression grid search over all dataset splits.

    For every combination of (dataset split × alpha) the function fits a
    :class:`~sklearn.linear_model.PoissonRegressor`, evaluates it on the
    test set and appends a result record to the output.

    Args:
        data_list: List of dataset-split dicts (see module docstring).

    Returns:
        A list of result dicts with reduction type, model name, alpha, and
        a ``scores`` sub-dict with MSE, MAE, F1, and accuracy.
    """
    rows: list[dict] = []

    for split in data_list:
        for alpha in POISSON_ALPHAS:

            # Standardise features before fitting: Poisson regression uses a
            # log-link with gradient descent; large unscaled feature values
            # cause numerical overflow (matmul invalid value warnings).
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(split["train_predictors"])
            x_test_scaled = scaler.transform(split["test_predictors"])

            # Instantiate a Poisson regressor for this regularisation strength.
            model = PoissonRegressor(alpha=alpha, max_iter=1000)

            # Compute per-sample weights to counteract class imbalance.
            sample_weight = compute_sample_weight("balanced", split["train_target"])

            # Train on the scaled training split.
            model.fit(x_train_scaled, split["train_target"], sample_weight=sample_weight)

            # Predict; Poisson outputs non-negative floats.
            predictions = model.predict(x_test_scaled)

            # Round to the nearest integer for the within-±1-group tolerance metric.
            rounded_predictions = predictions.round().astype(int)

            rows.append(
                {
                    "dimension_reduction_type": split["name"],
                    "model": "Poisson Regression",
                    "alpha": alpha,
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
