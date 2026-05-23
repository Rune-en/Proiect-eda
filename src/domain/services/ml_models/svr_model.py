"""
Domain service – SVR (RBF kernel) grid search.

DDD: pure domain service.  SVR with an RBF kernel maps the reduced feature
space into a higher-dimensional Hilbert space, allowing nonlinear separation
of ordinal Gleason groups that a linear model cannot capture.

Each element of ``data_list`` must be a dict with keys:
    ``name``              – label for this dataset variant.
    ``train_predictors``  – 2-D float array for training.
    ``train_target``      – 1-D float array of training labels.
    ``test_predictors``   – 2-D float array for evaluation.
    ``test_target``       – 1-D float array of evaluation labels.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Hyper-parameter grid
# ---------------------------------------------------------------------------

SVR_C_VALUES: list[float] = [0.1, 1.0, 10.0]
SVR_GAMMA_VALUES: list[str] = ["scale", "auto"]


# ---------------------------------------------------------------------------
# Domain service function
# ---------------------------------------------------------------------------

def grid_search_svr(data_list: list[dict]) -> list[dict]:
    """Run an SVR (RBF kernel) grid search over all dataset splits.

    For every combination of (dataset split × C × gamma) the function fits a
    :class:`~sklearn.svm.SVR` with ``kernel='rbf'``, evaluates it on the test
    set, and appends a result record.

    Args:
        data_list: List of dataset-split dicts (see module docstring).

    Returns:
        A list of result dicts with reduction type, model name,
        hyper-parameter values, and a ``scores`` sub-dict.
    """
    rows: list[dict] = []

    for split in data_list:
        # SVR is sensitive to feature scale; standardise each split separately.
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(split["train_predictors"])
        x_test_scaled  = scaler.transform(split["test_predictors"])

        sample_weight = compute_sample_weight("balanced", split["train_target"])

        for C in SVR_C_VALUES:
            for gamma in SVR_GAMMA_VALUES:
                model = SVR(kernel="rbf", C=C, gamma=gamma)
                model.fit(x_train_scaled, split["train_target"], sample_weight=sample_weight)

                predictions = model.predict(x_test_scaled)
                rounded_predictions = predictions.round().astype(int)

                rows.append(
                    {
                        "dimension_reduction_type": split["name"],
                        "model": "SVR (rbf)",
                        "C": C,
                        "gamma": gamma,
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
