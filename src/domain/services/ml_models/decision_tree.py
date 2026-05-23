"""
Domain service – Decision Tree grid search.

DDD: this service encapsulates the logic for sweeping the Decision Tree
hyper-parameter space across a list of pre-processed dataset splits.  It is
pure (no I/O) and depends only on domain types (numpy arrays in dicts).

Each element of ``data_list`` must be a dict with keys:
    ``name``              – human-readable label for this dataset variant.
    ``train_predictors``  – 2-D float array for training.
    ``train_target``      – 1-D int array of training labels.
    ``test_predictors``   – 2-D float array for evaluation.
    ``test_target``       – 1-D int array of evaluation labels.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Hyper-parameter grid
# ---------------------------------------------------------------------------

# Maximum depth of each tree.  ``None`` means nodes expand until all leaves
# are pure or contain fewer than ``min_samples_split`` samples.
DECISION_TREE_MAX_DEPTHS: list[int | None] = [10, 20, 30, None]

# Minimum number of samples required to split an internal node.
DECISION_TREE_MIN_SAMPLES_SPLITS: list[int] = [2, 5, 10]


# ---------------------------------------------------------------------------
# Domain service type alias
# ---------------------------------------------------------------------------

# Each result record returned by the grid search.
_ResultRecord = dict


# ---------------------------------------------------------------------------
# Domain service function
# ---------------------------------------------------------------------------

def grid_search_decision_tree(data_list: list[dict]) -> list[_ResultRecord]:
    """Run a Decision Tree grid search over all dataset splits.

    For every combination of (dataset split × max_depth × min_samples_split)
    the function fits a :class:`~sklearn.tree.DecisionTreeRegressor`, evaluates
    it on the test set and appends a result record to the output list.

    Args:
        data_list: List of dataset-split dicts (see module docstring).

    Returns:
        A list of result dicts, each containing the reduction type, model
        name, hyper-parameter values, and a ``scores`` sub-dict with
        MSE, MAE, F1, and accuracy.
    """
    rows: list[_ResultRecord] = []

    for split in data_list:
        for max_depth in DECISION_TREE_MAX_DEPTHS:
            for min_samples_split in DECISION_TREE_MIN_SAMPLES_SPLITS:

                # Instantiate a fresh model for each hyper-parameter combination.
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                )

                # Compute per-sample weights to counteract class imbalance.
                sample_weight = compute_sample_weight("balanced", split["train_target"])

                # Fit on the training portion of this split.
                model.fit(split["train_predictors"], split["train_target"], sample_weight=sample_weight)

                # Generate predictions on the held-out test set.
                predictions = model.predict(split["test_predictors"])

                # Round predictions to integers for the within-±1-group tolerance metric.
                rounded_predictions = predictions.round().astype(int)

                # Collect evaluation metrics.
                rows.append(
                    {
                        "dimension_reduction_type": split["name"],
                        "model": "Decision Tree",
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
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
