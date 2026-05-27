"""
Domain service – ElasticNetCV regressor.

DDD: pure domain service.  ElasticNet combines L1 (sparsity) and L2 (grouping
of correlated features) in a single regressor.  The CV variant tunes alpha and
l1_ratio internally via cross-validation, so no external hyper-parameter grid
is needed.

Each element of ``data_list`` must be a dict with keys:
    ``name``              – label for this dataset variant.
    ``train_predictors``  – 2-D float array for training.
    ``train_target``      – 1-D float array of training labels.
    ``test_predictors``   – 2-D float array for evaluation.
    ``test_target``       – 1-D float array of evaluation labels.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_sample_weight


def grid_search_elasticnet_model(data_list: list[dict]) -> list[dict]:
    """Fit and evaluate an ElasticNetCV regressor for each dataset split.

    ``ElasticNetCV`` selects the best ``alpha`` and ``l1_ratio`` via internal
    5-fold cross-validation on the training data.  One record is produced per
    split (no external hyper-parameter loop).

    Args:
        data_list: List of dataset-split dicts (see module docstring).

    Returns:
        A list of result dicts with reduction type, model name, the
        CV-selected hyper-parameters, and a ``scores`` sub-dict.
    """
    rows: list[dict] = []

    for split in data_list:
        sample_weight = compute_sample_weight("balanced", split["train_target"])

        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
            cv=5,
            max_iter=10000,
            n_jobs=-1,
        )
        model.fit(
            split["train_predictors"],
            split["train_target"],
            sample_weight=sample_weight,
        )

        predictions = model.predict(split["test_predictors"])
        rounded_predictions = predictions.round().astype(int)

        rows.append(
            {
                "dimension_reduction_type": split["name"],
                "model": "ElasticNetCV",
                "alpha": float(model.alpha_),
                "l1_ratio": float(model.l1_ratio_),
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
