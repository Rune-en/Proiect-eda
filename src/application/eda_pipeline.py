"""
Application service – EDA pipeline orchestrator.

DDD: the *application layer* sits between the infrastructure and domain layers.
It coordinates the flow:

    1. Load raw data      (infrastructure)
    2. Reduce features    (domain – Lasso / Ridge / PCA)
    3. Split train/test   (domain – scikit-learn utility)
    4. Run grid searches  (domain – ml_models services)
    5. Persist results    (infrastructure)

12-factor:
    - Factor VI  (Processes): the pipeline is a stateless, one-shot process.
    - Factor XI  (Logs):      structured logging via the standard ``logging``
                              module; callers attach their own handlers.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import settings
from src.domain.services.dimension_reduction import elasticnet, lasso, pca, ridge
from src.domain.services.ml_models import (
    # decision_tree,  # disabled – consistently poor R² / unstable
    elasticnet_model,
    gradient_boosting,
    linear_model,
    poisson_model,
    random_forest,
    svr_model,
)
from src.infrastructure import dataset_loader

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as 'Xm Ys' or 'Ys'."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _build_split_dict(
    name: str,
    predictors_train: np.ndarray,
    predictors_test: np.ndarray,
    target_train: np.ndarray,
    target_test: np.ndarray,
) -> dict[str, Any]:
    """Package train/test arrays into the dict format expected by ML services.

    Args:
        name: Human-readable label for this dataset variant (e.g. the
            dimension-reduction method and its parameters).
        predictors_train: Training feature matrix.
        predictors_test: Test feature matrix.
        target_train: Training labels.
        target_test: Test labels.

    Returns:
        A dict with keys ``name``, ``train_predictors``, ``train_target``,
        ``test_predictors``, ``test_target``.
    """
    return {
        "name": name,
        "train_predictors": predictors_train,
        "train_target": target_train,
        "test_predictors": predictors_test,
        "test_target": target_test,
    }


def _resampled(
    smote_obj: SMOTE, x_tr: np.ndarray, y_tr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return SMOTE-resampled arrays when USE_SMOTE is enabled, else pass through."""
    if settings.USE_SMOTE:
        x_sm, y_sm = smote_obj.fit_resample(x_tr, y_tr.astype(int))
        return x_sm, y_sm.astype(float)
    return x_tr, y_tr


def run_pipeline() -> dict[str, list[dict]]:
    """Execute the full EDA pipeline end-to-end.

    Steps:
        1. Load and preprocess the raw CSV using the infrastructure layer.
        2. Separate features (X) from the target (y).
        3. Apply each dimension-reduction strategy (Lasso, Ridge, PCA) with
           a representative set of hyper-parameters to produce reduced
           feature matrices.
        4. Split each reduced matrix into train/test sets.
        5. Run all four ML model grid searches across the splits.
        6. Save the processed dataset to the configured output path.

    Returns:
        A dict mapping model names to their list of grid-search result records.
    """
    # ------------------------------------------------------------------
    # 1. Load data (infrastructure concern)
    # ------------------------------------------------------------------
    df: pd.DataFrame = dataset_loader.load_dataset(settings.DATASET_PATH)

    # ------------------------------------------------------------------
    # 2. Separate features and target
    # ------------------------------------------------------------------
    # ``Gleason Group`` is the ordinal classification target (0–5).
    target: np.ndarray = df[settings.TARGET_COLUMN].to_numpy().astype(float)

    # All remaining columns are numeric gene-expression features.
    predictors: np.ndarray = df.drop(columns=[settings.TARGET_COLUMN]).to_numpy().astype(float)

    logger.info(
        "Feature matrix shape: %s | Target shape: %s",
        predictors.shape,
        target.shape,
    )

    # ------------------------------------------------------------------
    # 2b. Optional near-zero-variance gene removal (global preprocessing).
    #     Genes with variance < VARIANCE_THRESHOLD across all samples are
    #     dropped before any fold-level processing.
    # ------------------------------------------------------------------
    if settings.REMOVE_LOW_VARIANCE:
        vt = VarianceThreshold(threshold=settings.VARIANCE_THRESHOLD)
        n_orig = predictors.shape[1]
        predictors = vt.fit_transform(predictors)
        logger.info(
            "VarianceThreshold: removed %d low-variance features, kept %d / %d (threshold=%.3f)",
            n_orig - predictors.shape[1], predictors.shape[1], n_orig, settings.VARIANCE_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # 3. Cross-validation or single stratified split.
    #    USE_CV=True  → 5-fold StratifiedKFold (default)
    #    USE_CV=False → one stratified train/test split
    #    Each fold re-fits all reductions on its training data only.
    #    Reductions are saved to disk only for fold 0 (representative sample).
    # ------------------------------------------------------------------
    if settings.USE_CV:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.RANDOM_STATE)
        fold_splits: list[tuple[int, tuple]] = list(enumerate(kf.split(predictors, target)))
        n_folds = kf.n_splits
    else:
        _tr_idx, _te_idx = train_test_split(
            np.arange(len(target)), test_size=settings.TEST_SIZE,
            stratify=target.astype(int), random_state=settings.RANDOM_STATE,
        )
        fold_splits = [(0, (_tr_idx, _te_idx))]
        n_folds = 1

    smote = SMOTE(random_state=settings.RANDOM_STATE)
    _smote_suffix = " + SMOTE" if settings.USE_SMOTE else ""
    data_splits: list[dict] = []

    for fold_idx, (train_idx, test_idx) in fold_splits:
        x_train_fold = predictors[train_idx]
        x_test_fold  = predictors[test_idx]
        y_train_fold = target[train_idx]
        y_test_fold  = target[test_idx]

        save = fold_idx == 0  # persist reductions only for the first fold

        # --- Lasso ---
        x_tr_lasso, x_te_lasso = lasso.reduce_features_lasso(
            x_train_fold, y_train_fold, x_test_fold, nr_of_features=100, alpha=1.0
        )
        if save:
            dataset_loader.save_reduction(
                "lasso", x_tr_lasso, x_te_lasso, y_train_fold, y_test_fold, settings.REDUCTIONS_DIR
            )
        x_tr_lasso_sm, y_tr_lasso_sm = _resampled(smote, x_tr_lasso, y_train_fold)
        data_splits.append(_build_split_dict(
            f"Lasso (alpha=1.0, top-100){_smote_suffix}", x_tr_lasso_sm, x_te_lasso, y_tr_lasso_sm, y_test_fold,
        ))

        # --- Ridge ---
        x_tr_ridge, x_te_ridge = ridge.reduce_features_ridge(
            x_train_fold, y_train_fold, x_test_fold, nr_of_features=100, alpha=1.0
        )
        if save:
            dataset_loader.save_reduction(
                "ridge", x_tr_ridge, x_te_ridge, y_train_fold, y_test_fold, settings.REDUCTIONS_DIR
            )
        x_tr_ridge_sm, y_tr_ridge_sm = _resampled(smote, x_tr_ridge, y_train_fold)
        data_splits.append(_build_split_dict(
            f"Ridge (alpha=1.0, top-100){_smote_suffix}", x_tr_ridge_sm, x_te_ridge, y_tr_ridge_sm, y_test_fold,
        ))

        # --- PCA (full gene space) ---
        x_tr_pca, x_te_pca = pca.reduce_features_pca(
            x_train_fold, x_test_fold, variance_threshold=50
        )
        if save:
            dataset_loader.save_reduction(
                "pca", x_tr_pca, x_te_pca, y_train_fold, y_test_fold, settings.REDUCTIONS_DIR
            )
        x_tr_pca_sm, y_tr_pca_sm = _resampled(smote, x_tr_pca, y_train_fold)
        data_splits.append(_build_split_dict(
            f"PCA (n_components=50){_smote_suffix}", x_tr_pca_sm, x_te_pca, y_tr_pca_sm, y_test_fold,
        ))

        # --- Lasso→PCA (chained) ---
        x_tr_lasso_pca, x_te_lasso_pca = pca.reduce_features_pca(
            x_tr_lasso, x_te_lasso, variance_threshold=0.95
        )
        if save:
            dataset_loader.save_reduction(
                "lasso_pca", x_tr_lasso_pca, x_te_lasso_pca, y_train_fold, y_test_fold, settings.REDUCTIONS_DIR
            )
        x_tr_lasso_pca_sm, y_tr_lasso_pca_sm = _resampled(smote, x_tr_lasso_pca, y_train_fold)
        data_splits.append(_build_split_dict(
            f"Lasso->PCA (top-100, 95%var){_smote_suffix}", x_tr_lasso_pca_sm, x_te_lasso_pca, y_tr_lasso_pca_sm, y_test_fold,
        ))

        # --- Ridge→PCA (chained) ---
        x_tr_ridge_pca, x_te_ridge_pca = pca.reduce_features_pca(
            x_tr_ridge, x_te_ridge, variance_threshold=0.95
        )
        if save:
            dataset_loader.save_reduction(
                "ridge_pca", x_tr_ridge_pca, x_te_ridge_pca, y_train_fold, y_test_fold, settings.REDUCTIONS_DIR
            )
        x_tr_ridge_pca_sm, y_tr_ridge_pca_sm = _resampled(smote, x_tr_ridge_pca, y_train_fold)
        data_splits.append(_build_split_dict(
            f"Ridge->PCA (top-100, 95%var){_smote_suffix}", x_tr_ridge_pca_sm, x_te_ridge_pca, y_tr_ridge_pca_sm, y_test_fold,
        ))

        # --- ElasticNet (top-200) ---
        x_tr_en, x_te_en = elasticnet.reduce_features_elasticnet(
            x_train_fold, y_train_fold, x_test_fold, nr_of_features=200, alpha=1.0, l1_ratio=0.5
        )
        if save:
            dataset_loader.save_reduction(
                "elasticnet", x_tr_en, x_te_en, y_train_fold, y_test_fold, settings.REDUCTIONS_DIR
            )
        x_tr_en_sm, y_tr_en_sm = _resampled(smote, x_tr_en, y_train_fold)
        data_splits.append(_build_split_dict(
            f"ElasticNet (top-200, l1r=0.5){_smote_suffix}", x_tr_en_sm, x_te_en, y_tr_en_sm, y_test_fold,
        ))

        # --- ElasticNet→PCA (chained) ---
        x_tr_en_pca, x_te_en_pca = pca.reduce_features_pca(
            x_tr_en, x_te_en, variance_threshold=0.95
        )
        if save:
            dataset_loader.save_reduction(
                "elasticnet_pca", x_tr_en_pca, x_te_en_pca, y_train_fold, y_test_fold, settings.REDUCTIONS_DIR
            )
        x_tr_en_pca_sm, y_tr_en_pca_sm = _resampled(smote, x_tr_en_pca, y_train_fold)
        data_splits.append(_build_split_dict(
            f"ElasticNet->PCA (top-200, 95%var){_smote_suffix}", x_tr_en_pca_sm, x_te_en_pca, y_tr_en_pca_sm, y_test_fold,
        ))

    logger.info(
        "CV splits ready: %d %s × 7 reductions%s = %d total splits",
        n_folds,
        "folds" if settings.USE_CV else "split",
        _smote_suffix,
        len(data_splits),
    )

    # ------------------------------------------------------------------
    # 4. Run ML model grid searches (domain concern)
    # ------------------------------------------------------------------
    _MODEL_FUNS = [
        # ("decision_tree",     decision_tree.grid_search_decision_tree),  # disabled
        ("elasticnet_model",  elasticnet_model.grid_search_elasticnet_model),
        ("gradient_boosting", gradient_boosting.grid_search_gradient_boosting),
        ("linear_model",      linear_model.grid_search_linear_model),
        ("poisson_model",     poisson_model.grid_search_poisson_model),
        ("random_forest",     random_forest.grid_search_random_forest),
        ("svr_model",         svr_model.grid_search_svr),
    ]
    n_models = len(_MODEL_FUNS)
    logger.info("Running grid searches across %d splits × %d models …", len(data_splits), n_models)

    results: dict[str, list[dict]] = {}
    _t0 = time.monotonic()
    for _i, (_model_name, _fn) in enumerate(_MODEL_FUNS, start=1):
        logger.info("  [%d/%d] Starting: %s", _i, n_models, _model_name)
        results[_model_name] = _fn(data_splits)
        _elapsed = time.monotonic() - _t0
        _avg_per_model = _elapsed / _i
        _remaining = _avg_per_model * (n_models - _i)
        _eta = datetime.now() + timedelta(seconds=_remaining)
        logger.info(
            "  [%d/%d] Done:    %-20s | elapsed %s | remaining ~%s | ETA %s",
            _i, n_models, _model_name,
            _fmt_duration(_elapsed), _fmt_duration(_remaining),
            _eta.strftime("%H:%M:%S") if _remaining > 0 else "—",
        )

    logger.info("Grid search complete – total %s.", _fmt_duration(time.monotonic() - _t0))

    # ------------------------------------------------------------------
    # 5. Persist processed dataset (infrastructure concern)
    # ------------------------------------------------------------------
    dataset_loader.save_dataset(df, settings.OUTPUT_PATH)

    # ------------------------------------------------------------------
    # 6. Persist per-model grid-search results (infrastructure concern)
    # ------------------------------------------------------------------
    dataset_loader.save_model_results(results, settings.RESULTS_DIR)

    return results
