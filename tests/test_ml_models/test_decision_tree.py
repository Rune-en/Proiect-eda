"""
Unit tests for the Decision Tree grid-search domain service.

Tests use a tiny in-memory dataset so no CSV or real data is needed.
They verify:

    - The function returns a non-empty list of result records.
    - Each record contains the expected keys.
    - The number of records equals len(data_list) × len(depths) × len(splits).
    - Score values are numeric and in expected ranges.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.domain.services.ml_models.decision_tree import (
    DECISION_TREE_MAX_DEPTHS,
    DECISION_TREE_MIN_SAMPLES_SPLITS,
    grid_search_decision_tree,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_split(name: str = "test") -> dict:
    """Return a minimal dataset split dict using a synthetic classification dataset."""
    X, y = make_classification(
        n_samples=60,
        n_features=10,
        n_classes=3,
        n_informative=5,
        random_state=0,
    )
    y = y.astype(float)
    mid = len(X) // 2
    return {
        "name": name,
        "train_predictors": X[:mid],
        "train_target": y[:mid],
        "test_predictors": X[mid:],
        "test_target": y[mid:],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGridSearchDecisionTree:
    """Test suite for ``grid_search_decision_tree``."""

    def test_returns_nonempty_list(self) -> None:
        """The function must return at least one result record."""
        results = grid_search_decision_tree([_make_split()])
        assert len(results) > 0

    def test_result_count_equals_grid_size(self) -> None:
        """Number of records must equal n_splits × n_depths × n_min_splits."""
        n_splits = 2
        data_list = [_make_split(f"split-{i}") for i in range(n_splits)]
        expected_count = (
            n_splits
            * len(DECISION_TREE_MAX_DEPTHS)
            * len(DECISION_TREE_MIN_SAMPLES_SPLITS)
        )
        results = grid_search_decision_tree(data_list)
        assert len(results) == expected_count

    def test_each_record_has_required_keys(self) -> None:
        """Every result record must contain the mandatory top-level keys."""
        required_keys = {
            "dimension_reduction_type",
            "model",
            "max_depth",
            "min_samples_split",
            "scores",
        }
        results = grid_search_decision_tree([_make_split()])
        for record in results:
            assert required_keys.issubset(record.keys()), (
                f"Missing keys: {required_keys - record.keys()}"
            )

    def test_scores_sub_dict_has_all_metrics(self) -> None:
        """The ``scores`` dict inside each record must contain all regression metrics."""
        required_metrics = {
            "rmse",
            "mae",
            "r2",
            "spearman_r",
            "within_1_acc",
        }
        results = grid_search_decision_tree([_make_split()])
        for record in results:
            assert required_metrics.issubset(record["scores"].keys())

    def test_within_1_acc_in_valid_range(self) -> None:
        """Within-±1-group tolerance accuracy must lie within [0, 1]."""
        results = grid_search_decision_tree([_make_split()])
        for record in results:
            acc = record["scores"]["within_1_acc"]
            assert 0.0 <= acc <= 1.0, f"within_1_acc out of range: {acc}"

    def test_model_label_is_correct(self) -> None:
        """The ``model`` field must equal 'Decision Tree' for all records."""
        results = grid_search_decision_tree([_make_split()])
        for record in results:
            assert record["model"] == "Decision Tree"

    def test_dimension_reduction_type_matches_input_name(self) -> None:
        """The ``dimension_reduction_type`` must match the split's ``name``."""
        split_name = "MyReduction"
        results = grid_search_decision_tree([_make_split(name=split_name)])
        for record in results:
            assert record["dimension_reduction_type"] == split_name
