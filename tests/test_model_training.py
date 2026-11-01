"""Tests for Module 2 — model training, feature engineering, and evaluation."""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.train_model import (
    _slope,
    add_slope_features,
    evaluate_threshold,
    select_threshold,
    FEATURE_COLS,
    TARGET_COL,
    THRESHOLD_CANDIDATES,
)


def test_slope_function_basic():
    """_slope should return the linear regression slope of a series."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _slope(s)
    assert abs(result - 1.0) < 1e-10, f"Expected slope ~1.0, got {result}"


def test_slope_function_negative():
    """_slope should return negative slope for decreasing series."""
    s = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    result = _slope(s)
    assert abs(result - (-1.0)) < 1e-10, f"Expected slope ~-1.0, got {result}"


def test_slope_function_flat():
    """_slope should return ~0 for a flat series."""
    s = pd.Series([3.0, 3.0, 3.0, 3.0])
    result = _slope(s)
    assert abs(result - 0.0) < 1e-10, f"Expected slope ~0.0, got {result}"


def test_slope_function_single_element():
    """_slope should return NaN for a single element."""
    s = pd.Series([5.0])
    result = _slope(s)
    assert np.isnan(result), f"Expected NaN for single element, got {result}"


def test_add_slope_features_adds_columns():
    """add_slope_features should add the three slope columns."""
    df = pd.DataFrame({
        "unit_id": [1, 1, 1, 1, 1, 1, 1, 1],
        "day": [1, 2, 3, 4, 5, 6, 7, 8],
        "tds_ppm": [20.0, 22.0, 25.0, 28.0, 32.0, 35.0, 39.0, 42.0],
        "flow_rate_lpm": [2.0, 1.95, 1.9, 1.85, 1.8, 1.75, 1.7, 1.65],
        "pressure_diff_psi": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
        "days_to_failure": [7, 6, 5, 4, 3, 2, 1, 0],
        "failed_within_7d": [1, 1, 1, 1, 1, 1, 1, 1],
    })
    result = add_slope_features(df.copy())
    assert "tds_slope_5d" in result.columns
    assert "flow_slope_5d" in result.columns
    assert "pressure_slope_5d" in result.columns
    assert len(result) == len(df)


def test_add_slope_features_preserves_unit_id():
    """add_slope_features should not drop the unit_id column."""
    df = pd.DataFrame({
        "unit_id": [1, 1, 1, 2, 2, 2],
        "day": [1, 2, 3, 1, 2, 3],
        "tds_ppm": [20.0, 22.0, 25.0, 20.0, 22.0, 25.0],
        "flow_rate_lpm": [2.0, 1.9, 1.8, 2.0, 1.9, 1.8],
        "pressure_diff_psi": [2.0, 2.1, 2.2, 2.0, 2.1, 2.2],
        "days_to_failure": [5, 4, 3, 5, 4, 3],
        "failed_within_7d": [1, 1, 1, 1, 1, 1],
    })
    result = add_slope_features(df.copy())
    assert "unit_id" in result.columns
    assert (result["unit_id"] == df["unit_id"]).all()


def test_evaluate_threshold_basic():
    """evaluate_threshold should return metrics dict with expected keys."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.9])
    result = evaluate_threshold(y_true, y_prob, 0.5)
    expected_keys = {
        "threshold", "accuracy", "precision", "recall", "f1",
        "tn", "fp", "fn", "tp", "false_negative_rate", "false_positive_rate",
    }
    assert set(result.keys()) == expected_keys, f"Keys mismatch: {set(result.keys())}"
    assert result["threshold"] == 0.5


def test_evaluate_threshold_perfect():
    """With perfect predictions, all metrics should be 1.0 except FNR/FPR = 0."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    result = evaluate_threshold(y_true, y_prob, 0.5)
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["false_negative_rate"] == 0.0
    assert result["false_positive_rate"] == 0.0
    assert result["tn"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tp"] == 2


def test_select_threshold_maximizes_f1():
    """select_threshold should pick the threshold with highest F1 on validation data."""
    y_val = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_val_prob = np.array([0.1, 0.3, 0.4, 0.7, 0.2, 0.8, 0.35, 0.6])
    best, all_results = select_threshold(y_val, y_val_prob, THRESHOLD_CANDIDATES)
    assert "threshold" in best
    assert "f1" in best
    assert 0.0 <= best["threshold"] <= 1.0
    assert len(all_results) == len(THRESHOLD_CANDIDATES)


def test_threshold_candidates_range():
    """THRESHOLD_CANDIDATES should span a reasonable range."""
    assert min(THRESHOLD_CANDIDATES) < 0.5
    assert max(THRESHOLD_CANDIDATES) > 0.5
    assert len(THRESHOLD_CANDIDATES) >= 5


def test_feature_cols_expected():
    """FEATURE_COLS should contain the expected 7 features."""
    expected = {
        "tds_slope_5d", "flow_slope_5d", "pressure_slope_5d",
        "tds_ppm", "flow_rate_lpm", "pressure_diff_psi", "day",
    }
    assert set(FEATURE_COLS) == expected


def test_target_col():
    """TARGET_COL should be 'failed_within_7d'."""
    assert TARGET_COL == "failed_within_7d"


def test_training_metrics_file_exists():
    """training_metrics.json should exist and contain expected keys."""
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "training_metrics.json",
    )
    assert os.path.exists(metrics_path), "training_metrics.json not found"
    with open(metrics_path) as f:
        metrics = json.load(f)
    expected_keys = {
        "model_name", "threshold", "test_auc", "test_accuracy",
        "test_precision", "test_recall", "test_f1",
        "test_tn", "test_fp", "test_fn", "test_tp",
        "test_false_negative_rate", "test_false_positive_rate",
        "train_units", "val_units", "test_units",
        "train_rows", "val_rows", "test_rows",
        "model_comparison", "threshold_evaluation",
    }
    assert expected_keys.issubset(set(metrics.keys())), (
        f"Missing keys: {expected_keys - set(metrics.keys())}"
    )


def test_model_metrics_values_reasonable():
    """Test metrics should be in valid ranges."""
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "training_metrics.json",
    )
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert 0.0 <= metrics["test_precision"] <= 1.0
    assert 0.0 <= metrics["test_recall"] <= 1.0
    assert 0.0 <= metrics["test_f1"] <= 1.0
    assert 0.0 <= metrics["test_false_negative_rate"] <= 1.0
    assert 0.0 <= metrics["test_false_positive_rate"] <= 1.0
    assert 0.0 <= metrics["threshold"] <= 1.0
    assert metrics["test_tn"] + metrics["test_fp"] + metrics["test_fn"] + metrics["test_tp"] > 0


def test_model_comparison_has_three_models():
    """Model comparison should include GradientBoosting, RandomForest, LogisticRegression."""
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "training_metrics.json",
    )
    with open(metrics_path) as f:
        metrics = json.load(f)
    comparison = metrics["model_comparison"]
    assert "GradientBoosting" in comparison
    assert "RandomForest" in comparison
    assert "LogisticRegression" in comparison

