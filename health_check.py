"""Health check for AtomQuest FilterGuard artifacts.

Verifies that required data and model artifacts exist and are loadable.
Run with:  python health_check.py
"""

import json
import os
import sys

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_filter_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "filterguard_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "training_metrics.json")

FEATURE_COLS = [
    "tds_slope_5d",
    "flow_slope_5d",
    "pressure_slope_5d",
    "tds_ppm",
    "flow_rate_lpm",
    "pressure_diff_psi",
    "day",
]


def check_file_exists(path, label):
    """Check that a file exists."""
    if not os.path.exists(path):
        print(f"FAIL: {label} not found at {path}")
        return False
    print(f"OK: {label} found at {path}")
    return True


def check_data_loadable():
    """Check that synthetic data can be loaded."""
    try:
        df = pd.read_csv(DATA_PATH)
        required_cols = {
            "unit_id", "day", "tds_ppm", "flow_rate_lpm",
            "pressure_diff_psi", "days_to_failure", "failed_within_7d",
        }
        missing = required_cols - set(df.columns)
        if missing:
            print(f"FAIL: Data missing columns: {missing}")
            return False
        print(f"OK: Data loadable — {len(df)} rows, {df['unit_id'].nunique()} units")
        return True
    except Exception as e:
        print(f"FAIL: Could not load data: {e}")
        return False


def check_model_loadable():
    """Check that the trained model can be loaded."""
    try:
        model = joblib.load(MODEL_PATH)
        print(f"OK: Model loadable — {type(model).__name__}")
        return model
    except Exception as e:
        print(f"FAIL: Could not load model: {e}")
        return None


def check_metrics_loadable():
    """Check that training metrics can be loaded."""
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        required_keys = {
            "test_accuracy", "test_precision", "test_recall",
            "test_f1", "test_false_negative_rate", "threshold",
        }
        missing = required_keys - set(metrics.keys())
        if missing:
            print(f"FAIL: Metrics missing keys: {missing}")
            return False
        print(
            f"OK: Metrics loadable — accuracy={metrics['test_accuracy']:.4f}, "
            f"recall={metrics['test_recall']:.4f}, "
            f"FNR={metrics['test_false_negative_rate']:.4f}"
        )
        return metrics
    except Exception as e:
        print(f"FAIL: Could not load metrics: {e}")
        return None


def main():
    print("=" * 60)
    print("AtomQuest FilterGuard — Health Check")
    print("=" * 60)

    all_ok = True

    # Check files exist
    all_ok &= check_file_exists(DATA_PATH, "Synthetic data CSV")
    all_ok &= check_file_exists(MODEL_PATH, "Trained model pickle")
    all_ok &= check_file_exists(METRICS_PATH, "Training metrics JSON")

    # Check data loadable
    all_ok &= check_data_loadable()

    # Check model loadable
    model = check_model_loadable()
    all_ok &= model is not None

    # Check metrics loadable
    metrics = check_metrics_loadable()
    all_ok &= metrics is not None

    print("=" * 60)
    if all_ok:
        print("All checks passed. Project is ready to run.")
        print("  Next: streamlit run app/dashboard.py")
    else:
        print("Some checks failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
