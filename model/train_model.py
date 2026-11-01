"""Module 2 — Model training for AtomQuest FilterGuard.

Feature engineering (rolling over last 5 days per unit):
  tds_slope_5d, flow_slope_5d, pressure_slope_5d  — linear regression slope
                                                      over trailing 5-day window
  tds_ppm, flow_rate_lpm, pressure_diff_psi       — current readings
  day                                             — age of filter

Target: failed_within_7d (binary classification)

Model: GradientBoostingClassifier (scikit-learn)
Split: 80/20 train/test, split at unit level (no unit's data leaks across split),
       stratified by unit-level positive-rate quartile.

Saves trained model to model/filterguard_model.pkl via joblib and prints metrics.
Acceptance criterion: test-set accuracy >= 0.85.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_filter_data.csv")
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(MODEL_DIR, "filterguard_model.pkl")

FEATURE_COLS = [
    "tds_slope_5d",
    "flow_slope_5d",
    "pressure_slope_5d",
    "tds_ppm",
    "flow_rate_lpm",
    "pressure_diff_psi",
    "day",
]
TARGET_COL = "failed_within_7d"
WINDOW = 5
TEST_SIZE = 0.2
RANDOM_SEED = 42


def _slope(series):
    """Linear regression slope of a short series; x = 0..n-1 within the window."""
    if len(series) < 2:
        return np.nan
    x = np.arange(len(series), dtype=float)
    return float(np.polyfit(x, series.values, 1)[0])


def add_slope_features(data):
    """Add trailing 5-day slope features per unit.

    Uses groupby.transform so the result aligns to the original row index and
    no columns (including unit_id) are dropped.
    """
    data = data.sort_values(["unit_id", "day"]).reset_index(drop=True)
    slope_map = {
        "tds_ppm": "tds_slope_5d",
        "flow_rate_lpm": "flow_slope_5d",
        "pressure_diff_psi": "pressure_slope_5d",
    }
    for col, slope_col in slope_map.items():
        data[slope_col] = data.groupby("unit_id")[col].transform(
            lambda s: s.rolling(window=WINDOW, min_periods=2).apply(_slope, raw=False)
        )
    return data


def main():
    data = pd.read_csv(DATA_PATH)
    data = data.sort_values(["unit_id", "day"]).reset_index(drop=True)

    data = add_slope_features(data)

    # Drop rows where a 5-point slope could not be computed (start of each unit).
    data = data.dropna(subset=[c for c in FEATURE_COLS if c.endswith("_slope_5d")]).copy()

    # --- 80/20 split at unit level, stratified by unit positive-rate quartile ---
    units = data[["unit_id"]].drop_duplicates().sort_values("unit_id")
    unit_pos_rate = data.groupby("unit_id")[TARGET_COL].mean().reindex(units["unit_id"])
    try:
        stratify = pd.qcut(unit_pos_rate, q=4, labels=False, duplicates="drop")
    except ValueError:
        stratify = None

    train_units, test_units = train_test_split(
        units["unit_id"].values,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    train = data[data["unit_id"].isin(train_units)].copy()
    test = data[data["unit_id"].isin(test_units)].copy()

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET_COL]

    print(f"Train units: {len(train_units)}, test units: {len(test_units)}")
    print(f"Train rows: {len(train)}, test rows: {len(test)}")
    print(f"Train positive rate: {y_train.mean():.3f}, test positive rate: {y_test.mean():.3f}")

    # --- Model ---
    model = GradientBoostingClassifier(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n===== Test-set evaluation =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Confusion matrix (tn, fp, fn, tp): ({tn}, {fp}, {fn}, {tp})")
    print(f"False-negative rate (missed failures): {fnr:.4f}")
    print(f"False-positive rate                  : {fpr:.4f}")
    print("===============================\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Persist metrics for the dashboard / README to reference.
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_negative_rate": float(fnr),
        "false_positive_rate": float(fpr),
        "train_units": int(len(train_units)),
        "test_units": int(len(test_units)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    import json
    metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
