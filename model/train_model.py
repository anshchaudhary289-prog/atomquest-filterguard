"""Module 2 — Model training for AtomQuest FilterGuard.

Feature engineering (rolling over last 5 days per unit):
  tds_slope_5d, flow_slope_5d, pressure_slope_5d — linear regression slope
                                                     over trailing 5-day window
  tds_ppm, flow_rate_lpm, pressure_diff_psi       — current readings
  day                                             — age of filter

Target: failed_within_7d (binary classification)

Model: GradientBoostingClassifier (scikit-learn), compared against
RandomForestClassifier and LogisticRegression.

Split: 64/16/20 train/validation/test, split at unit level (no unit's data
leaks across split). Stratified by unit-level positive-rate quartile.

Threshold selection: multiple probability thresholds are evaluated on the
validation set only (never the test set). The threshold that maximizes
validation F1-score is selected and then applied to the test set.

Saves trained model to model/filterguard_model.pkl via joblib.
Saves reproducible evaluation results to model/training_metrics.json.
Prints final metrics to console on training run.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_filter_data.csv")
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(MODEL_DIR, "filterguard_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.json")

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
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # fraction of remaining data after test split
RANDOM_SEED = 42

# Candidate probability thresholds evaluated on validation set
THRESHOLD_CANDIDATES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


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


def evaluate_threshold(y_true, y_prob, threshold):
    """Compute metrics for a given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "threshold": float(threshold),
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
    }


def select_threshold(y_val, y_val_prob, candidates=THRESHOLD_CANDIDATES):
    """Select the probability threshold that maximizes validation F1-score.

    Only the validation set is used — never the test set.
    """
    results = []
    for thresh in candidates:
        m = evaluate_threshold(y_val, y_val_prob, thresh)
        results.append(m)
    # Sort by F1 descending, then by FNR ascending (fewer missed failures)
    results.sort(key=lambda m: (-m["f1"], m["false_negative_rate"]))
    best = results[0]
    return best, results


def main():
    data = pd.read_csv(DATA_PATH)
    data = data.sort_values(["unit_id", "day"]).reset_index(drop=True)

    data = add_slope_features(data)

    # Drop rows where a 5-point slope could not be computed (start of each unit).
    feature_slope_cols = [c for c in FEATURE_COLS if c.endswith("_slope_5d")]
    data = data.dropna(subset=feature_slope_cols).copy()

    # --- Unit-level split: 64% train / 16% validation / 20% test ---
    units = data[["unit_id"]].drop_duplicates().sort_values("unit_id")
    unit_pos_rate = data.groupby("unit_id")[TARGET_COL].mean().reindex(units["unit_id"])
    try:
        stratify = pd.qcut(unit_pos_rate, q=4, labels=False, duplicates="drop")
    except ValueError:
        stratify = None

    # First split: 80% trainval / 20% test
    trainval_units, test_units = train_test_split(
        units["unit_id"].values,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    # Second split: 80% train / 20% val of trainval → 64% / 16% overall
    trainval_data = data[data["unit_id"].isin(trainval_units)]
    trainval_units_unique = trainval_data[["unit_id"]].drop_duplicates().sort_values("unit_id")
    trainval_pos_rate = trainval_data.groupby("unit_id")[TARGET_COL].mean().reindex(
        trainval_units_unique["unit_id"]
    )
    try:
        stratify_v = pd.qcut(trainval_pos_rate, q=4, labels=False, duplicates="drop")
    except ValueError:
        stratify_v = None

    train_units, val_units = train_test_split(
        trainval_units_unique["unit_id"].values,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_v,
    )

    train = data[data["unit_id"].isin(train_units)].copy()
    val = data[data["unit_id"].isin(val_units)].copy()
    test = data[data["unit_id"].isin(test_units)].copy()

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_val = val[FEATURE_COLS]
    y_val = val[TARGET_COL]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET_COL]

    print(f"Train units: {len(train_units)}, val units: {len(val_units)}, test units: {len(test_units)}")
    print(f"Train rows: {len(train)}, val rows: {len(val)}, test rows: {len(test)}")
    print(f"Train positive rate: {y_train.mean():.3f}")
    print(f"Val positive rate:   {y_val.mean():.3f}")
    print(f"Test positive rate:  {y_test.mean():.3f}")

    # --- Class imbalance check ---
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    imbalance_ratio = neg_count / max(pos_count, 1)
    print(f"\nClass imbalance ratio (neg:pos) in training: {imbalance_ratio:.1f}:1")

    # --- Model comparison on validation set ---
    models = {
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, class_weight="balanced"
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                random_state=RANDOM_SEED, class_weight="balanced", max_iter=1000
            )),
        ]),
    }

    print("\n===== Model comparison (validation set) =====")
    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_prob)
        best_metric, all_thresholds = select_threshold(y_val, y_val_prob)
        model_results[name] = {
            "val_auc": float(auc),
            "val_best_threshold": best_metric["threshold"],
            "val_best_f1": best_metric["f1"],
            "val_best_recall": best_metric["recall"],
            "val_best_precision": best_metric["precision"],
            "val_best_fnr": best_metric["false_negative_rate"],
            "all_thresholds": all_thresholds,
        }
        print(
            f"{name:20s}  AUC={auc:.4f}  best_thresh={best_metric['threshold']:.2f}  "
            f"val_F1={best_metric['f1']:.4f}  val_recall={best_metric['recall']:.4f}  "
            f"val_FNR={best_metric['false_negative_rate']:.4f}"
        )

    # --- Select best model by validation F1 ---
    best_model_name = max(model_results, key=lambda n: model_results[n]["val_best_f1"])
    best_model = models[best_model_name]
    best_threshold = model_results[best_model_name]["val_best_threshold"]
    print(f"\nSelected model: {best_model_name} (threshold={best_threshold:.2f})")

    # --- Final evaluation on test set with selected threshold ---
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_threshold(y_test, y_test_prob, best_threshold)
    test_auc = roc_auc_score(y_test, y_test_prob)

    # Precision-recall curve data (for reporting)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_test_prob)

    print("\n===== Test-set evaluation (selected threshold) =====")
    print(f"Model        : {best_model_name}")
    print(f"Threshold    : {best_threshold:.2f}")
    print(f"Test AUC     : {test_auc:.4f}")
    print(f"Accuracy     : {test_metrics['accuracy']:.4f}")
    print(f"Precision    : {test_metrics['precision']:.4f}")
    print(f"Recall       : {test_metrics['recall']:.4f}")
    print(f"F1           : {test_metrics['f1']:.4f}")
    print(f"Confusion matrix (tn, fp, fn, tp): "
          f"({test_metrics['tn']}, {test_metrics['fp']}, {test_metrics['fn']}, {test_metrics['tp']})")
    print(f"False-negative rate (missed failures): {test_metrics['false_negative_rate']:.4f}")
    print(f"False-positive rate                  : {test_metrics['false_positive_rate']:.4f}")
    print("===============================\n")

    # --- Save model ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # --- Save comprehensive metrics ---
    metrics = {
        "model_name": best_model_name,
        "threshold": float(best_threshold),
        "test_auc": float(test_auc),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_tn": int(test_metrics["tn"]),
        "test_fp": int(test_metrics["fp"]),
        "test_fn": int(test_metrics["fn"]),
        "test_tp": int(test_metrics["tp"]),
        "test_false_negative_rate": float(test_metrics["false_negative_rate"]),
        "test_false_positive_rate": float(test_metrics["false_positive_rate"]),
        "train_units": int(len(train_units)),
        "val_units": int(len(val_units)),
        "test_units": int(len(test_units)),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
        "class_imbalance_ratio": float(imbalance_ratio),
        "model_comparison": {
            name: {
                "val_auc": float(r["val_auc"]),
                "val_best_threshold": float(r["val_best_threshold"]),
                "val_best_f1": float(r["val_best_f1"]),
                "val_best_recall": float(r["val_best_recall"]),
                "val_best_precision": float(r["val_best_precision"]),
                "val_best_fnr": float(r["val_best_fnr"]),
            }
            for name, r in model_results.items()
        },
        "threshold_evaluation": model_results[best_model_name]["all_thresholds"],
        "precision_recall_curve": {
            "precision": [float(p) for p in precision_curve],
            "recall": [float(r) for r in recall_curve],
            "thresholds": [float(t) for t in pr_thresholds],
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
