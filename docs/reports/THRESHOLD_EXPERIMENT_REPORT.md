# Threshold Experiment Report — AtomQuest FilterGuard

**Date:** 2026-11-04  
**Status:** Experiment 1 — Threshold-only evaluation (read-only)  
**Repository:** `anshchaudhary289-prog/atomquest-filterguard`  
**Targets:** Recall >= 80%, False-positive rate <= 15%

---

## 1. Baseline Results (Threshold 0.30)

The baseline threshold of 0.30 was selected on the validation set by maximizing F1-score. It is the currently deployed threshold in `model/training_metrics.json`.

Evaluated on the held-out test set (40 units / 6,031 rows) using the existing `GradientBoostingClassifier` model (`model/filterguard_model.pkl`) with no retraining:

| Metric | Value |
|---|---|
| Accuracy | 94.00% |
| Precision | 45.78% |
| Recall | 71.25% |
| F1-score | 55.75% |
| False-negative rate (FNR) | 28.75% |
| False-positive rate (FPR) | 4.73% |
| Confusion matrix (tn, fp, fn, tp) | (5,441, 270, 92, 228) |
| Threshold | 0.30 |

**Target assessment:** Recall 71.25% **FAILS** the >= 80% target. FPR 4.73% **PASSES** the <= 15% target.

---

## 2. Experimental Results (Threshold 0.20)

The threshold 0.20 was evaluated on the **same held-out test predictions** — no retraining, no new model, no data regeneration. The only change is the decision threshold applied to `model.predict_proba(X_test)[:, 1]`.

| Metric | Value |
|---|---|
| Accuracy | 92.62% |
| Precision | 40.60% |
| Recall | 84.38% |
| F1-score | 54.82% |
| False-negative rate (FNR) | 15.62% |
| False-positive rate (FPR) | 6.92% |
| Confusion matrix (tn, fp, fn, tp) | (5,316, 395, 50, 270) |
| Threshold | 0.20 |

**Target assessment:** Recall 84.38% **PASSES** the >= 80% target. FPR 6.92% **PASSES** the <= 15% target.

---

## 3. Validation Results Supporting the Threshold Choice

The threshold 0.20 was **not** tuned on the test set. It was one of nine candidate thresholds (0.10–0.90) evaluated on the validation set (32 units / 4,848 rows) during the original training run, as recorded in `model/training_metrics.json`.

| Threshold | Validation Recall | Validation FPR | Validation F1 | Validation Precision |
|---|---|---|---|---|
| 0.30 (baseline) | 74.61% | 4.36% | 59.04% | 48.85% |
| 0.20 (experimental) | 83.20% | 6.40% | 55.83% | 42.01% |

The validation results at 0.20 (83.20% recall, 6.40% FPR) are consistent with the test-set results (84.38% recall, 6.92% FPR), confirming the threshold choice generalizes and was not overfit to the test set.

---

## 4. Test-Set Comparison (0.30 vs 0.20)

| Metric | Threshold 0.30 | Threshold 0.20 | Change |
|---|---|---|---|
| Accuracy | 94.00% | 92.62% | -1.38 pp |
| Precision | 45.78% | 40.60% | -5.18 pp |
| Recall | 71.25% | 84.38% | **+13.12 pp** |
| F1-score | 55.75% | 54.82% | -0.93 pp |
| FNR | 28.75% | 15.62% | **-13.12 pp** |
| FPR | 4.73% | 6.92% | **+2.19 pp** |
| TN | 5,441 | 5,316 | -125 |
| FP | 270 | 395 | +125 |
| FN | 92 | 50 | -42 |
| TP | 228 | 270 | +42 |

### Interpretation

- **Recall increases by 13.12 percentage points** (71.25% → 84.38%), crossing the 80% target. This means 42 additional true failures are correctly flagged within the 7-day window that were previously missed.
- **FPR increases by 2.19 percentage points** (4.73% → 6.92%), remaining well below the 15% ceiling. This means 125 additional non-failure rows are flagged as potential failures (more false alarms / unnecessary replacements).
- **Precision drops by 5.18 points** (45.78% → 40.60%): roughly 4 out of 10 positive predictions are true failures at 0.20, versus roughly 4.6 out of 10 at 0.30.
- **F1 decreases slightly** (55.75% → 54.82%) because the precision loss outweighs the recall gain under equal weighting.
- **Accuracy drops by 1.38 points** (94.00% → 92.62%) due to the additional false positives.

---

## 5. Trade-offs: False Alarms vs. Missed Failures

| | Threshold 0.30 | Threshold 0.20 | Impact of switching |
|---|---|---|---|
| Missed failures (FN) | 92 | 50 | **42 fewer missed failures** |
| False alarms (FP) | 270 | 395 | **125 more false alarms** |
| True failures caught (TP) | 228 | 270 | **42 more failures caught** |

**For a safety-critical system (filter failure in water purifiers):**
- Missing a failure (FN) means a customer loses safe drinking water unexpectedly — high cost.
- A false alarm (FP) means an unnecessary filter replacement — lower cost (cartridge cost ₹400–600).

Given that the safety-relevant error (false negative) is reduced by 45.7% (92 → 50) at the cost of 46.3% more false alarms (270 → 395), threshold 0.20 is defensible for this application. The FPR of 6.92% remains comfortably below the 15% ceiling.

---

## 6. Recommendation

**Adopt threshold 0.20 as the operating threshold.**

### Rationale
- Threshold 0.20 meets **both** Round 2 targets on the held-out test set: recall 84.38% (>= 80%) and FPR 6.92% (<= 15%).
- The threshold was selected based on validation-set performance (83.20% recall, 6.40% FPR at 0.20), not test-set tuning — test results confirm the validation estimate.
- No retraining, no model modification, no data regeneration required — only a configuration change to the decision threshold applied to existing model probabilities.
- The trade-off (more false alarms, fewer missed failures) is appropriate for a safety-critical filter-failure prediction system where missed failures are costlier than unnecessary replacements.

### What to implement (pending approval)
1. Update `FAIL_PROB_THRESHOLD` in `app/dashboard.py` from `0.50` to `0.20` (note: dashboard currently uses 0.50 for fail-safe, while training used 0.30 — this is a pre-existing inconsistency that threshold adoption should resolve).
2. Update `training_metrics.json` or record the operating threshold separately.
3. Update `README.md` and `PROGRESS_REPORT_DRAFT.md` to reflect the adopted threshold and its metrics.
4. Re-run the dashboard and end-to-end verification.

### What NOT to do
- Do NOT retrain the model or regenerate synthetic data for this experiment.
- Do NOT tune further on the test set.
- Do NOT claim real-world performance — all results are on synthetic data.

---

## 7. File Integrity Verification

No project source files, model artifacts, or dataset files were modified during this experiment. The only untracked file in the repository is `ML_IMPROVEMENT_AUDIT.md` (from the preceding audit phase).

| File | Status |
|---|---|
| `model/filterguard_model.pkl` | Unchanged |
| `data/synthetic_filter_data.csv` | Unchanged |
| `model/train_model.py` | Unchanged |
| `app/dashboard.py` | Unchanged |
| `model/training_metrics.json` | Unchanged (read-only reference) |
| `ML_IMPROVEMENT_AUDIT.md` | Untracked (audit report, pre-existing) |

---

**Next step:** Awaiting approval before implementing the threshold change in `app/dashboard.py` and updating documentation.
