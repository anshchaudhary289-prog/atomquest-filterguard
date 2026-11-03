# ML Improvement Audit — AtomQuest FilterGuard

**Date:** 2026-11-04  
**Status:** Phase 1 — Audit Only (no code, data, model, or metric changes made)  
**Repository:** `anshchaudhary289-prog/atomquest-filterguard`  
**Target:** Recall >= 80%, False-positive rate <= 15%

---

## 1. Current Model and Dataset Analysis

### Dataset (Module 1)
- **Source:** `data/generate_synthetic_data.py` — fully synthetic, 200 virtual purifier units
- **Lifecycle:** 120–180 days per unit (randomized, seed=42)
- **Sensors:** TDS (ppm), flow_rate_lpm (L/min), pressure_diff_psi (psi)
- **Degradation model:** TDS follows exponential curve toward 150 ppm ceiling (`TDS_CEILING - (TDS_CEILING - base_tds) * exp(-r * days)`), flow declines, pressure rises as clogging progresses
- **Failure threshold:** TDS >= 100 ppm (synthetic experiment threshold, not a universal standard)
- **Target label (`failed_within_7d`):** 1 when `0 <= days_to_failure <= 7` (failure day + 7 preceding days); post-failure days are NOT positives
- **Total rows:** ~30,335 across 200 units
- **Positive rate:** ~5.3% (1,600 positives / 30,335 rows) — severely imbalanced

### Model (Module 2)
- **Algorithm:** GradientBoostingClassifier (scikit-learn), default hyperparameters, `random_state=42`
- **Note:** GradientBoosting does NOT use `class_weight` (unlike RandomForest and LogisticRegression which use `class_weight="balanced"`)
- **Alternative models compared:** RandomForestClassifier (`class_weight="balanced"`), LogisticRegression (`class_weight="balanced"`, with `StandardScaler` pipeline)
- **Features (7):** `tds_slope_5d`, `flow_slope_5d`, `pressure_slope_5d` (trailing 5-day linear regression slopes), `tds_ppm`, `flow_rate_lpm`, `pressure_diff_psi`, `day`
- **Split:** Unit-level 64/16/20 train/validation/test (128 / 32 / 40 units), stratified by unit-level positive-rate quartile

### Current Test Metrics (held-out test set, 40 units / 6,031 rows)
| Metric | Value | Target |
|---|---|---|
| Accuracy | 94.00% | — |
| Precision | 45.78% | — |
| Recall | 71.25% | **>= 80%** |
| F1-score | 55.75% | — |
| False-negative rate | 28.75% | — |
| False-positive rate | 4.73% | **<= 15%** |
| ROC-AUC | 0.9635 | — |
| Confusion matrix (tn, fp, fn, tp) | (5441, 270, 92, 228) | — |
| Threshold | 0.30 (validation-selected) | — |

---

## 2. Train/Validation/Test Split Verification

**Status: VERIFIED — No unit leakage.**

The code performs two nested `train_test_split` calls at the **unit level**:

1. **First split:** 200 units → 160 trainval / 40 test (stratified by unit positive-rate quartile)
2. **Second split:** 160 trainval units → 128 train / 32 val (stratified again)

No unit appears in more than one split. Unit IDs are disjoint across train/validation/test. The stratification uses `pd.qcut(unit_pos_rate, q=4)`, which ensures each split has a similar distribution of unit-level failure rates.

**Row counts:** Train 19,256 / Validation 4,848 / Test 6,031 (total 30,135 after dropping rows without slope features at the start of each unit's lifecycle).

---

## 3. Data Leakage Checks

**Status: VERIFIED — No leakage detected.**

| Potential leakage | Finding |
|---|---|
| Unit across splits | None — unit-level split ensures disjoint unit IDs |
| Future rows in features | Slopes use `rolling(window=5)` trailing window only — no future observations |
| Target leakage | `failed_within_7d` is computed from `days_to_failure` which is derived from the full lifecycle, but features only use current + trailing 5 days — no future sensor data used for prediction |
| Test set in threshold selection | Threshold selected on validation set only — test set never touched during model/threshold choice |
| Data preprocessing fit on test | `StandardScaler` is inside a `Pipeline` fit only on training data |

One minor observation: `pd.qcut` with `duplicates="drop"` may produce fewer than 4 strata if many units share the same positive rate, but this does not cause leakage — it only affects stratification granularity.

---

## 4. Class Imbalance Analysis

**Severity: High (17.8:1 negative:positive ratio)**

| Split | Positive rate | Negative:Positive ratio |
|---|---|---|
| Train | 5.32% | 17.8:1 |
| Validation | 5.28% | ~18:1 |
| Test | 5.31% | ~18:1 |

### Impact
- Accuracy (94.00%) is heavily inflated by the majority class. A trivial "always predict negative" classifier would achieve ~94.7% accuracy.
- GradientBoostingClassifier is trained **without** `class_weight` or `sample_weight`, unlike RandomForest and LogisticRegression which both use `class_weight="balanced"`. This means the gradient boosting model treats every misclassification equally, despite positives being ~5% of data.
- With only ~8 positive rows per unit (failure day + 7 preceding days), the model has limited positive examples to learn the pre-failure signature.

---

## 5. Validation Threshold Comparison

Thresholds evaluated on validation set (32 units / 4,848 rows):

| Threshold | Accuracy | Precision | Recall | F1 | FNR | FPR |
|---|---|---|---|---|---|---|
| 0.10 | 91.07% | 36.28% | 91.41% | 51.94% | 8.59% | 8.95% |
| 0.20 | 93.05% | 42.01% | **83.20%** | 55.83% | 16.80% | 6.40% |
| **0.30 (selected)** | 94.53% | 48.85% | 74.61% | **59.04%** | 25.39% | 4.36% |
| 0.40 | 95.26% | 55.08% | 55.08% | 55.08% | 44.92% | 2.50% |
| 0.50 | 95.46% | 62.16% | 35.94% | 45.54% | 64.06% | 1.22% |

### Key observations:
- **Threshold 0.20** achieves **83.20% recall** on validation with **6.40% FPR** — both targets met (recall >= 80%, FPR <= 15%).
- **Threshold 0.30** (currently selected by maximizing F1) achieves 74.61% recall, 4.36% FPR — recall falls short of 80%.
- The F1-maximizing threshold (0.30) systematically under-recalls compared to the recall target because F1 equally weights precision and recall, and precision drops sharply at lower thresholds due to class imbalance.

---

## 6. Potential Reasons for Current 71.25% Recall

1. **Threshold selection optimizes F1, not recall.** The code selects the threshold maximizing validation F1-score (0.30), which balances precision and recall. At this threshold, recall is 74.61% on validation and 71.25% on test. A lower threshold (e.g., 0.20) would raise recall above 80% but reduce precision and F1.

2. **GradientBoostingClassifier lacks class balancing.** Unlike RandomForest and LogisticRegression (both use `class_weight="balanced"`), GradientBoosting is trained without any class-imbalance correction. With a 17.8:1 ratio, the model under-weights the minority failure class during gradient boosting.

3. **Limited positive examples per unit.** Only 8 rows per unit are positive (failure day + 7 days before). With 200 units, that's ~1,600 positive rows total — but many are temporally adjacent and highly correlated, so the effective independent signal is weaker.

4. **Features may not capture pre-failure dynamics strongly.** The slope features use only a 5-day window. The failure signature (rising TDS, falling flow, rising pressure) develops over days, and a 5-day window may be too short to distinguish early clogging from normal variation, especially with sigma=2 noise on TDS.

5. **Test set variance.** With only 40 test units and 92 false negatives, the test recall has sampling uncertainty. A single unit's failure pattern can swing recall by several percentage points.

---

## 7. Three Recommended Improvement Experiments

### Experiment 1: Threshold-only adjustment (lowest risk, fastest)
**Action:** Change the decision threshold from 0.30 to 0.20 (or scan a finer grid around 0.15–0.25) applied on the validation set, then evaluate on test.

**Rationale:** Validation data already shows threshold 0.20 achieves 83.20% recall with 6.40% FPR — both targets met. This requires no retraining, no feature changes, and no new data.

**Expected outcome:** Recall likely rises to ~80%+, FPR likely stays under 10%. Precision will drop (likely to ~40–45%), and F1 may decrease, but the recall/FPR targets are met.

**Risk:** Lower precision means more false alarms (unnecessary filter replacements). For a safety-critical system, this may be acceptable.

---

### Experiment 2: Add class_weight to GradientBoostingClassifier (medium effort)
**Action:** Add `class_weight="balanced"` to `GradientBoostingClassifier` in `model/train_model.py`, retrain, and re-select threshold on validation.

**Rationale:** GradientBoosting is the selected model but is the only one without class balancing. RandomForest and LogisticRegression both use it. Adding `class_weight="balanced"` should increase the model's sensitivity to the minority class, improving recall.

**Expected outcome:** Higher recall at the same threshold, or ability to use a higher threshold (better precision) while maintaining recall. Risk: may increase FPR.

**Risk:** GradientBoosting's `class_weight` support varies by scikit-learn version — must verify compatibility. May overfit the small positive class if not careful.

---

### Experiment 3: Enrich feature set with longer-window and aggregated features (medium-high effort)
**Action:** Add features such as:
- TDS slope over 10-day and 15-day windows (longer trend)
- TDS acceleration (second derivative / change in slope)
- Days since TDS first exceeded 90 ppm (proximity to failure threshold)
- Rolling max/min of TDS over 7 days
- Time-since-failure-proximity features

**Rationale:** The current 5-day slope window may be too short to reliably detect the clogging trend amid noise. Longer-window and trend features should better capture the pre-failure signature.

**Expected outcome:** Better separation between positive and negative classes, potentially improving both recall and precision at the same threshold.

**Risk:** More features increase overfitting risk on 1,600 positives. Requires careful validation. Longer windows reduce the number of usable early-lifecycle rows.

---

## 8. Risks and Expected Trade-offs

| Risk | Description | Mitigation |
|---|---|---|
| **Overfitting to validation** | Repeated threshold/model tuning on validation can overfit the 32 validation units | Keep validation set untouched until final; report test metrics honestly |
| **Precision collapse** | Lowering threshold to boost recall will reduce precision (more false alarms) | Acceptable for safety-critical system; document the trade-off |
| **Test set variance** | Only 40 test units; recall estimate has uncertainty | Report confidence intervals; consider cross-validation across units |
| **Synthetic-to-real gap** | All data is synthetic; real purifier sensors may behave differently | Clearly document this limitation; plan field validation |
| **Class weighting instability** | `class_weight="balanced"` on small positive class can cause unstable gradients | Monitor validation stability; use conservative learning rate |

---

## 9. Recommendation: What to Test First

**Test threshold-only adjustment first (Experiment 1).**

**Reasoning:**
- Zero retraining required — uses existing model artifacts
- Validation evidence already shows threshold 0.20 meets both targets (83.20% recall, 6.40% FPR)
- Fastest path to determining if the targets are achievable with the current model
- If threshold-only works, no code changes to the model are needed — just a configuration change
- If threshold-only fails to meet targets on the test set, proceed to Experiment 2 (class_weight) and then Experiment 3 (feature enrichment)

**Do NOT:** Optimize against the test set, fabricate metrics, or guarantee the 80% recall target will be met on real data.

---

**Next step:** Awaiting your approval before implementing any changes. No files have been modified during this audit.
