# BUILD_LOG — AtomQuest FilterGuard

Log of development, corrections, testing, and verification. Streamlit runtime
is NOT interacted with directly; only `streamlit run app/dashboard.py` startup
is confirmed via a short headless launch.

---

## 2026-11-02 — Initial module development

Modules 3, 4, 5 built incrementally, each verified (py_compile exit 0 +
streamlit run starts without crashing), each committed & pushed separately:

- `feat(module 3): base dashboard (unit selector, trends, prediction panel,
  comparison banner, playback control)` — commit 40bda27
- `feat(module 4): business impact panel (early-warning advantage, projected
  cost impact, scalability framing)` — commit a01727c
- `feat(module 5): simulated fail-safe panel (trigger at days_to_failure <= 2
  + Round 3 disclaimer)` — commit a6a7eb2

PROGRESS_REPORT_DRAFT.md drafted per SPEC template — commit 6921b59.

---

## 2026-11-02 — Critical Validation & Correction Pass (commit 0117064)

### Issue found: synthetic target labels were wrong

`failed_within_7d = (days_to_failure <= 7)` labeled ALL post-failure days
(negative values) as positive, inflating positives to 32.8% (9,943 rows) and
producing a misleading 97.39% accuracy.

**Old metric (SUPERSEDED):** accuracy 97.39%, precision 96.92%, recall 95.19%,
F1 96.05%, FNR 4.81% — this was an artifact of the buggy labels and is NOT
reproduced or claimed.

### Fix

`failed_within_7d = ((days_to_failure >= 0) & (days_to_failure <= 7))` —
positive only on failure day + 7 days before; post-failure days are NOT
positives. Verified: 1600 positives (5.3%, exactly 200 units × 8 days), 0
positives with days_to_failure < 0 or > 7.

### Corrections made

1. **data/generate_synthetic_data.py** — Fixed label definition. Regenerated
   CSV. Verified label counts and examples before/during/after failure threshold.
2. **model/train_model.py** — Retrained on corrected dataset. Unit-level
   80/20 split preserved. New metrics: accuracy 95.14%, precision 58.28%,
   recall 29.69%, F1 39.34%, FNR 70.31%, FPR 1.19%.
3. **app/dashboard.py** — Removed future-day sensor-reading scan. Prediction
   panel shows probability only. Fail-safe uses current prediction. Business
   impact labeled as illustrative. 100 ppm described as synthetic threshold.
4. **README.md, PROGRESS_REPORT_DRAFT.md** — Updated with corrected metrics.

---

## 2026-11-02 — Test infrastructure added (commit 2d22d19)

- Created `tests/test_data_generation.py` with 12 tests covering:
  - DataFrame columns, lifecycle range, consecutive days, unit_id consistency
  - Reproducibility with fixed seed
  - Label correctness: 0 <= days_to_failure <= 7 only
  - Post-failure days not labeled as positives
  - Full dataset: all 200 units cross threshold, exactly 8 positives per unit
  - No missing values, no duplicate records
  - TDS degrades upward toward failure threshold
- Created `requirements-dev.txt` with pytest dependency
- All 12 tests pass

---

## 2026-11-02 — Model training improvements (commit 07fe588)

- Rewrote `model/train_model.py` with:
  - 64/16/20 train/validation/test split at unit level
  - Model comparison: GradientBoosting vs RandomForest vs LogisticRegression
  - Class imbalance handling (17.8:1 ratio, class_weight='balanced')
  - Threshold selection on validation set (9 candidates: 0.10–0.90)
  - Threshold selected by maximizing validation F1 (never tuned on test set)
  - Classification report, confusion matrix, precision-recall curve data
  - Comprehensive metrics saved to `model/training_metrics.json`

### New model results (commit 07fe588)

Selected model: GradientBoosting, threshold 0.30

| Metric | Value |
|---|---|
| Test accuracy | 94.00% |
| Test precision | 45.78% |
| Test recall | 71.25% |
| Test F1 | 55.75% |
| Test FNR | 28.75% |
| Test FPR | 4.73% |
| Test AUC | 0.96 |

**Key improvement:** Missed failures (FNR) reduced from 70.31% to 28.75%
— the right trade-off for a safety-critical system.

---

## 2026-11-02 — Dashboard reliability improvements (commit 2242f16)

- Added `import time` and controlled auto-play interval (Slow/Normal/Fast)
- Added "Reset" button to return to day 1
- Playback controls: Prev, Next, Reset, Auto-play with speed selection
- No unstable hacks — `time.sleep()` before `st.rerun()` for visible delay

---

## 2026-11-02 — Dashboard correctness fix (commit e4c6510)

- Labeled comparison banner as "retrospective full-lifecycle comparison"
- Added explicit caption: not a real-time prediction
- Updated `FAIL_PROB_THRESHOLD` from 0.50 to 0.30 to match trained model
- Updated FilterGuard alert caption to reference 30% threshold

---

## 2026-11-02 — Documentation fixes (commit fc538bf)

- Updated `README.md` with honest claims:
  - Changed claim from "predicts 5–7 days at ≥85% accuracy" to
    "estimates probability of failure within 7 days"
  - Reported actual metrics: accuracy 94.00%, recall 71.25%, FNR 28.75%
  - Clearly documented synthetic-data limitation
  - Clarified 100 ppm is synthetic threshold, not universal standard
- Added `requirements-dev.txt` installation step to run instructions

---

## Current state

- All 5 modules complete and verified
- `streamlit run app/dashboard.py` starts without errors
- All 12 data generation tests pass
- Model trained with corrected labels and improved threshold selection
- Working tree clean, all commits pushed to origin/main
