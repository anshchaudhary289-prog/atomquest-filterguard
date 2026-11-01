# AtomQuest FilterGuard

**Team:** Code&Design (Ansh Chaudhary, Ayush Gupta)
**Track 2 — Water Purifier** | **Area:** Prognostics of filter failure using data analytics

## What FilterGuard does

FilterGuard is a **simulated software-only prototype** that estimates the **probability of filter failure within seven days** using only the purifier's existing TDS / flow-rate / pressure-differential telemetry — no added hardware, no GPU, no cloud, no paid APIs.

**All data is synthetic** (200 virtual purifier units, 120–180 day lifecycles). All results are simulated on synthetic data, not real purifier field results. No real purifier hardware or field data was used.

## Actual evaluation results (corrected labels, test set)

| Metric | Value |
|---|---|
| Test-set accuracy | **94.00%** |
| Precision | 45.78% |
| Recall | 71.25% |
| F1-score | 55.75% |
| False-negative rate (missed failures) | 28.75% |
| False-positive rate | 4.73% |
| Test AUC | 0.96 |
| Confusion matrix (tn, fp, fn, tp) | (5441, 270, 92, 228) |
| Train / test units | 128 / 32 / 40 (train/val/test, stratified by unit) |
| Train / test rows | 19,256 / 4,848 / 6,031 |
| Classification threshold | 0.30 (selected on validation set, not test set) |
| Model | GradientBoostingClassifier (best of 3 compared) |

**Important limitations:**
- Accuracy (94.00%) is inflated by severe class imbalance — 94.7% of rows are negatives.
- The false-negative rate is 28.75% — the model still misses about 3 out of 10 actual failures.
- Recall (71.25%) and precision (45.78%) indicate the model is not yet reliable enough for real-world deployment.
- **100 ppm is this experiment's chosen synthetic TDS threshold, not a universal filter-failure or drinking-water safety standard.**
- All results are simulated on synthetic data. Do not interpret as real-world purifier performance.

## Problem

Current water purifiers alert the user only after the filter has already failed (TDS crosses the failure threshold) — 0 days of warning — forcing premature or emergency cartridge replacement.

## Solution (simulated prototype)

FilterGuard predicts the probability of filter failure within 7 days from current and historical TDS, flow-rate, and pressure-differential readings. It does **not** predict an exact number of days until failure.

## Approach

The pipeline is built in 5 modules, in order:

1. **`data/generate_synthetic_data.py`** — Simulates 200 virtual purifier units over 120–180 day filter lifecycles. Each unit's TDS degrades upward on a logistic curve toward a ~150 ppm ceiling (+σ≈2 noise), flow rate declines and pressure rises as the filter clogs (+ noise). Each unit's degradation rate is randomized ±20% so curves differ. Ground-truth labels: `days_to_failure` (days back from the day TDS first crosses 100 ppm; 0 on failure day, negative after) and `failed_within_7d` (1 if 0 ≤ days_to_failure ≤ 7; post-failure days are NOT positives). Output: `data/synthetic_filter_data.csv`.

2. **`model/train_model.py`** — Engineers trailing 5-day slope features (`tds_slope_5d`, `flow_slope_5d`, `pressure_slope_5d`) plus current readings and filter age. Splits data 64/16/20 train/validation/test at the unit level (no unit's data leaks across split). Compares GradientBoosting, RandomForest, and LogisticRegression. Selects probability threshold on validation set (maximizing F1). Trains GradientBoostingClassifier with class-weight balancing for the 17.8:1 class imbalance. Saves `model/filterguard_model.pkl` via joblib and prints metrics.

3. **`app/dashboard.py`** — Streamlit demo dashboard (simulated): unit selector, live Plotly trend charts with a "today" marker, prediction panel (model probability of failure within 7 days from current/historical readings only — no future-day scan), retrospective reactive-vs-FilterGuard comparison banner (labeled as retrospective), playback control (slider + prev/next/reset + auto-simulate with speed control), business impact panel (illustrative scenario only), and simulated fail-safe panel.

4. **Business Impact Panel** (in `app/dashboard.py`) — Early-warning advantage, illustrative cost impact scenario (clearly labeled, assumptions visible, input controls affect the calculation), and scalability framing (hypothetical only — no field deployment evidence).

5. **Simulated Fail-Safe Panel** (in `app/dashboard.py`) — When the model's current probability of failure within 7 days ≥ 30%, shows "⚠ FAIL-SAFE TRIGGERED (Simulated): auto flow-restriction + replace-now alert," with an explicit disclaimer that physical actuation is the committed Round 3 task.

## Tech stack

Python 3.11+, pandas, numpy, scikit-learn, streamlit, plotly, joblib. pytest for testing. No GPU, no paid APIs, no cloud spend.

## How to run

```bash
# 1. Create / activate the virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for running tests

# 3. Generate synthetic data (Module 1)
python data/generate_synthetic_data.py

# 4. Train the model (Module 2) — reports test metrics to console
python model/train_model.py

# 5. Run tests
pytest tests/

# 6. Launch the dashboard (Modules 3–5)
streamlit run app/dashboard.py
```

## Round 2 note (software-only, simulated)

A physical TDS-sensor + relay testbed was evaluated but components could not be sourced within the submission window. The team deliberately shipped a complete, validated software pipeline — synthetic data generation, a trained model (94.00% test accuracy, 28.75% false-negative rate), a live interactive dashboard, and complete fail-safe decision logic — over a rushed or incomplete hardware integration. Physical sensor/actuator integration is the committed first task for Round 3.

**All results are simulated on synthetic data. No real purifier hardware or field data was used. The 100 ppm threshold is this experiment's chosen synthetic threshold, not a universal standard.**

## Files

```
atomquest-filterguard/
├── data/
│   ├── generate_synthetic_data.py      (Module 1)
│   └── synthetic_filter_data.csv       (generated output, synthetic data)
├── model/
│   ├── train_model.py                  (Module 2)
│   ├── filterguard_model.pkl           (trained artifact, synthetic data)
│   └── training_metrics.json           (actual metrics)
├── app/
│   └── dashboard.py                    (Modules 3, 4, 5 — simulated prototype)
├── tests/
│   └── test_data_generation.py         (automated tests for Module 1)
├── requirements.txt
├── requirements-dev.txt                (test dependencies)
├── README.md
├── PROGRESS_REPORT_DRAFT.md
└── BUILD_LOG.md
```
