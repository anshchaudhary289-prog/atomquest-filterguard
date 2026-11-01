# AtomQuest FilterGuard

**Team:** Code&Design (Ansh Chaudhary, Ayush Gupta)
**Track 2 — Water Purifier** | **Area:** Prognostics of filter failure using data analytics

## Claim

> Predict filter failure 5–7 days in advance at ≥85% accuracy, vs. reactive (0-day) threshold alerts in current purifiers.

## Achieved results (actual, from the trained model on corrected labels)

| Metric | Value |
|---|---|
| Test-set accuracy | **95.14%** |
| Precision | 58.28% |
| Recall | 29.69% |
| F1-score | 39.34% |
| False-negative rate (missed failures) | 70.31% |
| False-positive rate | 1.19% |
| Confusion matrix (tn, fp, fn, tp) | (5643, 68, 225, 95) |
| Train / test units | 160 / 40 (stratified, no unit leakage) |
| Train / test rows | 24,104 / 6,031 |

**Important:** Accuracy (95.14%) is inflated by severe class imbalance — 94.7% of rows are negatives. The false-negative rate is 70.31%, meaning the model misses about 7 out of 10 actual failures. This is a known limitation of the current prototype and would need to be addressed before any real deployment. Results below are **simulated on synthetic data**, not real purifier field results.

**Simulated advance warning:** For the 190 of 200 units where the model issued an alert, the average advance warning vs. the reactive 100 ppm threshold was 7.3 days (range: -5 to 21 days). Negative values indicate the model alerted after the reactive threshold for some units. 100 ppm is this experiment's chosen synthetic threshold, not a universal filter-failure or drinking-water safety standard.

## Problem

Current water purifiers alert the user only after the filter has already failed (TDS crosses the failure threshold) — 0 days of warning — forcing premature or emergency cartridge replacement.

## Solution (simulated prototype)

FilterGuard is a **simulated software-only prototype** that predicts the probability of filter failure within 7 days using only the purifier's existing TDS / flow-rate / pressure-differential telemetry — no added hardware, no GPU, no cloud, no paid APIs.

**Critical limitation:** All data is synthetic (200 virtual purifier units, 120–180 day lifecycles). All results are simulated. No real purifier hardware or field data was used. The model's recall (29.69%) and false-negative rate (70.31%) indicate it is not yet reliable enough for real-world deployment.

## Approach

The pipeline is built in 5 modules, in order:

1. **`data/generate_synthetic_data.py`** — Simulates 200 virtual purifier units over 120–180 day filter lifecycles. Each unit's TDS degrades upward on a logistic curve toward a ~150 ppm ceiling (+σ≈2 noise), flow rate declines and pressure rises as the filter clogs (+ noise). Each unit's degradation rate is randomized ±20% so curves differ. Ground-truth labels: `days_to_failure` (days back from the day TDS first crosses 100 ppm; 0 on failure day, negative after) and `failed_within_7d` (1 if 0 ≤ days_to_failure ≤ 7; post-failure days are NOT positives). Output: `data/synthetic_filter_data.csv`.

2. **`model/train_model.py`** — Engineers trailing 5-day slope features (`tds_slope_5d`, `flow_slope_5d`, `pressure_slope_5d`) plus current readings and filter age. Trains a `GradientBoostingClassifier` on an 80/20 split made **at the unit level** (stratified, so no unit's data leaks across the split). Saves `model/filterguard_model.pkl` via joblib and prints accuracy, precision, recall, F1, confusion matrix, and false-negative rate.

3. **`app/dashboard.py`** — Streamlit demo dashboard (simulated): unit selector, live Plotly trend charts with a "today" marker, prediction panel (model probability of failure within 7 days from current/historical readings only — no future-day scan, no exact days-to-failure claim), reactive-vs-FilterGuard comparison banner (simulated advance-warning days gained), playback control (slider + next/prev + auto-simulate), business impact panel (illustrative scenario only), and simulated fail-safe panel.

4. **Business Impact Panel** (in `app/dashboard.py`) — Early-warning advantage, illustrative cost impact scenario (clearly labeled, assumptions visible, input controls affect the calculation), and scalability framing (hypothetical only — no field deployment evidence).

5. **Simulated Fail-Safe Panel** (in `app/dashboard.py`) — When the model's current probability of failure within 7 days ≥ 50%, shows "⚠ FAIL-SAFE TRIGGERED (Simulated): auto flow-restriction + replace-now alert," with an explicit disclaimer that physical actuation is the committed Round 3 task.

## Tech stack

Python 3.11+, pandas, numpy, scikit-learn, streamlit, plotly, joblib. No GPU, no paid APIs, no cloud spend.

## How to run

```bash
# 1. Create / activate the virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data (Module 1)
python data/generate_synthetic_data.py

# 4. Train the model (Module 2) — reports test accuracy to console
python model/train_model.py

# 5. Launch the dashboard (Modules 3–5)
streamlit run app/dashboard.py
```

## Round 2 note (software-only, simulated)

A physical TDS-sensor + relay testbed was evaluated but components could not be sourced within the submission window. The team deliberately shipped a complete, validated software pipeline — synthetic data generation, a trained model (95.14% test accuracy, but 70.31% false-negative rate), a live interactive dashboard, and complete fail-safe decision logic — over a rushed or incomplete hardware integration. Physical sensor/actuator integration is the committed first task for Round 3.

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
├── requirements.txt
├── README.md
├── PROGRESS_REPORT_DRAFT.md
└── BUILD_LOG.md
```
