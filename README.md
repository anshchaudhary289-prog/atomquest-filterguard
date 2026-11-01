# AtomQuest FilterGuard

**Team:** Code&Design (Ansh Chaudhary, Ayush Gupta)
**Track 2 — Water Purifier** | **Area:** Prognostics of filter failure using data analytics

## Claim

> Predict filter failure **5–7 days in advance at ≥85% accuracy**, vs. reactive (0-day) threshold alerts in current purifiers.

## Achieved results (actual, from the trained model)

| Metric | Value |
|---|---|
| Test-set accuracy | **97.39%** |
| Precision | 96.92% |
| Recall | 95.19% |
| F1-score | 96.05% |
| False-negative rate (missed failures) | 4.81% |
| False-positive rate | 1.51% |
| Confusion matrix (tn, fp, fn, tp) | (3968, 61, 97, 1920) |
| Train / test units | 160 / 40 (stratified, no unit leakage) |
| Train / test rows | 24,089 / 6,046 |

**Average advance warning gained:** ~5–7 days earlier than the reactive TDS-threshold baseline (computed per unit from the model's real predictions: reactive alert day − FilterGuard alert day).

## Problem

Current purifiers alert only *after* the filter has already failed — the TDS threshold is crossed and you get 0 days of warning, forcing premature or emergency cartridge replacement. FilterGuard predicts filter failure 5–7 days ahead using only the purifier's existing TDS, flow-rate, and pressure-differential telemetry — no added hardware, no GPU, no cloud, no paid APIs.

## Approach

The pipeline is built in 5 modules, in order:

1. **`data/generate_synthetic_data.py`** — Simulates 200 virtual purifier units over 120–180 day filter lifecycles. Each unit's TDS degrades upward on a logistic curve toward a ~150 ppm ceiling (+σ≈2 noise), flow rate declines and pressure rises as the filter clogs (+ noise). Each unit's degradation rate is randomized ±20% so curves differ. Ground-truth labels: `days_to_failure` (days back from the day TDS first crosses 100 ppm) and `failed_within_7d`. Output: `data/synthetic_filter_data.csv`.
2. **`model/train_model.py`** — Engineers trailing 5-day slope features (`tds_slope_5d`, `flow_slope_5d`, `pressure_slope_5d`) plus current readings and filter age. Trains a `GradientBoostingClassifier` on an 80/20 split made **at the unit level** (stratified, so no unit's data leaks across the split). Saves `model/filterguard_model.pkl` via joblib and prints accuracy, precision, recall, F1, confusion matrix, and false-negative rate.
3. **`app/dashboard.py`** — Streamlit demo dashboard: unit selector, live Plotly trend charts with a "today" marker, prediction panel ("Filter failure predicted in X days" with confidence), a reactive-vs-FilterGuard comparison banner showing the advance-warning days gained, and playback control (slider + next/prev + auto-simulate).
4. **Business Impact Panel** (in `app/dashboard.py`) — Early-warning advantage, projected cost impact (avoided premature/emergency cartridge replacements), and scalability framing (near-zero marginal cost across the IoT fleet), all traced to the model's real metrics.
5. **Simulated Fail-Safe Panel** (in `app/dashboard.py`) — When predicted days-to-failure ≤ 2, shows "⚠ FAIL-SAFE TRIGGERED (Simulated): auto flow-restriction + replace-now alert," with an explicit disclaimer that physical actuation is the committed Round 3 task.

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

## Round 2 note (software-only)

A physical TDS-sensor + relay testbed was evaluated but components could not be sourced within the submission window. The team deliberately shipped a complete, validated software pipeline — synthetic data generation, a trained model (97.39% test accuracy), a live interactive dashboard, and complete fail-safe decision logic — over a rushed or incomplete hardware integration. Physical sensor/actuator integration is the committed first task for Round 3.

## Files

```
atomquest-filterguard/
├── data/
│   ├── generate_synthetic_data.py      (Module 1)
│   └── synthetic_filter_data.csv       (generated output)
├── model/
│   ├── train_model.py                  (Module 2)
│   ├── filterguard_model.pkl           (trained artifact)
│   └── training_metrics.json           (actual metrics)
├── app/
│   └── dashboard.py                    (Modules 3, 4, 5)
├── requirements.txt
└── README.md
```
