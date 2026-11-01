# Progress Report — AtomQuest FilterGuard (Draft)

**File naming convention (per SPEC):** `atomQUEST2026_CodeDesign_MIET_Track2_Report`

---

## 1. Team name, college, track, area of exploration

- **Team:** Code&Design (Ansh Chaudhary, Ayush Gupta) — submitted under `CodeDesign` per filename convention
- **College:** MIET
- **Track:** Track 2 — Water Purifier
- **Area of exploration:** Prognostics of filter failure using data analytics

---

## 2. Problem + solution recap

**Problem:** Current water purifiers alert the user only after the filter has already failed (TDS crosses the failure threshold) — 0 days of warning — forcing premature or emergency cartridge replacement.

**Solution:** FilterGuard is a **simulated software-only prototype** that estimates the probability of filter failure within 7 days using only the purifier's existing TDS / flow-rate / pressure-differential telemetry — no added hardware, no GPU, no cloud, no paid APIs.

**Critical limitation:** All data is synthetic (200 virtual purifier units, 120–180 day lifecycles). All results are simulated on synthetic data, not real purifier field results. 100 ppm is this experiment's chosen TDS threshold, not a universal filter-failure or drinking-water safety standard.

## Achieved results (actual, from the trained model on corrected labels)

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
| Classification threshold | 0.30 (selected on validation set) |
| Model | GradientBoostingClassifier |

**Important:** Accuracy (94.00%) is inflated by severe class imbalance (94.7% negatives). The false-negative rate is 28.75% — the model misses about 3 out of 10 actual failures. This is a known limitation requiring further work before any real deployment.

For the 190 of 200 units where the model issued an alert, average simulated advance warning vs. the reactive 100 ppm threshold was 7.3 days (range: -5 to 21 days).

---

## 3. Components/modules completed so far (incl. sourcing-risk flags)

| Module | Component | Status |
|---|---|---|
| Module 1 | `data/generate_synthetic_data.py` — synthetic purifier sensor data (200 units), corrected labels `0 <= days_to_failure <= 7`, 12 automated tests | Complete |
| Module 2 | `model/train_model.py` — GradientBoostingClassifier, 94.00% test accuracy, threshold selection on validation set, model + metrics saved | Complete |
| Module 3 | `app/dashboard.py` — base dashboard: unit selector, live plotly trends, prediction panel (probability only, no future scan), retrospective comparison banner, playback control with stable auto-play | Complete |
| Module 4 | Business Impact Panel — early-warning advantage, illustrative cost impact scenario, scalability framing (hypothetical only) | Complete |
| Module 5 | Simulated Fail-Safe Panel — trigger at current model prob ≥ 30%, with Round 3 disclaimer | Complete |

**Sourcing-risk flags (from abstract):** The abstract proposed a physical TDS-sensor + relay accelerated-degradation testbed (ESP32/Arduino, TDS sensor, relay). Components could **not** be sourced within the submission window, so the hardware addendum was cancelled. The team deliberately shipped a complete, validated software pipeline over a rushed/incomplete hardware integration. This was raised as a sourcing risk and resolved by pivoting to software-only.

---

## 4. Amount spent so far vs. the abstract's cost estimate

- Amount spent so far: **[FILL IN]** (₹)
- Abstract cost estimate: **[FILL IN]** (₹)
- Notes: Software-only build — zero hardware spend. Original BOM for the cancelled hardware testbed was ~₹600–900 (ESP32/Arduino, analog TDS sensor, 5V relay, container, jumper wires), sourced same-day locally; not purchased due to timeline.

---

## 5. Changes since the abstract (hardware pivot stated explicitly)

The original software-only plan became high-risk after the Round 2 guideline change: *"The focus of this round is on hardware prototypes. A software-only prototype will be considered only if the value it delivers is immense."* We evaluated adding a physical TDS sensor + relay testbed but could not source components within the submission window. We prioritized shipping a complete, validated software pipeline (synthetic data generation, trained model at 94.00% accuracy with known limitations, live interactive dashboard, business impact panel, and complete fail-safe decision logic) over a rushed or incomplete hardware integration. Physical sensor/actuator integration is the committed first task for Round 3. This frames the software-only choice as a deliberate engineering trade-off, not an oversight.

---

## 6. Status of the safety/fail-safe mechanism

**Module 5 (Simulated Fail-Safe Panel) is complete in software.** When the model's current probability of failure within 7 days ≥ 30% (based on readings available at the selected simulation day only), the dashboard shows a prominent alert: *"⚠ FAIL-SAFE TRIGGERED (Simulated): auto flow-restriction + replace-now alert"*. The panel is explicitly labeled: *"Decision logic and trigger condition complete and demonstrated in software. Physical actuation (relay-driven valve) is the committed first task for Round 3."* Physical relay-driven valve actuation is pending hardware procurement (Round 3).

---

## 7. Planned next steps for Round 3

1. Procure and assemble the physical TDS-sensor + relay testbed (ESP32/Arduino, analog TDS module, 5V relay, water container) — same-day local sourcing, no online ordering given timeline.
2. Build the accelerated-degradation method: progressively raise TDS via measured salt increments over a short session; stream live TDS (+ flow, if added) to the trained model for live prediction.
3. Wire the physical fail-safe: relay triggers LED + buzzer (or simulated inlet shutoff) when the model predicts failure within 7 days — the explicit Round 2 video/report requirement #5.
4. Improve model recall (currently 71.25%, FNR 28.75%) — address class imbalance and feature quality before any real deployment.
5. Replace simulated dashboard input with the live ESP32 feed for the on-camera demo.
6. Record the Round 2 video (plain phone recording sufficient) and finalize the progress report PDF.

---

**Key achieved metrics (for reference):**
- Model test accuracy: **94.00%** (≥ 85% acceptance criterion met, but driven by class imbalance)
- Precision: 45.78% | Recall: 71.25% | F1: 55.75%
- False-negative rate (missed failures): 28.75%
- Average simulated advance warning vs. reactive baseline: 7.3 days (190/200 units with an alert)

**All results are simulated on synthetic data. No real purifier hardware or field data was used.**
