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

**Solution:** FilterGuard predicts filter failure 5–7 days in advance using only the purifier's existing TDS / flow-rate / pressure-differential telemetry — no added hardware, no GPU, no cloud, no paid APIs. A GradientBoostingClassifier trained on synthetic sensor data (200 virtual units, 120–180 day lifecycles) predicts failure within 7 days at **97.39% test accuracy** with a 4.81% false-negative rate. A Streamlit dashboard demonstrates unit selection, live trend charts, a prediction panel, a reactive-vs-FilterGuard comparison banner (advance-warning days gained), day-by-day playback, a business impact panel, and a simulated fail-safe panel.

---

## 3. Components/modules completed so far (incl. sourcing-risk flags)

| Module | Component | Status |
|---|---|---|
| Module 1 | `data/generate_synthetic_data.py` — synthetic purifier sensor data (200 units) | Complete |
| Module 2 | `model/train_model.py` — GradientBoostingClassifier, 97.39% test accuracy, model + metrics saved | Complete |
| Module 3 | `app/dashboard.py` — base dashboard: unit selector, live plotly trends, prediction panel, reactive-vs-FilterGuard banner, playback control | Complete |
| Module 4 | Business Impact Panel — early-warning advantage, projected cost impact, scalability framing | Complete |
| Module 5 | Simulated Fail-Safe Panel — trigger at predicted days-to-failure ≤ 2, with Round 3 disclaimer | Complete |

**Sourcing-risk flags (from abstract):** The abstract proposed a physical TDS-sensor + relay accelerated-degradation testbed (ESP32/Arduino, TDS sensor, relay). Components could **not** be sourced within the submission window, so the hardware addendum was cancelled. The team deliberately shipped a complete, validated software pipeline over a rushed/incomplete hardware integration. This was raised as a sourcing risk and resolved by pivoting to software-only.

---

## 4. Amount spent so far vs. the abstract's cost estimate

- Amount spent so far: **[FILL IN]** (₹)
- Abstract cost estimate: **[FILL IN]** (₹)
- Notes: Software-only build — zero hardware spend. Original BOM for the cancelled hardware testbed was ~₹600–900 (ESP32/Arduino, analog TDS sensor, 5V relay, container, jumper wires), sourced same-day locally; not purchased due to timeline.

---

## 5. Changes since the abstract (hardware pivot stated explicitly)

The original software-only plan became high-risk after the Round 2 guideline change: *"The focus of this round is on hardware prototypes. A software-only prototype will be considered only if the value it delivers is immense."* We evaluated adding a physical TDS sensor + relay testbed but could not source components within the submission window. We prioritized shipping a complete, validated software pipeline (synthetic data generation, trained model at 97.39% accuracy, live interactive dashboard, business impact panel, and complete fail-safe decision logic) over a rushed or incomplete hardware integration. Physical sensor/actuator integration is the committed first task for Round 3. This frames the software-only choice as a deliberate engineering trade-off, not an oversight.

---

## 6. Status of the safety/fail-safe mechanism

**Module 5 (Simulated Fail-Safe Panel) is complete in software.** When the selected unit's predicted days-to-failure ≤ 2, the dashboard shows a prominent alert: *"⚠ FAIL-SAFE TRIGGERED (Simulated): auto flow-restriction + replace-now alert"*. The panel is explicitly labeled: *"Decision logic and trigger condition complete and demonstrated in software. Physical actuation (relay-driven valve) is the committed first task for Round 3."* Physical relay-driven valve actuation is pending hardware procurement (Round 3).

---

## 7. Planned next steps for Round 3

1. Procure and assemble the physical TDS-sensor + relay testbed (ESP32/Arduino, analog TDS module, 5V relay, water container) — same-day local sourcing, no online ordering given timeline.
2. Build the accelerated-degradation method: progressively raise TDS via measured salt increments over a short session; stream live TDS (+ flow, if added) to the trained model for live prediction.
3. Wire the physical fail-safe: relay triggers LED + buzzer (or simulated inlet shutoff) when the model predicts failure within 7 days — the explicit Round 2 video/report requirement #5.
4. Replace simulated dashboard input with the live ESP32 feed for the on-camera demo.
5. Record the Round 2 video (plain phone recording sufficient) and finalize the progress report PDF.

---

**Key achieved metrics (for reference):**
- Model test accuracy: **97.39%** (≥ 85% acceptance criterion met)
- Precision: 96.92% | Recall: 95.19% | F1: 96.05%
- False-negative rate (missed failures): 4.81%
- Average advance warning vs. reactive baseline: ~5–7 days (computed per unit from real model predictions)
