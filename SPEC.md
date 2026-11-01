# AtomQuest FilterGuard — Build Spec for OpenCode

**Team:** Code&Design (Ansh Chaudhary, Ayush Gupta) | **Track 2 — Water Purifier** | Area: Prognostics of filter failure using data analytics

**Claim being demonstrated:** Predict filter failure 5–7 days in advance at ≥85% accuracy, vs. reactive (0-day) threshold alerts in current purifiers.

**Constraint:** Zero hardware budget. Entire system runs on synthetic sensor data — no physical purifier required for Round 2.

---

## Tech Stack

- Python 3.11
- pandas, numpy — data handling
- scikit-learn — model (RandomForestClassifier or GradientBoostingClassifier)
- streamlit — dashboard
- plotly — charts
- No GPU, no paid APIs, no cloud spend required

---

## Repo Structure

```
atomquest-filterguard/
├── data/
│   └── generate_synthetic_data.py
├── model/
│   ├── train_model.py
│   └── filterguard_model.pkl   (output artifact)
├── app/
│   └── dashboard.py
├── requirements.txt
└── README.md
```

---

## Module 1 — `data/generate_synthetic_data.py`

Simulate N=200 virtual purifier units, each over a filter lifecycle of 120–180 days (randomized per unit). Daily readings per unit:

| Column | Type | Behavior |
| --- | --- | --- |
| `unit_id` | int | 1–200 |
| `day` | int | days since filter install |
| `tds_ppm` | float | starts \~20–40 ppm, degrades upward following a logistic/exponential curve toward a failure ceiling (\~150 ppm), plus gaussian noise (σ≈2) |
| `flow_rate_lpm` | float | starts \~2.0 L/min, slowly declines as filter clogs, noise σ≈0.05 |
| `pressure_diff_psi` | float | starts \~2 psi, rises as filter clogs (inverse of flow decline), noise σ≈0.1 |
| `days_to_failure` | int | ground-truth label — computed backward from the day `tds_ppm` first crosses the failure threshold (100 ppm) for that unit |
| `failed_within_7d` | 0/1 | binary label — 1 if `days_to_failure <= 7` |

Randomize each unit's degradation rate (±20%) so curves aren't identical — this is what makes the prediction task non-trivial. Output: `data/synthetic_filter_data.csv`.

---

## Module 2 — `model/train_model.py`

**Feature engineering** (rolling over last 5 days per unit):

- `tds_slope_5d`, `flow_slope_5d`, `pressure_slope_5d` — linear regression slope over trailing 5-day window
- `tds_ppm`, `flow_rate_lpm`, `pressure_diff_psi` — current readings
- `day` — age of filter

**Target:** `failed_within_7d` (binary classification)

**Model:** `GradientBoostingClassifier` (scikit-learn), 80/20 train/test split, stratified by `unit_id` (no unit's data leaks across split).

**Acceptance criterion:** test-set accuracy ≥ 0.85. Also report precision/recall and a confusion matrix — judges will want to see false-negative rate (missed failures) specifically, since that's the safety-relevant error.

Save trained model to `model/filterguard_model.pkl` via joblib. Print final metrics to console on training run.

---

## Module 3 — `app/dashboard.py` (Streamlit)

**Purpose:** this is what gets recorded for the 90-second Round 2 video. Build it to *demo well*, not just function.

Views:

1. **Unit selector** — dropdown to pick one of the 200 simulated units
2. **Live trend charts** (plotly) — TDS, flow rate, pressure differential over the unit's lifecycle so far, with a vertical line marking "today"
3. **Prediction panel** — big, clear readout: "Filter failure predicted in **X days**" with a confidence/probability score from the model
4. **Comparison banner** — side-by-side: "Reactive alert: would trigger on day Y (when TDS crosses threshold)" vs. "FilterGuard alert: triggered on day Z" — Y−Z = the advance-warning days gained. This visual is the entire point of the pitch, make it prominent.
5. **Playback control** — a slider or "simulate forward" button that advances the selected unit day-by-day, so in the video you can show the prediction firing *before* the reactive threshold does, in real time.

---

## Acceptance Criteria (Definition of Done for Round 2)

- [ ] `streamlit run app/dashboard.py` runs locally with no errors
- [ ] Model hits ≥85% test accuracy (logged in README with the actual number achieved)
- [ ] At least one unit in the dataset visibly demonstrates a 5–7 day advance warning in the dashboard
- [ ] README documents: problem, approach, how to run, actual metrics achieved (for Documentation & Explainability scoring criterion)
- [ ] Repo is public, clean (no .venv, no **pycache**, no API keys — per hackathon repo rules), with this spec's claim and the achieved numbers both stated in README

---

## Round 2 Video Script (90 seconds)

| Time | Content |
| --- | --- |
| 0:00–0:15 | Problem: "Current purifiers alert you only after the filter has already failed — 0 days of warning." |
| 0:15–0:45 | Live dashboard demo: select a unit, show trends, hit play/simulate-forward |
| 0:45–1:15 | The moment of payoff: FilterGuard's prediction fires, reactive threshold hasn't yet — show the day gap on screen |
| 1:15–1:30 | Close: state the achieved accuracy number and advance-warning days from the actual trained model (not the target — the real result) |

---

## Notes for OpenCode

- Build Module 1 first, inspect the CSV output manually before building Module 2 — if the synthetic degradation curves look unrealistic (too clean, too noisy), tune noise/curve parameters before training on it.
- If accuracy comes in under 85% on first pass, the likely fix is better features (slope features are doing most of the work) before reaching for a bigger model.
- Keep requirements.txt minimal — every extra dependency is something that can break the demo right before recording.

---

## Round 2 Update (as of 2 Nov 2026)

**Corrected deadline: 4 November 2026, 11:59 PM IST.** Submit on the same platform used for the abstract.

**Critical rule change:** official Round 2 guidelines state *"The focus of this round is on hardware prototypes. A software-only prototype will be considered only if the value it delivers is immense."* The original software-only plan is now high-risk. Pivoting to add a minimal real hardware element.

### Hardware Addendum — Accelerated Degradation Testbed

**BOM (\~₹600–900, source same-day locally — do not order online given the timeline):**

- ESP32 or Arduino Uno
- Analog TDS sensor module
- 5V relay module (for the fail-safe demo)
- Small water container, jumper wires/breadboard
- Optional: cheap flow sensor if time allows

**Method:** Don't wait for natural filter degradation (months). Accelerate it — progressively raise TDS of a water sample via measured salt increments over a short session (hours). ESP32 logs live TDS (+ flow, if added) and streams it to the trained model for a live prediction, instead of only replaying synthetic data.

**Safety / fail-safe mechanism (explicitly required in the video, item 5):** wire the relay to trigger an LED + buzzer (or simulated inlet shutoff) when the model predicts failure within 7 days. This is the single most important addition — it's named explicitly in both the video and report requirements.

**Software impact:** Modules 1–3 from the original spec are largely unchanged. The synthetic data generator (Module 1) remains the training-data source (a few hours of real readings isn't enough to train on alone); the live ESP32 feed becomes the on-camera demo input for the dashboard instead of purely replayed synthetic data.

**Video note:** production quality does not affect evaluation — a plain phone recording is sufficient as long as the prototype and its core function are clearly visible. Don't spend time on editing polish.

### Progress Report Template (PDF, 1–2 pages)

1. Team name, college, track, area of exploration
2. Problem + solution recap
3. Components/modules completed so far, incl. status of any sourcing-risk flags raised in the abstract
4. Amount spent so far vs. the abstract's cost estimate
5. Changes since the abstract (state the hardware pivot here explicitly, with reason: Round 2 rules favor hardware)
6. Status of the safety/fail-safe mechanism
7. Planned next steps for Round 3

### Naming Convention

Format: `atomQUEST2026_[TeamName]_[CollegeName]_Track[No]_Video` / `_Report`

**Note:** team name "Code&Design" contains `&`, which risks breaking the filename convention — use `CodeDesign` or `CodeAndDesign` for the actual submitted filenames:

- `atomQUEST2026_CodeDesign_MIET_Track2_Video`
- `atomQUEST2026_CodeDesign_MIET_Track2_Report`

---

## Round 2 Revision (Nov 2, later) — Hardware Not Feasible, Going Software-Only

The hardware addendum above is **cancelled** — sourcing components isn't possible within the submission window. Reverting to a software-only build, but adding two modules specifically to maximize demonstrated value, since the rules say software-only is acceptable "if the value it delivers is immense."

Modules 1–3 from the original spec (synthetic data generator, model training, dashboard) are unchanged and still the core. Add the following two modules on top.

### Module 4 — Business Impact Panel (add to the dashboard)

A panel on the dashboard showing, clearly labeled as **estimated/projected**:

- **Early-warning advantage:** the actual days-earlier detection achieved by the model vs. the reactive threshold baseline (pull this number from Module 2's real evaluation results — not a guess)
- **Cost impact:** assume an average filter cartridge cost of ₹400–600; show projected annual savings per household from avoiding premature/emergency replacements
- **Scalability framing:** one line stating that since the system runs on existing TDS/flow/pressure telemetry with no added hardware cost, it could be deployed across Atomberg's full IoT-connected purifier fleet at near-zero marginal cost

Keep all three numbers traceable to the model's actual test metrics — don't invent figures that can't be defended if a judge asks.

### Module 5 — Simulated Fail-Safe Panel (add to the dashboard)

Logic: when the selected unit's predicted `days_to_failure <= 2`, show a prominent alert panel: **"⚠ FAIL-SAFE TRIGGERED (Simulated): auto flow-restriction + replace-now alert"**.

Label it explicitly: *"Decision logic and trigger condition complete and demonstrated in software. Physical actuation (relay-driven valve) is the committed first task for Round 3."*

This directly answers the video's requirement #5 (status of the safety/fail-safe mechanism) with an honest, specific answer rather than an omission.

### Progress Report — Section 5 Guidance (Changes Since Abstract)

Write this section head-on, don't bury it:

> "We evaluated adding a physical TDS sensor + relay testbed for this round but could not source components within the submission window. We prioritized shipping a complete, validated software pipeline — synthetic data generation, a trained model (see Section 3 for achieved accuracy), a live interactive dashboard, and complete fail-safe decision logic — over a rushed or incomplete hardware integration. Physical sensor and actuator integration is the committed first task for Round 3."

This frames the software-only choice as a deliberate engineering trade-off, not an oversight — which is a stronger position than leaving it unaddressed.

### Updated OpenCode Instruction

Build Modules 1–5 in order. After Module 2, report the actual test accuracy achieved before proceeding — if it's below 85%, fix features before building the dashboard on top of a weak model.
