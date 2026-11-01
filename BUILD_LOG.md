# BUILD_LOG — AtomQuest FilterGuard (Modules 3–5)

Log of incremental dashboard builds per module. Streamlit runtime is NOT
interacted with directly; only `streamlit run app/dashboard.py` startup is
confirmed via a short headless launch.

## 2026-11-02 — Module 3 (base dashboard)
- Plan: reset app/dashboard.py to Module 3 only (views 1–5 per SPEC):
  1. Unit selector dropdown (200 units)
  2. Live plotly trend charts (TDS/flow/pressure) with "today" line + 100 ppm threshold
  3. Prediction panel: "Filter failure predicted in X days" + confidence
  4. Comparison banner: reactive day Y vs FilterGuard day Z = advance-warning days
  5. Playback control: slider + simulate-forward advancing day-by-day
- After writing: py_compile + streamlit run startup check, then commit & push "feat(module 3): base dashboard (unit selector, trends, prediction, comparison banner, playback)"
- Then proceed to Module 4.
- DONE 2026-11-02: Module 3 ... pushed as "feat(module 3): ...".

## 2026-11-02 — Module 4 (Business Impact Panel)
- Plan: add a Module 4 section to app/dashboard.py with three labeled "estimated/projected" blocks:
  1. Early-warning advantage — mean(reactive day − FilterGuard day) across units (from facts), plus Module 2 test accuracy/FNR.
  2. Cost impact — cartridge cost (₹400-600 input) × avoided premature/emergency replacements per year = projected annual savings/household (labeled projected).
  3. Scalability framing — existing telemetry, near-zero marginal cost across IoT fleet.
- After writing: py_compile + streamlit run startup check, then commit & push "feat(module 4): business impact panel (early warning, cost impact, scalability)".
- Then proceed to Module 5.
- DONE 2026-11-02: Module 4 written (Business Impact Panel added to dashboard). py_compile exit 0. streamlit run on :8503 stayed ALIVE, no errors. Committed & pushed as "feat(module 4): business impact panel (early-warning advantage, projected cost impact, scalability framing)".

## 2026-11-02 — Module 5 (Simulated Fail-Safe Panel)
- Plan: add Module 5 section to app/dashboard.py:
  - When predicted days_to_failure <= 2, show "⚠ FAIL-SAFE TRIGGERED (Simulated): auto flow-restriction + replace-now alert"
  - Explicit disclaimer: decision logic/trigger complete in software; physical actuation (relay-driven valve) committed Round 3.
- After writing: py_compile + streamlit run startup check, then commit & push "feat(module 5): simulated fail-safe panel (trigger at days_to_failure <= 2 + Round 3 disclaimer)".
- Then draft PROGRESS_REPORT_DRAFT.md and final commit/push.
- DONE 2026-11-02: Module 5 written (Simulated Fail-Safe Panel added). py_compile exit 0. streamlit run on :8504 stayed ALIVE, no errors. Committed & pushed as "feat(module 5): simulated fail-safe panel (trigger at days_to_failure <= 2 + Round 3 disclaimer)".

## 2026-11-02 — Critical Validation & Correction Pass

### Issue found: synthetic target labels were wrong
- `failed_within_7d = (days_to_failure <= 7)` labeled ALL post-failure days (negative values) as positive, inflating positives to 32.8% (9,943 rows) and producing a misleading 97.39% accuracy.
- Fix: `failed_within_7d = ((days_to_failure >= 0) & (days_to_failure <= 7))` — positive only on failure day + 7 days before; post-failure days are NOT positives.
- Verified: 1600 positives (5.3%, exactly 200 units × 8 days), 0 positives with days_to_failure < 0 or > 7.

### Corrections made (all committed & pushed)

1. **data/generate_synthetic_data.py** — Fixed label definition to `0 <= days_to_failure <= 7`. Regenerated CSV. Verified label counts and examples before/during/after failure threshold.

2. **model/train_model.py** — Retrained on corrected dataset. Unit-level 80/20 split preserved. New metrics:
   - Accuracy: 95.14% (meets ≥85% criterion, but driven by class imbalance)
   - Precision: 58.28% | Recall: 29.69% | F1: 39.34%
   - FNR: 70.31% | FPR: 1.19%
   - Confusion matrix: (5643, 68, 225, 95)
   - Saved model/filterguard_model.pkl + training_metrics.json with corrected values.
   - Previous 97.39% result NOT retained or claimed (not reproduced).

3. **app/dashboard.py** — Multiple corrections:
   - Removed future-day sensor-reading scan (`pred_days` loop) used to calculate "failure predicted in X days".
   - Prediction panel now shows model's probability of failure within 7 days from current/historical readings only; no exact days-to-failure claim.
   - Fail-safe (Module 5) uses current prediction (prob ≥ 50%), not future readings.
   - Business impact (Module 4): savings labeled as illustrative scenario, assumptions visible, input controls (cartridge cost, replacements/year, avoidable fraction) affect calculation, not described as measured/guaranteed.
   - Scalability framing softened: hypothetical only, no field deployment evidence claimed.
   - 100 ppm described as synthetic experiment's chosen threshold, not universal standard.
   - Page title/caption marked as "(Simulated Prototype)"; results distinguished from real field results.
   - Verified: py_compile exit 0, streamlit run on :8505 stayed ALIVE, no errors.
   - Tested: multiple units (1, 50, 100, 150, 200), playback days before/during/after failure, fail-safe triggered (Unit 1 day 75-83) and not-triggered (post-failure days), missing values (none where features exist), labels correct, metrics match saved output.

4. **README.md, PROGRESS_REPORT_DRAFT.md** — Updated with corrected metrics (95.14% accuracy, 29.69% recall, 70.31% FNR), documented synthetic-data limitation and software-only prototype status, clarified 100 ppm is synthetic threshold.

5. **BUILD_LOG.md** — This entry.

- DONE: All corrections committed & pushed to main. Working tree clean.

## 2026-11-02 — All modules complete + progress report draft
- Modules 3, 4, 5 all built incrementally, each verified (py_compile exit 0 + streamlit run starts without crashing), each committed & pushed separately.
- PROGRESS_REPORT_DRAFT.md drafted per SPEC template (team/college/track, problem+solution, modules completed, spend [FILL IN], hardware pivot, fail-safe status, Round 3 next steps, actual metrics incl. 97.39% accuracy). Committed & pushed as "docs: add progress report draft per SPEC template".
- BUILD: all 5 modules complete. Repo clean (no project __pycache__, no .env, venv ignored). Remote origin set and pushed to main.
- DONE 2026-11-02: Module 5 written (Simulated Fail-Safe Panel added). py_compile exit 0. streamlit run on :8504 stayed ALIVE, no errors. Committed & pushed as "feat(module 5): simulated fail-safe panel (trigger at days_to_failure <= 2 + Round 3 disclaimer)".

## 2026-11-02 — Critical Validation & Correction Pass

### Issue found: synthetic target labels were wrong
- `failed_within_7d = (days_to_failure <= 7)` labeled ALL post-failure days (negative values) as positive, inflating positives to 32.8% (9,943 rows) and producing a misleading 97.39% accuracy.
- Fix: `failed_within_7d = ((days_to_failure >= 0) & (days_to_failure <= 7))` — positive only on failure day + 7 days before; post-failure days are NOT positives.
- Verified: 1600 positives (5.3%, exactly 200 units × 8 days), 0 positives with days_to_failure < 0 or > 7.

### Corrections made (all committed & pushed)

1. **data/generate_synthetic_data.py** — Fixed label definition to `0 <= days_to_failure <= 7`. Regenerated CSV. Verified label counts and examples before/during/after failure threshold.

2. **model/train_model.py** — Retrained on corrected dataset. Unit-level 80/20 split preserved. New metrics:
   - Accuracy: 95.14% (meets ≥85% criterion, but driven by class imbalance)
   - Precision: 58.28% | Recall: 29.69% | F1: 39.34%
   - FNR: 70.31% | FPR: 1.19%
   - Confusion matrix: (5643, 68, 225, 95)
   - Saved model/filterguard_model.pkl + training_metrics.json with corrected values.
   - Previous 97.39% result NOT retained or claimed (not reproduced).

3. **app/dashboard.py** — Multiple corrections:
   - Removed future-day sensor-reading scan (`pred_days` loop) used to calculate "failure predicted in X days".
   - Prediction panel now shows model's probability of failure within 7 days from current/historical readings only; no exact days-to-failure claim.
   - Fail-safe (Module 5) uses current prediction (prob ≥ 50%), not future readings.
   - Business impact (Module 4): savings labeled as illustrative scenario, assumptions visible, input controls (cartridge cost, replacements/year, avoidable fraction) affect calculation, not described as measured/guaranteed.
   - Scalability framing softened: hypothetical only, no field deployment evidence claimed.
   - 100 ppm described as synthetic experiment's chosen threshold, not universal standard.
   - Page title/caption marked as "(Simulated Prototype)"; results distinguished from real field results.
   - Verified: py_compile exit 0, streamlit run on :8505 stayed ALIVE, no errors.
   - Tested: multiple units (1, 50, 100, 150, 200), playback days before/during/after failure, fail-safe triggered (Unit 1 day 75-83) and not-triggered (post-failure days), missing values (none where features exist), labels correct, metrics match saved output.

4. **README.md, PROGRESS_REPORT_DRAFT.md** — Updated with corrected metrics (95.14% accuracy, 29.69% recall, 70.31% FNR), documented synthetic-data limitation and software-only prototype status, clarified 100 ppm is synthetic threshold.

5. **BUILD_LOG.md** — This entry.

- DONE: All corrections committed & pushed to main. Working tree clean.
