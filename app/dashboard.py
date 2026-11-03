"""Module 3 — AtomQuest FilterGuard dashboard (views 1-5).

Views:
  1. Unit selector — dropdown to pick one of the 200 simulated units.
  2. Live trend charts (plotly) — TDS, flow rate, pressure differential over
     the unit's lifecycle so far, with a vertical line marking "today".
  3. Prediction panel — model's probability of filter failure within 7 days,
     computed from current and historical readings only (no future-day scan).
  4. Comparison banner — reactive alert day Y vs FilterGuard alert day Z
     (first day the model's probability, computed from readings available at
     that day, exceeded the threshold); Y−Z = advance-warning days gained.
  5. Playback control — slider + Next/Previous + auto-simulate advancing the
     selected unit day-by-day.

Important: this is a synthetic-data prototype. 100 ppm is the experiment's
chosen TDS threshold, not a universal filter-failure or drinking-water safety
standard. All results are simulated, not real purifier field results.

Run:  streamlit run app/dashboard.py
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_filter_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "filterguard_model.pkl")

from app.config import FAIL_PROB_THRESHOLD, TDS_FAILURE_THRESHOLD

SLOPE_WINDOW = 5


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["unit_id", "day"]).reset_index(drop=True)
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    metrics_path = os.path.join(BASE_DIR, "model", "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Feature engineering (matches Module 2 exactly)
# ---------------------------------------------------------------------------
def _slope(series):
    if len(series) < 2:
        return np.nan
    x = np.arange(len(series), dtype=float)
    return float(np.polyfit(x, series.values, 1)[0])


@st.cache_data
def enrich_features(df):
    out = df.copy()
    slope_map = {
        "tds_ppm": "tds_slope_5d",
        "flow_rate_lpm": "flow_slope_5d",
        "pressure_diff_psi": "pressure_slope_5d",
    }
    for col, slope_col in slope_map.items():
        out[slope_col] = (
            out.groupby("unit_id")[col]
            .transform(lambda s: s.rolling(window=SLOPE_WINDOW, min_periods=2).apply(_slope, raw=False))
        )
    feature_cols = [
        "tds_slope_5d",
        "flow_slope_5d",
        "pressure_slope_5d",
        "tds_ppm",
        "flow_rate_lpm",
        "pressure_diff_psi",
        "day",
    ]
    model = load_model()
    model_rows = out.dropna(subset=list(slope_map.values())).copy()
    probs = model.predict_proba(model_rows[feature_cols])[:, 1]
    model_rows["fail_prob"] = probs
    out = out.merge(model_rows[["unit_id", "day", "fail_prob"]], on=["unit_id", "day"], how="left")
    return out, feature_cols


# ---------------------------------------------------------------------------
# Per-unit derived facts (reactive day, FilterGuard alert day, advance warning)
# ---------------------------------------------------------------------------
@st.cache_data
def unit_facts(df):
    facts = {}
    for uid, g in df.groupby("unit_id"):
        g = g.sort_values("day")
        reactive_day = int(g.loc[g["tds_ppm"] >= TDS_FAILURE_THRESHOLD, "day"].iloc[0])
        alert_rows = g[g["fail_prob"] >= FAIL_PROB_THRESHOLD]
        alert_day = int(alert_rows["day"].iloc[0]) if len(alert_rows) else None
        advance = (reactive_day - alert_day) if alert_day is not None else None
        facts[uid] = {
            "reactive_day": reactive_day,
            "alert_day": alert_day,
            "advance": advance,
            "lifecycle": int(g["day"].max()),
            "base_tds": float(g["tds_ppm"].iloc[0]),
        }
    return facts


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AtomQuest FilterGuard", page_icon="💧", layout="wide")

st.title("💧 AtomQuest FilterGuard — Filter Failure Prognostics (Simulated Prototype)")
st.caption(
    "Simulated prototype: predicts probability of filter failure within 7 days "
    "from TDS / flow-rate / pressure-differential telemetry — no added hardware. "
    "100 ppm is this experiment's chosen TDS threshold, not a universal "
    "filter-failure or drinking-water safety standard. Results are simulated on "
    "synthetic data, not real purifier field results. Track 2 • Water Purifier • "
    "Prognostics of filter failure using data analytics"
)

df = load_data()
model = load_model()
metrics = load_metrics()
enriched, feature_cols = enrich_features(df)
facts = unit_facts(enriched)

unit_ids = sorted(enriched["unit_id"].unique())

# ---------------------------------------------------------------------------
# Sidebar — unit selector + playback control (Views 1 & 5)
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")
selected_unit = st.sidebar.selectbox("Select purifier unit (1–200)", unit_ids, index=0)

unit_df = enriched[enriched["unit_id"] == selected_unit].sort_values("day")
lifecycle = int(unit_df["day"].max())

if "sim_day" not in st.session_state:
    st.session_state.sim_day = 1
if "sim_playing" not in st.session_state:
    st.session_state.sim_playing = False

col_sld, col_btns = st.sidebar.columns([3, 1])
today = col_sld.slider("Today (day)", 1, lifecycle, value=st.session_state.sim_day)
if st.session_state.sim_day != today:
    st.session_state.sim_day = today
    st.session_state.sim_playing = False

# Playback speed control
sim_speed = st.sidebar.select_slider(
    "Playback speed",
    options=["Slow", "Normal", "Fast"],
    value="Normal",
    help="Delay between auto-advance steps",
)
speed_delay = {"Slow": 1.5, "Normal": 0.8, "Fast": 0.3}[sim_speed]

p1, p2, p3 = col_btns.columns(3)
if p1.button("◀ Prev", use_container_width=True):
    st.session_state.sim_day = max(1, st.session_state.sim_day - 1)
    st.session_state.sim_playing = False
    st.rerun()
if p2.button("Next ▶", use_container_width=True):
    st.session_state.sim_day = min(lifecycle, st.session_state.sim_day + 1)
    st.session_state.sim_playing = False
    st.rerun()
if p3.button("↻ Reset", use_container_width=True):
    st.session_state.sim_day = 1
    st.session_state.sim_playing = False
    st.rerun()

if st.sidebar.button("▶ Simulate forward (auto)", use_container_width=True):
    st.session_state.sim_playing = True

if st.session_state.sim_playing:
    time.sleep(speed_delay)
    st.session_state.sim_day = min(lifecycle, st.session_state.sim_day + 1)
    if st.session_state.sim_day >= lifecycle:
        st.session_state.sim_playing = False
    st.rerun()

today = st.session_state.sim_day
st.sidebar.markdown(f"**Lifecycle:** {lifecycle} days   •   **Today:** day {today}")

# ---------------------------------------------------------------------------
# Row for "today"
# ---------------------------------------------------------------------------
today_row = unit_df[unit_df["day"] == today]
has_features = len(today_row) > 0 and pd.notna(today_row.iloc[0]["tds_slope_5d"])
row = today_row.iloc[0] if len(today_row) else None

fact = facts[selected_unit]
reactive_day = fact["reactive_day"]
alert_day = fact["alert_day"]
advance = fact["advance"]

if has_features:
    prob = float(model.predict_proba(pd.DataFrame([row[feature_cols]], columns=feature_cols))[0, 1])
else:
    prob = None

# NOTE: No future-day sensor-reading scan is performed. The prediction uses
# only current and historical readings available at the selected simulation
# day. We do not claim an exact days-to-failure forecast; we report the
# model's probability of failure within the next 7 days.

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Unit", selected_unit)
k2.metric("Day", f"{today} / {lifecycle}")
k3.metric("P(failure ≤ 7 days)", f"{prob:.0%}" if prob is not None else "insufficient data")
k4.metric("Reactive alert day", reactive_day if reactive_day is not None else "—")

# ---------------------------------------------------------------------------
# View 2 — Live trend charts (plotly)
# ---------------------------------------------------------------------------
st.subheader("📊 Live sensor trends (lifecycle so far)")
visible = unit_df[unit_df["day"] <= today]

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    subplot_titles=("TDS (ppm)", "Flow rate (L/min)", "Pressure differential (psi)"),
)

fig.add_trace(
    go.Scatter(x=visible["day"], y=visible["tds_ppm"], name="TDS",
               line=dict(color="#e74c3c", width=2)), row=1, col=1,
)
fig.add_hline(y=TDS_FAILURE_THRESHOLD, line_dash="dash", line_color="#c0392b",
              annotation_text="Failure threshold 100 ppm", annotation_position="bottom right", row=1, col=1)

fig.add_trace(
    go.Scatter(x=visible["day"], y=visible["flow_rate_lpm"], name="Flow",
               line=dict(color="#2ecc71", width=2)), row=2, col=1,
)

fig.add_trace(
    go.Scatter(x=visible["day"], y=visible["pressure_diff_psi"], name="Pressure",
               line=dict(color="#3498db", width=2)), row=3, col=1,
)

for r in (1, 2, 3):
    fig.add_vline(x=today, line_dash="dot", line_color="#7f8c8d", line_width=1.5, row=r, col=1)
    fig.add_annotation(x=today, y=1.0, xref=f"x{r}", yref="paper",
                       text="today", showarrow=False,
                       font=dict(size=10, color="#7f8c8d"),
                       xanchor="left", yanchor="bottom")

fig.update_xaxes(title_text="Day since install", row=3, col=1)
fig.update_yaxes(title_text="ppm", row=1, col=1)
fig.update_yaxes(title_text="L/min", row=2, col=1)
fig.update_yaxes(title_text="psi", row=3, col=1)
fig.update_layout(height=560, showlegend=False, margin=dict(t=60, b=40, l=60, r=20))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# View 3 — Prediction panel
# ---------------------------------------------------------------------------
st.subheader("🔮 Prediction panel")
if has_features and prob is not None:
    if prob >= FAIL_PROB_THRESHOLD:
        st.error(
            f"⚠ **Filter failure predicted within 7 days**  \n"
            f"Model probability of failure: **{prob:.0%}**  \n"
            f"Based on current and historical readings up to day {today} only. "
            f"No exact days-to-failure forecast is made."
        )
    else:
        st.success(
            f"✅ No filter failure predicted within the next 7 days  \n"
            f"Model probability of failure: **{prob:.0%}**"
        )
else:
    st.warning("Not enough history yet to compute slope features (need ≥ 2 days).")

# ---------------------------------------------------------------------------
# View 4 — Comparison banner (retrospective full-lifecycle comparison)
# ---------------------------------------------------------------------------
st.subheader("📡 Reactive vs. FilterGuard — advance warning gained (retrospective)")
st.caption(
    "This comparison is retrospective: it uses the full lifecycle of the unit "
    "to show when the reactive threshold and the model would each have fired. "
    "It is not a real-time prediction."
)
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.markdown("**Reactive alert** (today's purifiers)")
    st.metric("Triggers on day", reactive_day if reactive_day is not None else "—")
    st.caption("When TDS first crosses the 100 ppm threshold (synthetic experiment threshold) — 0 days of warning.")
with c2:
    st.markdown("**FilterGuard alert** (this model, simulated)")
    st.metric("Triggers on day", alert_day if alert_day is not None else "—")
    st.caption("First day the model's probability of failure within 7 days (computed from readings available that day) exceeded 20%.")
with c3:
    st.markdown("**Advance warning gained** (simulated)")
    if advance is not None:
        st.metric("Days earlier", f"{advance} day{'s' if advance != 1 else ''}")
        st.caption(f"FilterGuard's simulated alert fires {advance} day{'s' if advance != 1 else ''} before the reactive threshold.")
    else:
        st.metric("Days earlier", "—")
        st.caption("Model never issued an imminent-failure alert for this unit.")

# ---------------------------------------------------------------------------
# Module 4 — Business Impact Panel
# ---------------------------------------------------------------------------
st.divider()
st.header("💼 Module 4 — Business Impact (estimated / projected)")

accuracy = metrics.get("accuracy", None)
fnr = metrics.get("false_negative_rate", None)
advance_list = [v["advance"] for v in facts.values() if v["advance"] is not None]
avg_advance = float(np.mean(advance_list)) if advance_list else 0.0

b1, b2, b3 = st.columns(3)

with b1:
    st.subheader("Early-warning advantage")
    st.metric("Average days earlier detection", f"{avg_advance:.1f} days")
    st.caption(
        "Mean of (reactive alert day − FilterGuard alert day) across all 200 units, "
        "computed from the trained model's real predictions — not a guess."
    )
    if accuracy is not None:
        st.metric("Model test accuracy", f"{accuracy:.0%}")
        st.caption("From Module 2 evaluation (test set, stratified by unit).")
    if fnr is not None:
        st.metric("Missed-failure rate (FNR)", f"{fnr:.0%}")
        st.caption("Safety-relevant error from Module 2 test set.")

with b2:
    st.subheader("Cost impact (illustrative scenario)")
    st.caption("All figures below are an illustrative scenario — not measured or guaranteed.")
    cartridge_cost = st.number_input("Filter cartridge cost (₹)", min_value=100, max_value=2000, value=500, step=50)
    replacements_per_year = st.number_input("Replacements per household / year", min_value=1, max_value=12, value=2, step=1)
    avoidable_fraction = st.slider(
        "Fraction of replacements avoidable via early warning",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Illustrative assumption: share of annual replacements that early warning could help avoid.",
    )
    # Illustrative calculation: annual savings = cartridge cost × replacements/year × avoidable fraction.
    annual_savings = cartridge_cost * replacements_per_year * avoidable_fraction
    st.metric("Illustrative annual savings / household", f"₹{annual_savings:,.0f}")
    st.caption(
        f"Formula: ₹{cartridge_cost} × {replacements_per_year} replacements/year × "
        f"{avoidable_fraction:.0%} avoidable = ₹{annual_savings:,.0f}/year. "
        "This is an illustrative scenario, not a measured or guaranteed figure."
    )

with b3:
    st.subheader("Scalability framing (illustrative)")
    st.caption("Illustrative scenario — no field deployment evidence to date.")
    st.markdown(
        "The prototype runs on TDS / flow / pressure readings. If a purifier "
        "already exposes this telemetry, adding the model inference step would "
        "carry near-zero marginal software cost. This is a hypothetical "
        "scalability statement, not a claim about any real fleet."
    )

# ---------------------------------------------------------------------------
# Module 5 — Simulated Fail-Safe Panel
# ---------------------------------------------------------------------------
st.divider()
st.header("🚨 Module 5 — Simulated Fail-Safe Panel")

# Fail-safe trigger uses the CURRENT prediction (probability of failure
# within 7 days from readings available today), not future readings.
failsafe_triggered = has_features and prob is not None and prob >= FAIL_PROB_THRESHOLD

if failsafe_triggered:
    st.error(
        "⚠ **FAIL-SAFE TRIGGERED (Simulated):** auto flow-restriction + replace-now alert  \n"
        f"Model probability of failure within 7 days: **{prob:.0%}**  \n"
        f"Based on current and historical readings up to day {today} only."
    )
else:
    st.success(
        "✅ Fail-safe not triggered  \n"
        f"Model probability of failure within 7 days: "
        f"{f'{prob:.0%}' if prob is not None else 'insufficient data'}"
    )

st.caption(
    "**Decision logic and trigger condition complete and demonstrated in software. "
    "Physical actuation (relay-driven valve) is not part of the Round 3 software roadmap.**"
)

st.markdown(
    "AtomQuest FilterGuard • Code&Design • Track 2 — Water Purifier • "
    "Prognostics of filter failure using data analytics"
)