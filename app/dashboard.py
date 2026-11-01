"""Module 3 — AtomQuest FilterGuard dashboard (base views 1-5).

Views:
  1. Unit selector — dropdown to pick one of the 200 simulated units.
  2. Live trend charts (plotly) — TDS, flow rate, pressure differential over
     the unit's lifecycle so far, with a vertical line marking "today".
  3. Prediction panel — "Filter failure predicted in X days" with the model's
     confidence/probability score of failure within 7 days.
  4. Comparison banner — side-by-side reactive alert day Y vs FilterGuard
     alert day Z; Y−Z = advance-warning days gained.
  5. Playback control — slider + Next/Previous + auto-simulate advancing the
     selected unit day-by-day so the prediction can be shown firing before
     the reactive threshold.

Run:  streamlit run app/dashboard.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_filter_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "filterguard_model.pkl")

TDS_FAILURE_THRESHOLD = 100.0
FAIL_PROB_THRESHOLD = 0.5
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

st.title("💧 AtomQuest FilterGuard — Filter Failure Prognostics")
st.caption(
    "Predicts filter failure 5–7 days in advance using only existing "
    "TDS / flow / pressure telemetry — no added hardware. "
    "Track 2 • Water Purifier • Prognostics of filter failure"
)

df = load_data()
model = load_model()
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

p1, p2 = col_btns.columns(2)
if p1.button("◀ Prev", use_container_width=True):
    st.session_state.sim_day = max(1, st.session_state.sim_day - 1)
    st.session_state.sim_playing = False
    st.rerun()
if p2.button("Next ▶", use_container_width=True):
    st.session_state.sim_day = min(lifecycle, st.session_state.sim_day + 1)
    st.session_state.sim_playing = False
    st.rerun()

if st.sidebar.button("▶ Simulate forward (auto)", use_container_width=True):
    st.session_state.sim_playing = True

if st.session_state.sim_playing:
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

pred_days = None
if has_features:
    for d in range(today, min(today + 7, lifecycle) + 1):
        r = unit_df[unit_df["day"] == d]
        if len(r) == 0 or pd.isna(r.iloc[0]["tds_slope_5d"]):
            continue
        p_d = float(
            model.predict_proba(pd.DataFrame([r.iloc[0][feature_cols]], columns=feature_cols))[0, 1]
        )
        if p_d >= FAIL_PROB_THRESHOLD:
            pred_days = d - today
            break

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
        days_text = "today / imminent" if pred_days == 0 else f"in {pred_days} day{'s' if pred_days != 1 else ''}"
        st.error(
            f"⚠ **Filter failure predicted {days_text}**  \n"
            f"Confidence (model probability): **{prob:.0%}**"
        )
    else:
        st.success(
            f"✅ No filter failure predicted within the next 7 days  \n"
            f"Confidence of failure: **{prob:.0%}**"
        )
else:
    st.warning("Not enough history yet to compute slope features (need ≥ 2 days).")

# ---------------------------------------------------------------------------
# View 4 — Comparison banner (the pitch visual)
# ---------------------------------------------------------------------------
st.subheader("📡 Reactive vs. FilterGuard — advance warning gained")
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.markdown("**Reactive alert** (today's purifiers)")
    st.metric("Triggers on day", reactive_day if reactive_day is not None else "—")
    st.caption("When TDS first crosses the 100 ppm failure threshold — 0 days of warning.")
with c2:
    st.markdown("**FilterGuard alert** (this model)")
    st.metric("Triggers on day", alert_day if alert_day is not None else "—")
    st.caption("First day the model predicts failure within 7 days (P ≥ 50%).")
with c3:
    st.markdown("**Advance warning gained**")
    if advance is not None:
        st.metric("Days earlier", f"{advance} day{'s' if advance != 1 else ''}")
        st.caption(f"FilterGuard fires {advance} day{'s' if advance != 1 else ''} before the reactive threshold.")
    else:
        st.metric("Days earlier", "—")
        st.caption("Model never issued an imminent-failure alert for this unit.")

st.markdown(
    "AtomQuest FilterGuard • Code&Design • Track 2 — Water Purifier • "
    "Prognostics of filter failure using data analytics"
)
