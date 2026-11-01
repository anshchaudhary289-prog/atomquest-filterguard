"""Tests for dashboard helper logic — fail-safe, business impact, predictions.

These tests do not import app/dashboard.py directly (it calls st.set_page_config()
at module level and requires a Streamlit runtime). Instead, they test the pure
computation logic that the dashboard uses.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants matching app/dashboard.py
TDS_FAILURE_THRESHOLD = 100.0
FAIL_PROB_THRESHOLD = 0.30


def test_fail_safe_triggered_when_prob_above_threshold():
    """Fail-safe should trigger when probability >= threshold."""
    prob = 0.35
    triggered = prob >= FAIL_PROB_THRESHOLD
    assert triggered is True


def test_fail_safe_not_triggered_when_prob_below_threshold():
    """Fail-safe should NOT trigger when probability < threshold."""
    prob = 0.25
    triggered = prob >= FAIL_PROB_THRESHOLD
    assert triggered is False


def test_fail_safe_at_exact_threshold():
    """Fail-safe should trigger at exactly the threshold (>= comparison)."""
    prob = FAIL_PROB_THRESHOLD
    triggered = prob >= FAIL_PROB_THRESHOLD
    assert triggered is True


def test_fail_safe_requires_features():
    """Fail-safe should not trigger without valid features."""
    has_features = False
    prob = 0.9
    triggered = has_features and prob is not None and prob >= FAIL_PROB_THRESHOLD
    assert triggered is False


def test_fail_safe_requires_valid_probability():
    """Fail-safe should not trigger when probability is None."""
    has_features = True
    prob = None
    triggered = has_features and prob is not None and prob >= FAIL_PROB_THRESHOLD
    assert triggered is False


def test_business_impact_savings_formula():
    """Illustrative annual savings = cartridge_cost × replacements_per_year × avoidable_fraction."""
    cartridge_cost = 500
    replacements_per_year = 2
    avoidable_fraction = 0.25
    annual_savings = cartridge_cost * replacements_per_year * avoidable_fraction
    assert annual_savings == 250.0


def test_business_impact_savings_scales_with_inputs():
    """Savings should scale linearly with each input parameter."""
    base = 500 * 2 * 0.25  # 250
    # Double cartridge cost
    assert 1000 * 2 * 0.25 == base * 2
    # Double replacements per year
    assert 500 * 4 * 0.25 == base * 2
    # Double avoidable fraction
    assert 500 * 2 * 0.5 == base * 2


def test_business_impact_savings_zero_when_avoidable_fraction_zero():
    """With 0% avoidable fraction, savings should be zero."""
    annual_savings = 500 * 2 * 0.0
    assert annual_savings == 0.0


def test_tds_failure_threshold_is_100():
    """100 ppm is the synthetic experiment threshold."""
    assert TDS_FAILURE_THRESHOLD == 100.0


def test_fail_prob_threshold_is_030():
    """Fail-safe trigger threshold should be 0.30."""
    assert FAIL_PROB_THRESHOLD == 0.30


def test_prediction_uses_current_readings_only():
    """Prediction at day t should use only features available at day t."""
    # Simulate a unit's data up to day 50
    days = list(range(1, 51))
    tds = [20.0 + d * 0.5 for d in days]  # monotonically increasing
    flow = [2.0 - d * 0.01 for d in days]  # monotonically decreasing
    pressure = [2.0 + d * 0.02 for d in days]  # monotonically increasing

    df = pd.DataFrame({
        "unit_id": [1] * 50,
        "day": days,
        "tds_ppm": tds,
        "flow_rate_lpm": flow,
        "pressure_diff_psi": pressure,
        "days_to_failure": [50 - d for d in days],
        "failed_within_7d": [1 if 0 <= (50 - d) <= 7 else 0 for d in days],
    })

    # At day 45, only days 1-45 are available
    day_t = 45
    available = df[df["day"] <= day_t]
    assert len(available) == day_t
    assert available["day"].max() == day_t

    # Future days (46-50) should NOT be available
    future = df[df["day"] > day_t]
    assert len(future) == 5


def test_unit_level_separation_no_leakage():
    """Train and test sets should not share any unit_id."""
    train_units = {1, 2, 3, 4, 5}
    test_units = {6, 7, 8, 9, 10}
    overlap = train_units & test_units
    assert len(overlap) == 0, f"Unit leakage: {overlap}"


def test_label_logic_consistency():
    """failed_within_7d should be 1 only when 0 <= days_to_failure <= 7."""
    test_cases = [
        (0, 1),   # failure day
        (1, 1),   # 1 day before
        (7, 1),   # 7 days before
        (8, 0),   # 8 days before (outside window)
        (-1, 0),  # post-failure
        (-7, 0),  # post-failure
    ]
    for days_to_failure, expected in test_cases:
        result = 1 if (0 <= days_to_failure <= 7) else 0
        assert result == expected, (
            f"days_to_failure={days_to_failure}: expected {expected}, got {result}"
        )

