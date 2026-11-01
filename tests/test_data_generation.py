"""Tests for Module 1 — synthetic data generation and label correctness."""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path so we can import the generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_synthetic_data import (
    generate_unit,
    N_UNITS,
    LIFECYCLE_MIN,
    LIFECYCLE_MAX,
    TDS_FAILURE_THRESHOLD,
    RANDOM_SEED,
)


def test_generate_unit_returns_dataframe_with_expected_columns():
    """Each unit should return a DataFrame with all required columns."""
    rng = np.random.default_rng(RANDOM_SEED)
    df = generate_unit(1, rng)
    expected_cols = {
        "unit_id", "day", "tds_ppm", "flow_rate_lpm",
        "pressure_diff_psi", "days_to_failure", "failed_within_7d",
    }
    assert set(df.columns) == expected_cols, f"Columns mismatch: {set(df.columns)}"


def test_generate_unit_has_valid_lifecycle():
    """Lifecycle should be between 120 and 180 days inclusive."""
    rng = np.random.default_rng(RANDOM_SEED)
    for uid in [1, 50, 100, 200]:
        df = generate_unit(uid, rng)
        lifecycle = df["day"].max()
        assert LIFECYCLE_MIN <= lifecycle <= LIFECYCLE_MAX, (
            f"Unit {uid} lifecycle {lifecycle} out of range [{LIFECYCLE_MIN}, {LIFECYCLE_MAX}]"
        )


def test_generate_unit_days_are_consecutive_from_1():
    """Days should start at 1 and be consecutive integers."""
    rng = np.random.default_rng(RANDOM_SEED)
    df = generate_unit(42, rng)
    expected_days = list(range(1, len(df) + 1))
    assert list(df["day"]) == expected_days


def test_generate_unit_unit_id_is_consistent():
    """unit_id column should match the requested unit_id."""
    rng = np.random.default_rng(RANDOM_SEED)
    for uid in [1, 10, 200]:
        df = generate_unit(uid, rng)
        assert (df["unit_id"] == uid).all()


def test_generate_unit_tds_is_reproducible_with_same_seed():
    """Same seed should produce identical data."""
    rng1 = np.random.default_rng(RANDOM_SEED)
    rng2 = np.random.default_rng(RANDOM_SEED)
    df1 = generate_unit(7, rng1)
    df2 = generate_unit(7, rng2)
    pd.testing.assert_frame_equal(df1, df2)


def test_label_logic_failure_day_and_preceding_seven_days_only():
    """failed_within_7d should be 1 only when 0 <= days_to_failure <= 7."""
    rng = np.random.default_rng(RANDOM_SEED)
    df = generate_unit(1, rng)
    # Check all rows
    for _, row in df.iterrows():
        dtf = row["days_to_failure"]
        expected = 1 if (0 <= dtf <= 7) else 0
        assert row["failed_within_7d"] == expected, (
            f"Row day={row['day']}: days_to_failure={dtf}, "
            f"failed_within_7d={row['failed_within_7d']}, expected={expected}"
        )


def test_post_failure_days_are_not_positives():
    """Days after failure (negative days_to_failure) must all be label 0."""
    rng = np.random.default_rng(RANDOM_SEED)
    df = generate_unit(1, rng)
    post_failure = df[df["days_to_failure"] < 0]
    assert (post_failure["failed_within_7d"] == 0).all(), (
        "Post-failure days should not be labeled as failures"
    )


def test_days_before_seven_day_window_are_not_positives():
    """Days with days_to_failure > 7 must all be label 0."""
    rng = np.random.default_rng(RANDOM_SEED)
    df = generate_unit(1, rng)
    early = df[df["days_to_failure"] > 7]
    assert (early["failed_within_7d"] == 0).all(), (
        "Days outside the 7-day window should not be labeled as failures"
    )


def test_full_dataset_has_valid_labels():
    """Regenerated dataset should have valid label distribution."""
    rng = np.random.default_rng(RANDOM_SEED)
    frames = [generate_unit(uid, rng) for uid in range(1, N_UNITS + 1)]
    data = pd.concat(frames, ignore_index=True)

    # All units should cross the threshold
    n_cross = data.groupby("unit_id")["tds_ppm"].max().ge(TDS_FAILURE_THRESHOLD).sum()
    assert n_cross == N_UNITS, f"Only {n_cross}/{N_UNITS} units crossed threshold"

    # Each unit should have exactly 8 positive labels (days 0-7 before/incl failure)
    for uid in range(1, N_UNITS + 1):
        unit_data = data[data["unit_id"] == uid]
        n_pos = int(unit_data["failed_within_7d"].sum())
        assert n_pos == 8, (
            f"Unit {uid} has {n_pos} positive labels, expected 8"
        )


def test_no_missing_values_in_generated_data():
    """Generated data should have no missing values."""
    rng = np.random.default_rng(RANDOM_SEED)
    frames = [generate_unit(uid, rng) for uid in range(1, N_UNITS + 1)]
    data = pd.concat(frames, ignore_index=True)
    assert not data.isna().any().any(), "Generated data contains missing values"


def test_no_duplicate_records():
    """Each (unit_id, day) pair should be unique."""
    rng = np.random.default_rng(RANDOM_SEED)
    frames = [generate_unit(uid, rng) for uid in range(1, N_UNITS + 1)]
    data = pd.concat(frames, ignore_index=True)
    dupes = data.duplicated(subset=["unit_id", "day"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate (unit_id, day) records"


def test_tds_degrades_upward_toward_failure():
    """TDS should generally increase toward the failure threshold."""
    rng = np.random.default_rng(RANDOM_SEED)
    df = generate_unit(1, rng)
    failure_day = int(df.loc[df["days_to_failure"] == 0, "day"].iloc[0])
    pre_failure = df[df["day"] <= failure_day]
    # TDS at failure day should be >= 100 ppm (the threshold)
    tds_at_failure = pre_failure.loc[pre_failure["day"] == failure_day, "tds_ppm"].iloc[0]
    assert tds_at_failure >= TDS_FAILURE_THRESHOLD, (
        f"TDS at failure day ({tds_at_failure}) should be >= {TDS_FAILURE_THRESHOLD}"
    )
