"""Module 1 — Synthetic sensor data generator for AtomQuest FilterGuard.

Simulates N=200 virtual purifier units, each over a filter lifecycle of
120-180 days (randomized per unit). Daily readings per unit:

  unit_id          int    1-200
  day              int    days since filter install
  tds_ppm          float  starts ~20-40 ppm, degrades upward on a logistic
                          curve toward a ~150 ppm ceiling + gaussian noise (sigma~2)
  flow_rate_lpm    float  starts ~2.0 L/min, declines as filter clogs (sigma~0.05)
  pressure_diff_psi float starts ~2 psi, rises as filter clogs (sigma~0.1)
  days_to_failure  int    ground truth — days back from the day tds first
                          crosses the 100 ppm failure threshold (0 on failure
                          day, negative after failure)
  failed_within_7d 0/1    1 if 0 <= days_to_failure <= 7 (failure day plus
                          the 7 days before it); post-failure days are NOT
                          counted as positives

Each unit's degradation rate is randomized +/- 20% so curves are not identical.

Output: data/synthetic_filter_data.csv
"""

import os

import numpy as np
import pandas as pd

N_UNITS = 200
LIFECYCLE_MIN = 120
LIFECYCLE_MAX = 180
TDS_BASE_MIN = 20.0
TDS_BASE_MAX = 40.0
TDS_CEILING = 150.0
TDS_FAILURE_THRESHOLD = 100.0
TDS_NOISE_SIGMA = 2.0
FLOW_START = 2.0
FLOW_FLOOR = 0.5
FLOW_NOISE_SIGMA = 0.05
PRESSURE_START = 2.0
PRESSURE_END = 5.0
PRESSURE_NOISE_SIGMA = 0.1
RANDOM_SEED = 42


def generate_unit(unit_id, rng):
    """Generate one unit's full lifecycle as a DataFrame."""
    lifecycle = int(rng.integers(LIFECYCLE_MIN, LIFECYCLE_MAX + 1))
    base_tds = float(rng.uniform(TDS_BASE_MIN, TDS_BASE_MAX))

    # Nominal failure (tds crossing 100 ppm) at ~75% of lifecycle. Even with a
    # 20% slower degradation rate the crossing still lands before end-of-life,
    # so every unit is guaranteed to cross the threshold within its lifecycle.
    nominal_failure_day = 0.75 * lifecycle
    ratio = (TDS_CEILING - TDS_FAILURE_THRESHOLD) / (TDS_CEILING - base_tds)
    r_nominal = -np.log(ratio) / nominal_failure_day

    # Randomize degradation rate +/- 20% so curves differ across units.
    r = r_nominal * float(rng.uniform(0.8, 1.2))

    days = np.arange(1, lifecycle + 1, dtype=float)

    # TDS approaches the ceiling exponentially (clogging progress).
    progress = 1.0 - np.exp(-r * days)  # 0 -> 1 over the lifecycle
    tds = TDS_CEILING - (TDS_CEILING - base_tds) * np.exp(-r * days)
    tds = tds + rng.normal(0.0, TDS_NOISE_SIGMA, size=lifecycle)

    # Flow declines as filter clogs (inverse of tds progress).
    flow = FLOW_START - (FLOW_START - FLOW_FLOOR) * progress
    flow = flow + rng.normal(0.0, FLOW_NOISE_SIGMA, size=lifecycle)

    # Pressure rises as flow declines (inverse relationship).
    pressure = PRESSURE_START + (PRESSURE_END - PRESSURE_START) * progress
    pressure = pressure + rng.normal(0.0, PRESSURE_NOISE_SIGMA, size=lifecycle)

    # Ground truth: first day tds crosses the failure threshold.
    crossed = tds >= TDS_FAILURE_THRESHOLD
    if bool(np.any(crossed)):
        failure_day = int(np.argmax(crossed)) + 1  # days are 1-indexed
    else:
        failure_day = lifecycle  # degenerate guard; should not occur

    days_to_failure = failure_day - days
    # Positive only during the failure day and the 7 days before it.
    # Post-failure days (negative days_to_failure) are NOT positives.
    failed_within_7d = ((days_to_failure >= 0) & (days_to_failure <= 7)).astype(int)

    return pd.DataFrame({
        "unit_id": unit_id,
        "day": days.astype(int),
        "tds_ppm": tds,
        "flow_rate_lpm": flow,
        "pressure_diff_psi": pressure,
        "days_to_failure": days_to_failure.astype(int),
        "failed_within_7d": failed_within_7d,
    })


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    frames = [generate_unit(uid, rng) for uid in range(1, N_UNITS + 1)]
    data = pd.concat(frames, ignore_index=True)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_filter_data.csv")
    data.to_csv(out_path, index=False)

    n_cross = int(data.groupby("unit_id")["tds_ppm"].max().ge(TDS_FAILURE_THRESHOLD).sum())
    n_pos = int(data["failed_within_7d"].sum())
    print(f"Generated {len(data)} rows across {N_UNITS} units.")
    print(f"Units crossing {TDS_FAILURE_THRESHOLD} ppm threshold: {n_cross} / {N_UNITS}")
    print(f"failed_within_7d positives: {n_pos} ({100 * data['failed_within_7d'].mean():.1f}% of rows)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
