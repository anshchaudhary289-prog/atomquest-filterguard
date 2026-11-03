"""Shared configuration constants for AtomQuest FilterGuard.

This module holds configuration values that must be consistent across
the application code and its tests, so there is a single source of
truth for each threshold.
"""

# Probability threshold at or above which the dashboard triggers an
# imminent-failure alert and the simulated fail-safe panel.
# Chosen via validation-set threshold selection (see
# docs/reports/THRESHOLD_EXPERIMENT_REPORT.md).
FAIL_PROB_THRESHOLD = 0.20

# TDS failure threshold (ppm) used to determine the reactive alert day.
# This is the synthetic experiment's chosen threshold, not a universal
# drinking-water safety or filter-replacement standard.
TDS_FAILURE_THRESHOLD = 100.0
