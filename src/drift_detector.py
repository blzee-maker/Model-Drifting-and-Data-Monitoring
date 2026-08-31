"""Statistical drift detection between a baseline dataset and incoming data.

Two independent signals are computed per feature:

* **Mean drift** - the relative shift in the feature average exceeds a
  threshold. Cheap to compute and easy to explain, but blind to changes that
  preserve the mean (e.g. a distribution that widens symmetrically).
* **Distribution drift** - a two-sample Kolmogorov-Smirnov test rejects the
  null hypothesis that both samples come from the same distribution. Catches
  shape changes that the mean alone misses.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from scipy.stats import ks_2samp

DEFAULT_MEAN_THRESHOLD = 0.1
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class FeatureDrift:
    """Drift verdict and supporting statistics for a single feature."""

    feature: str
    baseline_mean: float
    current_mean: float
    relative_mean_shift: float
    ks_statistic: float
    p_value: float
    mean_drift: bool
    distribution_drift: bool

    @property
    def drifted(self) -> bool:
        """True if either signal fired."""
        return self.mean_drift or self.distribution_drift

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"drifted": self.drifted}


def relative_mean_shift(baseline_mean: float, current_mean: float) -> float:
    """Relative change in mean, guarding against a zero baseline.

    A zero baseline mean makes the usual ratio undefined, so we report an
    infinite shift when the current mean has moved away from zero and no shift
    when both are zero.
    """
    baseline_mean = float(baseline_mean)
    current_mean = float(current_mean)

    if math.isnan(baseline_mean) or math.isnan(current_mean):
        return math.nan

    denominator = abs(baseline_mean)
    difference = abs(baseline_mean - current_mean)

    if denominator == 0:
        return 0.0 if difference == 0 else math.inf

    return difference / denominator


def mean_drift_detected(
    baseline_mean: float,
    current_mean: float,
    threshold: float = DEFAULT_MEAN_THRESHOLD,
) -> bool:
    """True if the feature average moved by more than ``threshold`` (relative)."""
    return bool(relative_mean_shift(baseline_mean, current_mean) > threshold)


def ks_drift_detected(
    baseline: pd.Series,
    current: pd.Series,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float, bool]:
    """Run a two-sample KS test.

    Returns the KS statistic, the p-value, and whether the null hypothesis of
    "same distribution" is rejected at significance level ``alpha``.
    """
    statistic, p_value = ks_2samp(baseline, current)
    return float(statistic), float(p_value), bool(p_value < alpha)


def comparable_columns(baseline: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    """Numeric columns present in both frames, in baseline order.

    Non-numeric columns are skipped: neither the mean shift nor the KS test is
    defined for them.
    """
    numeric_baseline = baseline.select_dtypes(include="number").columns
    return [column for column in numeric_baseline if column in current.columns]


def detect_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    mean_threshold: float = DEFAULT_MEAN_THRESHOLD,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, FeatureDrift]:
    """Compare every shared numeric feature and return a per-feature verdict.

    Raises:
        ValueError: if the two frames share no comparable numeric columns, or
            if either frame is empty.
    """
    if baseline.empty or current.empty:
        raise ValueError("Both the baseline and current datasets must be non-empty.")

    columns = comparable_columns(baseline, current)
    if not columns:
        raise ValueError(
            "No shared numeric columns to compare. "
            f"Baseline columns: {list(baseline.columns)}; "
            f"current columns: {list(current.columns)}."
        )

    report: dict[str, FeatureDrift] = {}

    for column in columns:
        baseline_column = baseline[column].dropna()
        current_column = current[column].dropna()

        if baseline_column.empty or current_column.empty:
            continue

        baseline_mean = float(baseline_column.mean())
        current_mean = float(current_column.mean())
        shift = relative_mean_shift(baseline_mean, current_mean)
        statistic, p_value, distribution_drift = ks_drift_detected(
            baseline_column, current_column, alpha
        )

        report[column] = FeatureDrift(
            feature=column,
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            relative_mean_shift=shift,
            ks_statistic=statistic,
            p_value=p_value,
            mean_drift=bool(shift > mean_threshold),
            distribution_drift=distribution_drift,
        )

    return report
