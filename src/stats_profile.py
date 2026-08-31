"""Baseline statistical profiling of a dataset.

The profile is the "training-time snapshot" the architecture diagram refers to:
the summary you would persist alongside a trained model so that later batches
can be compared against it without shipping the original training data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_profile(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summarise every numeric column as mean, std, min, max and count.

    Non-numeric columns are skipped, mirroring what ``detect_drift`` compares.
    """
    profile: dict[str, dict[str, float]] = {}

    for column in df.select_dtypes(include="number").columns:
        series = df[column].dropna()
        if series.empty:
            continue

        profile[column] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    return profile


def format_profile(profile: dict[str, dict[str, Any]]) -> str:
    """Render a profile as a fixed-width table."""
    if not profile:
        return "  (no numeric features)"

    header = f"  {'feature':<16}{'count':>8}{'mean':>14}{'std':>14}{'min':>12}{'max':>12}"
    rows = [header, "  " + "-" * (len(header) - 2)]

    for feature, stats in profile.items():
        rows.append(
            f"  {feature:<16}{stats['count']:>8}{stats['mean']:>14.2f}"
            f"{stats['std']:>14.2f}{stats['min']:>12.2f}{stats['max']:>12.2f}"
        )

    return "\n".join(rows)
