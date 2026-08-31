"""System-level decision logic: turn per-feature drift into a single verdict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from drift_detector import FeatureDrift

DEFAULT_DRIFT_THRESHOLD = 0.3

STABLE = "MODEL STABLE"
RETRAIN = "RETRAIN MODEL"


@dataclass(frozen=True)
class SystemStatus:
    """The retraining decision, plus the evidence behind it."""

    status: str
    drift_ratio: float
    drifted_features: list[str]
    total_features: int
    threshold: float

    @property
    def retrain_required(self) -> bool:
        return self.status == RETRAIN

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "retrain_required": self.retrain_required,
            "drift_ratio": self.drift_ratio,
            "drifted_features": self.drifted_features,
            "total_features": self.total_features,
            "threshold": self.threshold,
        }


def monitor_system(
    drift_report: Mapping[str, FeatureDrift],
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> SystemStatus:
    """Recommend retraining when the share of drifted features exceeds the threshold.

    Raises:
        ValueError: if the drift report is empty, since no meaningful ratio can
            be derived from zero features.
    """
    if not drift_report:
        raise ValueError("Cannot assess system health from an empty drift report.")

    drifted_features = [
        feature for feature, result in drift_report.items() if result.drifted
    ]
    drift_ratio = len(drifted_features) / len(drift_report)

    return SystemStatus(
        status=RETRAIN if drift_ratio > drift_threshold else STABLE,
        drift_ratio=drift_ratio,
        drifted_features=drifted_features,
        total_features=len(drift_report),
        threshold=drift_threshold,
    )
