import pytest

from drift_detector import FeatureDrift
from monitor import RETRAIN, STABLE, monitor_system


def make_drift(feature: str, *, mean_drift=False, distribution_drift=False) -> FeatureDrift:
    """Build a FeatureDrift with only the verdict flags that matter here."""
    return FeatureDrift(
        feature=feature,
        baseline_mean=0.0,
        current_mean=0.0,
        relative_mean_shift=0.0,
        ks_statistic=0.0,
        p_value=1.0,
        mean_drift=mean_drift,
        distribution_drift=distribution_drift,
    )


def report(*flags: bool) -> dict[str, FeatureDrift]:
    """A report of len(flags) features, drifted where the flag is True."""
    return {
        f"f{i}": make_drift(f"f{i}", mean_drift=flag)
        for i, flag in enumerate(flags)
    }


class TestMonitorSystem:
    def test_no_drift_is_stable(self):
        status = monitor_system(report(False, False, False))
        assert status.status == STABLE
        assert status.retrain_required is False
        assert status.drift_ratio == 0.0

    def test_all_features_drifted_triggers_retrain(self):
        status = monitor_system(report(True, True, True))
        assert status.status == RETRAIN
        assert status.retrain_required is True
        assert status.drift_ratio == 1.0

    def test_ratio_at_threshold_is_stable(self):
        # 1 of 2 drifted = 0.5; the check is strictly greater than.
        status = monitor_system(report(True, False), drift_threshold=0.5)
        assert status.status == STABLE

    def test_ratio_above_threshold_triggers_retrain(self):
        status = monitor_system(report(True, False), drift_threshold=0.4)
        assert status.status == RETRAIN

    def test_distribution_drift_alone_counts(self):
        drift_report = {"x": make_drift("x", distribution_drift=True)}
        assert monitor_system(drift_report).retrain_required is True

    def test_lists_the_drifted_features(self):
        status = monitor_system(report(True, False, True))
        assert status.drifted_features == ["f0", "f2"]
        assert status.total_features == 3

    def test_empty_report_raises(self):
        # Guards the ZeroDivisionError that a naive ratio would hit.
        with pytest.raises(ValueError, match="empty drift report"):
            monitor_system({})

    def test_to_dict_is_json_friendly(self):
        payload = monitor_system(report(True, False)).to_dict()
        assert payload["status"] == RETRAIN
        assert payload["drifted_features"] == ["f0"]
        assert payload["total_features"] == 2
