import math

import numpy as np
import pandas as pd
import pytest

from drift_detector import (
    FeatureDrift,
    comparable_columns,
    detect_drift,
    ks_drift_detected,
    mean_drift_detected,
    relative_mean_shift,
)


class TestRelativeMeanShift:
    def test_reports_proportional_change(self):
        assert relative_mean_shift(100.0, 120.0) == pytest.approx(0.2)

    def test_is_symmetric_in_direction(self):
        assert relative_mean_shift(100.0, 80.0) == pytest.approx(0.2)

    def test_zero_baseline_with_no_change_is_not_drift(self):
        assert relative_mean_shift(0.0, 0.0) == 0.0

    def test_zero_baseline_with_change_is_infinite(self):
        # The naive ratio would raise ZeroDivisionError here.
        assert relative_mean_shift(0.0, 5.0) == math.inf

    def test_nan_propagates(self):
        assert math.isnan(relative_mean_shift(math.nan, 5.0))


class TestMeanDriftDetected:
    def test_below_threshold_is_stable(self):
        assert mean_drift_detected(100.0, 105.0, threshold=0.1) is False

    def test_above_threshold_is_drift(self):
        assert mean_drift_detected(100.0, 130.0, threshold=0.1) is True

    def test_threshold_is_exclusive(self):
        assert mean_drift_detected(100.0, 110.0, threshold=0.1) is False


class TestKsDriftDetected:
    def test_identical_distributions_are_stable(self):
        rng = np.random.default_rng(0)
        sample = pd.Series(rng.normal(0, 1, 500))
        _, p_value, drifted = ks_drift_detected(sample, sample)
        assert drifted is False
        assert p_value > 0.05

    def test_shifted_distribution_is_detected(self):
        rng = np.random.default_rng(0)
        baseline = pd.Series(rng.normal(0, 1, 500))
        current = pd.Series(rng.normal(3, 1, 500))
        _, _, drifted = ks_drift_detected(baseline, current)
        assert drifted is True

    def test_catches_variance_change_the_mean_misses(self):
        """Same mean, different spread: KS fires where mean drift cannot."""
        rng = np.random.default_rng(1)
        baseline = pd.Series(rng.normal(50, 1, 800))
        current = pd.Series(rng.normal(50, 12, 800))
        _, _, drifted = ks_drift_detected(baseline, current)
        assert drifted is True
        assert mean_drift_detected(baseline.mean(), current.mean()) is False


class TestComparableColumns:
    def test_skips_non_numeric_columns(self):
        baseline = pd.DataFrame({"age": [1, 2], "city": ["a", "b"]})
        current = pd.DataFrame({"age": [3, 4], "city": ["c", "d"]})
        assert comparable_columns(baseline, current) == ["age"]

    def test_skips_columns_missing_from_current(self):
        baseline = pd.DataFrame({"age": [1, 2], "income": [3, 4]})
        current = pd.DataFrame({"age": [5, 6]})
        assert comparable_columns(baseline, current) == ["age"]

    def test_preserves_baseline_order(self):
        baseline = pd.DataFrame({"b": [1, 2], "a": [3, 4]})
        current = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        assert comparable_columns(baseline, current) == ["b", "a"]


class TestDetectDrift:
    def test_stable_data_reports_no_drift(self):
        rng = np.random.default_rng(7)
        baseline = pd.DataFrame({"x": rng.normal(10, 2, 600)})
        current = pd.DataFrame({"x": rng.normal(10, 2, 600)})
        report = detect_drift(baseline, current)
        assert report["x"].drifted is False

    def test_shifted_data_reports_drift(self):
        rng = np.random.default_rng(7)
        baseline = pd.DataFrame({"x": rng.normal(10, 2, 600)})
        current = pd.DataFrame({"x": rng.normal(25, 2, 600)})
        report = detect_drift(baseline, current)
        result = report["x"]
        assert result.drifted is True
        assert result.mean_drift is True
        assert result.distribution_drift is True

    def test_returns_supporting_statistics(self):
        baseline = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        current = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        result = detect_drift(baseline, current)["x"]
        assert isinstance(result, FeatureDrift)
        assert result.baseline_mean == pytest.approx(2.5)
        assert result.current_mean == pytest.approx(2.5)
        assert 0.0 <= result.p_value <= 1.0

    def test_ignores_non_numeric_columns(self):
        baseline = pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})
        current = pd.DataFrame({"x": [1, 2, 3], "label": ["x", "y", "z"]})
        assert list(detect_drift(baseline, current)) == ["x"]

    def test_drops_missing_values(self):
        baseline = pd.DataFrame({"x": [1.0, 2.0, 3.0, None]})
        current = pd.DataFrame({"x": [1.0, 2.0, 3.0, None]})
        assert detect_drift(baseline, current)["x"].drifted is False

    def test_zero_baseline_mean_does_not_raise(self):
        baseline = pd.DataFrame({"x": [-1.0, 0.0, 1.0]})
        current = pd.DataFrame({"x": [9.0, 10.0, 11.0]})
        assert detect_drift(baseline, current)["x"].mean_drift is True

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            detect_drift(pd.DataFrame({"x": []}), pd.DataFrame({"x": [1.0]}))

    def test_no_shared_numeric_columns_raises(self):
        baseline = pd.DataFrame({"a": [1, 2]})
        current = pd.DataFrame({"b": [3, 4]})
        with pytest.raises(ValueError, match="No shared numeric columns"):
            detect_drift(baseline, current)

    def test_thresholds_are_configurable(self):
        baseline = pd.DataFrame({"x": [100.0] * 50})
        current = pd.DataFrame({"x": [105.0] * 50})
        assert detect_drift(baseline, current, mean_threshold=0.10)["x"].mean_drift is False
        assert detect_drift(baseline, current, mean_threshold=0.01)["x"].mean_drift is True
