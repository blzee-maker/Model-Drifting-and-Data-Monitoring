import pandas as pd
import pytest

from stats_profile import compute_profile, format_profile


class TestComputeProfile:
    def test_summarises_numeric_columns(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        profile = compute_profile(df)["x"]
        assert profile["count"] == 3
        assert profile["mean"] == pytest.approx(2.0)
        assert profile["min"] == pytest.approx(1.0)
        assert profile["max"] == pytest.approx(3.0)
        assert profile["std"] == pytest.approx(1.0)

    def test_skips_non_numeric_columns(self):
        df = pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})
        assert list(compute_profile(df)) == ["x"]

    def test_ignores_missing_values(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0]})
        assert compute_profile(df)["x"]["count"] == 2

    def test_all_missing_column_is_skipped(self):
        df = pd.DataFrame({"x": [None, None]}, dtype="float64")
        assert compute_profile(df) == {}

    def test_empty_frame_gives_empty_profile(self):
        assert compute_profile(pd.DataFrame()) == {}


class TestFormatProfile:
    def test_renders_a_row_per_feature(self):
        rendered = format_profile(compute_profile(pd.DataFrame({"x": [1.0, 2.0]})))
        assert "x" in rendered
        assert "feature" in rendered

    def test_handles_empty_profile(self):
        assert "no numeric features" in format_profile({})
