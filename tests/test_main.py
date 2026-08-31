import json
from pathlib import Path

import pandas as pd
import pytest

import main
from main import EXIT_ERROR, EXIT_RETRAIN, EXIT_STABLE

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


class _FakeStdout:
    """Stands in for sys.stdout, whose real ``encoding`` is read-only."""

    def __init__(self, encoding: str):
        self.encoding = encoding


@pytest.fixture
def csv_factory(tmp_path):
    def write(name: str, frame: pd.DataFrame) -> Path:
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return path

    return write


class TestExitCodes:
    def test_stable_scenario_exits_zero(self, capsys):
        code = main.main(["--current", str(DATA_DIR / "stable_data.csv")])
        capsys.readouterr()
        assert code == EXIT_STABLE

    def test_drift_scenario_exits_one(self, capsys):
        code = main.main(["--current", str(DATA_DIR / "new_data.csv")])
        capsys.readouterr()
        assert code == EXIT_RETRAIN

    def test_missing_file_exits_two_with_message(self, capsys):
        code = main.main(["--baseline", "does/not/exist.csv"])
        assert code == EXIT_ERROR
        assert "not found" in capsys.readouterr().err

    def test_incomparable_schemas_exit_two(self, capsys, csv_factory):
        baseline = csv_factory("a.csv", pd.DataFrame({"a": [1, 2, 3]}))
        current = csv_factory("b.csv", pd.DataFrame({"b": [4, 5, 6]}))
        code = main.main(["--baseline", str(baseline), "--current", str(current)])
        assert code == EXIT_ERROR
        assert "No shared numeric columns" in capsys.readouterr().err


class TestOutput:
    def test_default_run_reports_retraining(self, capsys):
        main.main([])
        out = capsys.readouterr().out
        assert "DRIFT REPORT" in out
        assert "RETRAIN MODEL" in out

    def test_profile_flag_adds_baseline_profile(self, capsys):
        main.main(["--profile"])
        assert "BASELINE PROFILE" in capsys.readouterr().out

    def test_json_flag_emits_parsable_json(self, capsys):
        main.main(["--json"])
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"features", "system"}
        assert payload["system"]["retrain_required"] is True
        assert payload["features"]["age"]["drifted"] is True

    @pytest.mark.parametrize("encoding", ["cp1252", "ascii", "latin-1"])
    def test_glyphs_downgrade_on_a_legacy_console(self, encoding, monkeypatch):
        """Regression test: emoji output used to crash on cp1252 terminals."""
        monkeypatch.setattr(main.sys, "stdout", _FakeStdout(encoding))
        for symbol in main._glyphs().values():
            symbol.encode(encoding)  # would raise UnicodeEncodeError before the fix

    def test_glyphs_stay_fancy_on_a_utf8_console(self, monkeypatch):
        monkeypatch.setattr(main.sys, "stdout", _FakeStdout("utf-8"))
        assert main._glyphs()["ok"] == "✅"

    def test_glyphs_survive_an_unknown_encoding(self, monkeypatch):
        monkeypatch.setattr(main.sys, "stdout", _FakeStdout("not-a-real-codec"))
        assert main._glyphs()["ok"] == "[OK]"


class TestThresholds:
    def test_lenient_drift_threshold_keeps_model_stable(self, capsys):
        code = main.main(["--drift-threshold", "1.0"])
        capsys.readouterr()
        assert code == EXIT_STABLE

    def test_strict_mean_threshold_flags_more_features(self, capsys):
        main.main(["--current", str(DATA_DIR / "stable_data.csv"),
                   "--mean-threshold", "0.001", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["system"]["retrain_required"] is True
