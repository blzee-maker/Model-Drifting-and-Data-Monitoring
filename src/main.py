"""Command-line entry point for the drift monitoring pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from drift_detector import (
    DEFAULT_ALPHA,
    DEFAULT_MEAN_THRESHOLD,
    FeatureDrift,
    detect_drift,
)
from monitor import DEFAULT_DRIFT_THRESHOLD, SystemStatus, monitor_system
from stats_profile import compute_profile, format_profile

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "data" / "train.csv"
DEFAULT_CURRENT = REPO_ROOT / "data" / "new_data.csv"

EXIT_STABLE = 0
EXIT_RETRAIN = 1
EXIT_ERROR = 2


def _glyphs() -> dict[str, str]:
    """Pick output symbols the active console can actually encode.

    Windows terminals often default to cp1252, where printing an emoji raises
    ``UnicodeEncodeError``. Fall back to ASCII markers rather than crashing.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    fancy = {"report": "\U0001F4CA", "alert": "\U0001F6A8", "ok": "\u2705"}
    try:
        for symbol in fancy.values():
            symbol.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return {"report": "==", "alert": "[!]", "ok": "[OK]"}
    return fancy


def format_drift_report(report: dict[str, FeatureDrift]) -> str:
    """Render the per-feature drift report as a fixed-width table."""
    header = (
        f"  {'feature':<16}{'baseline':>12}{'current':>12}"
        f"{'shift':>10}{'p-value':>10}  verdict"
    )
    rows = [header, "  " + "-" * (len(header) - 2)]

    for result in report.values():
        verdict = []
        if result.mean_drift:
            verdict.append("mean")
        if result.distribution_drift:
            verdict.append("distribution")
        label = " + ".join(verdict) + " drift" if verdict else "stable"

        shift = (
            "inf" if result.relative_mean_shift == float("inf")
            else f"{result.relative_mean_shift:.1%}"
        )
        rows.append(
            f"  {result.feature:<16}{result.baseline_mean:>12.2f}"
            f"{result.current_mean:>12.2f}{shift:>10}{result.p_value:>10.4f}  {label}"
        )

    return "\n".join(rows)


def load_dataset(path: Path, label: str) -> pd.DataFrame:
    """Read a CSV, failing with a clear message instead of a stack trace."""
    if not path.exists():
        raise FileNotFoundError(f"{label} dataset not found: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} dataset is empty: {path}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-drift-monitor",
        description=(
            "Compare an incoming dataset against a training baseline and "
            "recommend whether the model should be retrained."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline", type=Path, default=DEFAULT_BASELINE,
        help="CSV of training-time baseline data.",
    )
    parser.add_argument(
        "--current", type=Path, default=DEFAULT_CURRENT,
        help="CSV of incoming production data.",
    )
    parser.add_argument(
        "--mean-threshold", type=float, default=DEFAULT_MEAN_THRESHOLD,
        help="Relative change in a feature mean that counts as mean drift.",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help="Significance level for the Kolmogorov-Smirnov test.",
    )
    parser.add_argument(
        "--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD,
        help="Share of drifted features above which retraining is recommended.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of the formatted tables.",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Also print the baseline statistical profile.",
    )
    return parser


def render_text(
    report: dict[str, FeatureDrift],
    status: SystemStatus,
    baseline: pd.DataFrame,
    show_profile: bool,
) -> None:
    glyphs = _glyphs()

    if show_profile:
        print("\nBASELINE PROFILE")
        print(format_profile(compute_profile(baseline)))

    print(f"\n{glyphs['report']} DRIFT REPORT")
    print(format_drift_report(report))

    marker = glyphs["alert"] if status.retrain_required else glyphs["ok"]
    print(
        f"\n{marker} SYSTEM STATUS: {status.status}"
        f"  ({len(status.drifted_features)}/{status.total_features} features drifted,"
        f" {status.drift_ratio:.0%} vs {status.threshold:.0%} threshold)"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        baseline = load_dataset(args.baseline, "Baseline")
        current = load_dataset(args.current, "Current")
        report = detect_drift(
            baseline, current,
            mean_threshold=args.mean_threshold,
            alpha=args.alpha,
        )
        status = monitor_system(report, drift_threshold=args.drift_threshold)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        print(json.dumps(
            {
                "features": {name: r.to_dict() for name, r in report.items()},
                "system": status.to_dict(),
            },
            indent=2,
        ))
    else:
        render_text(report, status, baseline, args.profile)

    return EXIT_RETRAIN if status.retrain_required else EXIT_STABLE


if __name__ == "__main__":
    raise SystemExit(main())
