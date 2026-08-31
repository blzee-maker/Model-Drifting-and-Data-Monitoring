"""Regenerate the synthetic datasets used by the demo and the test suite.

Three datasets share one schema (age, income, credit_score):

* ``train.csv``  - the training-time baseline.
* ``new_data.csv`` - an incoming batch drawn from a shifted population: an
  older, higher-earning cohort with slightly weaker credit. This is the
  scenario that should trigger a retraining alert.
* ``stable_data.csv`` - an incoming batch drawn from the *same* population as
  the baseline. Sampling noise only, so the system should report stability.

Run from anywhere:  python scripts/generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

SEED = 42
N_ROWS = 1_000

# (mean, std) per feature, for the baseline and the drifted population.
BASELINE = {
    "age": (35.0, 9.0),
    "income": (50_000.0, 15_000.0),
    "credit_score": (700.0, 40.0),
}
DRIFTED = {
    "age": (44.0, 11.0),
    "income": (68_000.0, 22_000.0),
    "credit_score": (665.0, 55.0),
}


def sample(spec: dict[str, tuple[float, float]], rng: np.random.Generator) -> pd.DataFrame:
    """Draw a dataset from per-feature normal distributions, clipped to sane ranges."""
    frame = pd.DataFrame(
        {name: rng.normal(mean, std, N_ROWS) for name, (mean, std) in spec.items()}
    )
    frame["age"] = frame["age"].clip(18, 90).round().astype(int)
    frame["income"] = frame["income"].clip(0, None).round().astype(int)
    frame["credit_score"] = frame["credit_score"].clip(300, 850).round().astype(int)
    return frame


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train.csv": sample(BASELINE, rng),
        "new_data.csv": sample(DRIFTED, rng),
        "stable_data.csv": sample(BASELINE, rng),
    }

    for filename, frame in datasets.items():
        destination = DATA_DIR / filename
        frame.to_csv(destination, index=False)
        print(f"wrote {destination.relative_to(REPO_ROOT)}  ({len(frame)} rows)")


if __name__ == "__main__":
    main()
