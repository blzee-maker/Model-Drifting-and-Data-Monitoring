# Model Drift & Data Monitoring

[![CI](https://github.com/blzee-maker/Model-Drifting-and-Data-Monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/blzee-maker/Model-Drifting-and-Data-Monitoring/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A monitoring pipeline that detects **data drift** by comparing incoming production
data against a training-time baseline, and raises a retraining signal when the
share of drifted features crosses a configurable threshold.

Most ML projects stop at training and evaluation. In production, models fail
*silently*: the code keeps running and the predictions keep flowing, but the
world the model was trained on has moved. This project covers what happens after
deployment.

---

## Quickstart

```bash
git clone https://github.com/blzee-maker/Model-Drifting-and-Data-Monitoring.git
cd Model-Drifting-and-Data-Monitoring

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python src/main.py --profile
```

---

## Example run

Comparing the baseline against a drifted batch:

```text
BASELINE PROFILE
  feature            count          mean           std         min         max
  ----------------------------------------------------------------------------
  age                 1000         34.85          8.64       18.00       64.00
  income              1000      48779.34      15218.11     4286.00    93714.00
  credit_score        1000        701.35         40.65      577.00      822.00

📊 DRIFT REPORT
  feature             baseline     current     shift   p-value  verdict
  ---------------------------------------------------------------------
  age                    34.85       44.01     26.3%    0.0000  mean + distribution drift
  income              48779.34    67531.71     38.4%    0.0000  mean + distribution drift
  credit_score          701.35      667.70      4.8%    0.0000  distribution drift

🚨 SYSTEM STATUS: RETRAIN MODEL  (3/3 features drifted, 100% vs 30% threshold)
```

Against the control batch, drawn from the same population as the baseline:

```bash
python src/main.py --current data/stable_data.csv
```

```text
📊 DRIFT REPORT
  feature             baseline     current     shift   p-value  verdict
  ---------------------------------------------------------------------
  age                    34.85       35.24      1.1%    0.2407  stable
  income              48779.34    49357.26      1.2%    0.5729  stable
  credit_score          701.35      700.67      0.1%    0.5729  stable

✅ SYSTEM STATUS: MODEL STABLE  (0/3 features drifted, 0% vs 30% threshold)
```

---

## How drift is detected

Two independent signals run per feature, because neither is sufficient alone.

### Mean shift

Flags a feature when its average moves by more than a relative threshold
(default `0.1`, i.e. 10%). Cheap and easy to explain, but blind to any change
that leaves the mean intact — a distribution can double in spread and keep the
same centre.

### Distribution drift (Kolmogorov–Smirnov)

A two-sample KS test compares the full empirical distributions and rejects the
null hypothesis of "same population" at significance level `alpha`
(default `0.05`). This catches changes in shape, spread, and skew.

**`credit_score` in the example above is exactly why both exist:** its mean moved
only 4.8%, under the 10% threshold, so mean drift alone would have called it
stable. The KS test still rejected the null hypothesis.

### System-level decision

`monitor.py` aggregates the per-feature verdicts into one decision: if the share
of drifted features exceeds `--drift-threshold` (default `0.3`), retraining is
recommended. One noisy feature out of twenty should not page anyone; a majority
shifting should.

---

## Usage

```
python src/main.py [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--baseline PATH` | `data/train.csv` | CSV of training-time baseline data |
| `--current PATH` | `data/new_data.csv` | CSV of incoming production data |
| `--mean-threshold FLOAT` | `0.1` | Relative mean change that counts as drift |
| `--alpha FLOAT` | `0.05` | Significance level for the KS test |
| `--drift-threshold FLOAT` | `0.3` | Share of drifted features that triggers retraining |
| `--profile` | off | Also print the baseline statistical profile |
| `--json` | off | Emit machine-readable JSON instead of tables |

### Exit codes

The exit code carries the verdict, so the pipeline drops straight into cron, CI,
or an orchestrator without parsing stdout:

| Code | Meaning |
| --- | --- |
| `0` | Model stable |
| `1` | Retraining recommended |
| `2` | Error (missing file, empty dataset, no comparable columns) |

```bash
python src/main.py || echo "drift detected - kicking off retraining"
```

### JSON output

```bash
python src/main.py --json
```

```json
{
  "features": {
    "age": {
      "feature": "age",
      "baseline_mean": 34.852,
      "current_mean": 44.012,
      "relative_mean_shift": 0.26282566280270875,
      "ks_statistic": 0.386,
      "p_value": 8.25723185131597e-67,
      "mean_drift": true,
      "distribution_drift": true,
      "drifted": true
    }
  },
  "system": {
    "status": "RETRAIN MODEL",
    "retrain_required": true,
    "drift_ratio": 1.0,
    "drifted_features": ["age", "income", "credit_score"],
    "total_features": 3,
    "threshold": 0.3
  }
}
```

Any numeric columns shared by both CSVs are compared; non-numeric columns and
columns missing from either side are skipped, and missing values are dropped
per feature.

---

## Project structure

```
.
├── data/
│   ├── train.csv              # Training-time baseline (1,000 rows)
│   ├── new_data.csv           # Incoming batch from a shifted population
│   └── stable_data.csv        # Control batch from the baseline population
├── notebooks/
│   └── exploration.ipynb      # Visual walkthrough with distribution plots
├── scripts/
│   └── generate_data.py       # Regenerates all three datasets (seeded)
├── src/
│   ├── stats_profile.py       # Baseline statistical profiling
│   ├── drift_detector.py      # Per-feature drift detection
│   ├── monitor.py             # System-level retraining decision
│   └── main.py                # CLI entry point
├── tests/                     # pytest suite
├── .github/workflows/ci.yml   # CI across Linux/Windows, Python 3.10–3.13
├── requirements.txt
├── requirements-dev.txt
└── LICENSE
```

---

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the statistical helpers, the aggregation logic, and the CLI
end to end — including the edge cases that a naive implementation gets wrong:

- a zero baseline mean (a plain ratio raises `ZeroDivisionError`)
- an empty drift report (same problem, one level up)
- non-numeric and mismatched columns
- a variance-only shift, where KS fires and mean drift cannot
- console output on legacy `cp1252` terminals, where emoji used to crash the run

### Regenerating the data

The datasets are synthetic and reproducible from a fixed seed:

```bash
python scripts/generate_data.py
```

`train.csv` and `stable_data.csv` are drawn from the same distributions;
`new_data.csv` comes from an older, higher-earning cohort with weaker credit.

---

## Tech stack

Python 3.10+ · pandas · NumPy · SciPy · pytest · GitHub Actions

## License

[MIT](LICENSE)
