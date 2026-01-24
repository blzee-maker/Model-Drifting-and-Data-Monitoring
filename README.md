# 🚨 Model Drift & Data Monitoring System

A production-style machine learning monitoring system that detects **data drift** by comparing incoming data distributions with training-time baselines and triggers retraining alerts when reliability degrades.

This project focuses on a **critical but often ignored part of ML systems**: what happens *after* a model is deployed.

---

## 🧠 Why This Project?

Most ML projects stop at model training and evaluation.  
In real-world systems, models **fail silently** due to changes in data over time (population drift, seasonality, user behavior changes).

This project simulates how ML teams:
- Monitor incoming data
- Detect distribution shifts
- Decide when retraining is necessary

---

## 🏗️ System Architecture

Training Data (Baseline)
↓
Statistical Profiling
↓
Incoming Production Data
↓
Distribution Comparison
↓
Drift Detection
↓
Alert / Retraining Signal


---

## 🔍 Types of Drift Detected

### 1️⃣ Mean Shift
Detects significant changes in feature averages.

Used to identify:
- Population changes
- Sudden value scale shifts

---

### 2️⃣ Distribution Drift (KS Test)
Uses the **Kolmogorov–Smirnov test** to detect statistically significant differences between feature distributions.

This allows detection beyond simple averages.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- SciPy
- PyTest (optional testing)

---

## 📁 Project Structure

model_drift_monitor/
│
├── data/
│ ├── train.csv # Training / baseline data
│ ├── new_data.csv # Incoming production data
│
├── src/
│ ├── data_loader.py # Data loading utilities
│ ├── stats_profile.py # Training data profiling
│ ├── drift_detector.py # Drift detection logic
│ ├── monitor.py # System-level decision logic
│ └── main.py # Pipeline runner
│
├── tests/
│ └── test_drift.py # Unit tests (optional)
│
├── requirements.txt
└── README.md


---

## ▶️ How to Run

### 1️⃣ Install Dependencies

'''bash
pip install -r requirements.txt
2️⃣ Run Drift Detection

python src/main.py
3️⃣ Sample Output (Drift Detected)
📊 DRIFT REPORT
age: {'mean_drift': True, 'distribution_drift': True}
income: {'mean_drift': True, 'distribution_drift': True}
credit_score: {'mean_drift': True, 'distribution_drift': True}

SYSTEM STATUS: 🚨 RETRAIN MODEL
🧪 Testing Strategy
The system is validated using controlled synthetic datasets:

Stable data scenario → Model marked as stable

Shifted data scenario → Drift detected and retraining recommended
'''