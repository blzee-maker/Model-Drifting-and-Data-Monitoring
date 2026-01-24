def monitor_system(drift_report, drift_threshold=0.3):
    drifted_features = [
        f for f, v in drift_report.items()
        if v["mean_drift"] or v["distribution_drift"]
    ]

    drift_ratio = len(drifted_features) / len(drift_report)

    if drift_ratio > drift_threshold:
        return "🚨 RETRAIN MODEL"
    else:
        return "✅ MODEL STABLE"
