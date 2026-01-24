import pandas as pd
from drift_detector import detect_drift
from monitor import monitor_system

def main():
    train_df = pd.read_csv("data/train.csv")
    new_df = pd.read_csv("data/new_data.csv")

    drift_report = detect_drift(train_df, new_df)

    print("\n📊 DRIFT REPORT")
    for feature, result in drift_report.items():
        print(f"{feature}: {result}")

    status = monitor_system(drift_report)
    print("\nSYSTEM STATUS:", status)

if __name__ == "__main__":
    main()
