from scipy.stats import ks_2samp


def mean_drift_detected(train_mean, new_mean, threshold=0.1):
    return abs(train_mean - new_mean) / train_mean > threshold

def ks_drift_detected(train_col, new_col, alpha=0.05):
    stat, p_value = ks_2samp(train_col, new_col)
    return p_value < alpha

def detect_drift(train_df, new_df):
    
    drift_report = {}

    for col in train_df.columns:
        mean_shift = mean_drift_detected(
            train_df[col].mean(),
            new_df[col].mean()
        )

        ks_shift = ks_drift_detected(
            train_df[col],
            new_df[col]
        )

        drift_report[col] = {
            "mean_drift": mean_shift,
            "distribution_drift": ks_shift
        }

    return drift_report
