import pandas as pd

def compute_profile(df):
    profile = {}

    for col in df.columns:
        profile[col] = {
            "mean": df[col].mean(),
            "std" : df[col].std(),
            "min" : df[col].min(),
            "max" : df[col].max()
        }
    
    return profile