import numpy as np
import pandas as pd

def generate_time_series(n_steps=2000, seed=42):
    np.random.seed(seed)
    t = np.arange(n_steps)

    trend = 0.004 * t
    seasonal_1 = 2.5 * np.sin(2 * np.pi * t / 50)
    seasonal_2 = 1.8 * np.sin(2 * np.pi * t / 200)

    exog_1 = np.random.normal(0, 1, n_steps)
    exog_2 = np.random.normal(0, 0.7, n_steps)
    noise = np.random.normal(0, 0.4, n_steps)

    target_1 = trend + seasonal_1 + 0.4 * exog_1 + noise
    target_2 = seasonal_2 + 0.3 * exog_2 + noise

    df = pd.DataFrame({
        "target_1": target_1,
        "target_2": target_2,
        "trend": trend,
        "seasonal_1": seasonal_1,
        "seasonal_2": seasonal_2,
        "exog_1": exog_1,
        "exog_2": exog_2
    })

    return df

if __name__ == "__main__":
    generate_time_series().to_csv("timeseries.csv", index=False)
