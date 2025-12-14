import numpy as np
import pandas as pd

def generate_time_series(n_steps=1500, seed=42):
    np.random.seed(seed)
    time = np.arange(n_steps)

    trend = 0.005 * time
    seasonal_short = 2 * np.sin(2 * np.pi * time / 50)
    seasonal_long = 1.5 * np.sin(2 * np.pi * time / 200)

    exog_1 = np.random.normal(0, 1, n_steps)
    exog_2 = np.random.normal(0, 0.5, n_steps)
    noise = np.random.normal(0, 0.3, n_steps)

    target = trend + seasonal_short + seasonal_long + 0.3 * exog_1 + noise

    df = pd.DataFrame({
        "target": target,
        "trend": trend,
        "seasonal_short": seasonal_short,
        "seasonal_long": seasonal_long,
        "exog_1": exog_1,
        "exog_2": exog_2
    })

    return df

if __name__ == "__main__":
    df = generate_time_series()
    df.to_csv("timeseries.csv", index=False)
