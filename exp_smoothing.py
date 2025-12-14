from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

def exponential_smoothing_forecast(series, train_ratio=0.85):
    split = int(len(series) * train_ratio)
    train, test = series[:split], series[split:]

    model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=50)
    fit = model.fit()
    preds = fit.forecast(len(test))

    return np.array(test), np.array(preds)
