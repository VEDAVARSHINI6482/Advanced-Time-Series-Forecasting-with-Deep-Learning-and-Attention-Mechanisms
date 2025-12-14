# Technical Report

## Dataset
A synthetic multivariate time series with 1500 observations including:
- Trend
- Multiple seasonalities
- Exogenous variables
- Noise

## Experimental Setup
- Model: LSTM, LSTM + Attention
- Optimizer: Adam
- Loss Function: MSE
- Epochs: 20

## Results
Attention-based models outperform baseline LSTM in RMSE, MAE, and SMAPE.

## Interpretability
Attention weights highlight important time steps contributing to predictions.

## Conclusion
Attention mechanisms significantly enhance forecasting accuracy and explainability.
