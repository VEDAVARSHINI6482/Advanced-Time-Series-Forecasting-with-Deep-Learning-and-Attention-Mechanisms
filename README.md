# Advanced Time Series Forecasting with Deep Learning and Attention

## Models Implemented
1. Exponential Smoothing (baseline)
2. Standard LSTM
3. LSTM with Additive Attention

## Hyperparameter Optimization
A grid search was performed over:
- Hidden sizes: {64, 128}
- Learning rates: {0.001, 0.0005}

The best configuration was selected based on validation loss.

## Evaluation
Metrics (RMSE, MAE, SMAPE) are computed for:
- All models
- Both target variables (target_1 and target_2)

## Interpretability
Attention weights are explicitly extracted, saved, and visualized.
