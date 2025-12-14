# Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

## Project Objective
The objective of this project is to design, implement, and rigorously evaluate
advanced deep learning models for multivariate time series forecasting, with
a specific focus on attention mechanisms to improve accuracy and interpretability.

## Dataset Description
A synthetic multivariate time series dataset with 2000 observations is generated.
The dataset includes:
- Trend components
- Multiple seasonal patterns
- Exogenous variables
- Two correlated target variables

This setup simulates real-world forecasting scenarios such as economic or
environmental time series.

## Models Implemented
1. Exponential Smoothing (statistical baseline)
2. Standard LSTM (deep learning baseline)
3. LSTM with Additive Attention (proposed model)

## Experimental Setup
- Train / Validation / Test split: 70% / 15% / 15%
- Sequence length: 40
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)
- Hyperparameters tuned: hidden dimension, learning rate

## Evaluation Metrics
Models are evaluated on a held-out test set using:
- RMSE
- MAE
- SMAPE

## Attention Interpretability
The attention mechanism assigns importance weights to each time step.
A visualization of average attention weights is generated to interpret
temporal dependencies and seasonal influence.

## Reproducibility
The entire pipeline is deterministic, with fixed random seeds and
clearly defined data splits to ensure reproducibility.
