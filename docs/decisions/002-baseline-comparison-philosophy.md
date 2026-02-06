# ADR 002: Honest Baseline Comparison Philosophy

## Status
Accepted

## Context

Deep learning models for time series forecasting are often presented in isolation, compared only against other neural architectures. This creates a distorted view of their value. In practice, many production forecasting systems use statistical methods (ARIMA, exponential smoothing) that are faster to train, easier to interpret, and sometimes more accurate.

We need a comparison framework that:
1. Includes strong statistical baselines alongside deep learning models
2. Reports results honestly, including cases where simpler methods win
3. Provides statistical significance testing to distinguish real improvements from noise
4. Helps practitioners make informed model selection decisions

## Decision

Every deep learning model in this project is compared against a full suite of baselines:

### Baseline Hierarchy

| Tier | Model | Purpose |
|------|-------|---------|
| 1 | Naive (last value) | Absolute floor; any model must beat this |
| 2 | Seasonal Naive | Captures seasonal patterns without learning |
| 3 | ARIMA/SARIMA | Strong univariate statistical baseline |
| 4 | Prophet | Handles holidays, trend changes, multiple seasonalities |
| 5 | Holt-Winters | Triple exponential smoothing for seasonal data |
| 6 | LSTM with Attention | Neural sequence model |
| 7 | PatchTST Transformer | State-of-the-art neural architecture |

### Reporting Rules

1. **Always report baseline results alongside DL results** in the same table
2. **Bold the best model for each metric**, even when it is a baseline
3. **Include the Diebold-Mariano test** to determine if differences are statistically significant
4. **Report MASE** so readers can see whether each model beats the naive baseline (MASE < 1) or not
5. **Show per-horizon error curves** to reveal where DL excels (typically longer horizons)

### When We Expect Baselines to Win

Based on forecasting literature and practical experience:

- **Short horizons (1-6 hours)**: SARIMA and Holt-Winters are strong because they exploit autocorrelation structure efficiently
- **Strong single seasonality**: Seasonal naive with the right period is hard to beat
- **Univariate targets**: Statistical methods can be more data-efficient
- **Small datasets**: Neural networks need thousands of examples to generalize

### When We Expect DL to Win

- **Long horizons (48+ hours)**: Transformers can capture long-range dependencies
- **Multi-variate inputs**: LSTM and Transformer can leverage exogenous features that ARIMA cannot
- **Complex non-linear patterns**: Neural networks can model interactions that linear methods miss
- **Large datasets**: With sufficient data, the expressiveness of neural networks pays off

## Consequences

### Positive
- Builds credibility by showing intellectual honesty
- Helps practitioners choose the right model for their specific use case
- Demonstrates understanding of the full forecasting landscape
- Differentiates this project from superficial DL benchmarks

### Negative
- More work to implement and maintain multiple model families
- Results may be less "impressive" when DL does not always win
- Statistical model fitting (especially ARIMA) can be slow

### Mitigations
- The honest comparison IS the differentiator; it shows production experience
- We clearly explain when and why each model class excels
- We provide a decision framework for model selection based on data characteristics
