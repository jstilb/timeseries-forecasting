# ADR 001: Walk-Forward Validation over K-Fold Cross-Validation

## Status
Accepted

## Context

Time series forecasting requires a fundamentally different evaluation strategy than cross-sectional machine learning. The standard k-fold cross-validation randomly shuffles data into folds, which destroys temporal ordering and creates data leakage: the model trains on future data to predict the past.

We need an evaluation strategy that:
1. Respects temporal causality (train on past, predict future)
2. Simulates real-world deployment conditions
3. Provides reliable estimates of production performance
4. Allows statistical comparison between models

## Decision

We use **walk-forward validation** with expanding windows as the primary evaluation strategy.

### How It Works

```
Time ------>

Fold 1: [==TRAIN==][TEST]................
Fold 2: [====TRAIN====][TEST]...........
Fold 3: [======TRAIN======][TEST].......
Fold 4: [=========TRAIN=========][TEST].
```

At each fold:
1. Train (or re-fit) the model on all available historical data
2. Generate forecasts for the next `test_size` timesteps
3. Evaluate forecasts against actual values
4. Advance the window and repeat

### Why Expanding vs. Sliding

We default to expanding windows because:
- More training data generally improves model performance
- It matches how most production systems operate (you would not throw away old data)
- It better captures regime changes by including all historical context

We also support sliding windows for comparison, because:
- Some series have concept drift (old data hurts performance)
- It provides a more conservative estimate of model capability
- It tests the model's ability to generalize from a fixed data budget

## Consequences

### Positive
- Evaluation metrics accurately reflect production performance
- No data leakage from future to past
- Statistical significance can be computed across folds
- Results are reproducible and comparable across models

### Negative
- Computationally more expensive than a single train/test split
- Statistical models must be re-fit at each fold (slow for ARIMA)
- Deep learning models must be retrained from scratch at each fold (very slow)
- Fewer effective test samples than k-fold (only forward-looking folds)

### Mitigations
- For deep learning models, we train once on the full training set and evaluate on the held-out test set, using walk-forward only for the final published results
- ARIMA uses `update()` to incrementally incorporate new observations rather than re-fitting from scratch
- We cache intermediate results to avoid redundant computation

## Alternatives Considered

1. **K-Fold Cross-Validation**: Rejected due to temporal leakage.
2. **Single Train/Test Split**: Too dependent on split point; one unlucky split can be misleading.
3. **Time Series Cross-Validation (sklearn)**: Similar to walk-forward but less flexible; does not support the gap parameter needed to prevent leakage from feature engineering.
4. **Blocked Time Series Split**: Partitions data into contiguous blocks without overlap. Considered but rejected because it wastes data by not expanding the training set.
