"""Advanced forecast analysis: per-horizon degradation and statistical significance.

This module answers two critical questions:
1. How does forecast accuracy degrade as we predict further into the future?
2. Are the differences between models statistically significant?
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def per_horizon_analysis(
    actuals: np.ndarray,
    predictions: np.ndarray,
    metric_fn: callable = None,
) -> dict[int, float]:
    """Compute forecast error at each step of the forecast horizon.

    This reveals how quickly a model's accuracy degrades with forecast
    distance. Good models degrade slowly; poor models fall apart quickly.

    Example output for 24-step forecast:
        {1: 0.23, 2: 0.31, 3: 0.38, ..., 24: 1.47}

    Args:
        actuals: Ground truth, shape (n_samples, horizon) or (horizon,).
        predictions: Predicted values, same shape as actuals.
        metric_fn: Metric function(actual, predicted) -> float.
            Defaults to MAE if not specified.

    Returns:
        Dictionary mapping horizon step (1-indexed) to metric value.
    """
    if metric_fn is None:
        metric_fn = lambda a, p: float(np.mean(np.abs(a - p)))

    if actuals.ndim == 1:
        actuals = actuals.reshape(1, -1)
        predictions = predictions.reshape(1, -1)

    horizon = actuals.shape[1]
    results = {}

    for h in range(horizon):
        actual_h = actuals[:, h]
        pred_h = predictions[:, h]
        results[h + 1] = metric_fn(actual_h, pred_h)

    return results


def diebold_mariano_test(
    actual: np.ndarray,
    pred_model1: np.ndarray,
    pred_model2: np.ndarray,
    horizon: int = 1,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Diebold-Mariano test for comparing forecast accuracy.

    Tests the null hypothesis that two forecasts have equal predictive
    accuracy. Uses the modified DM test (Harvey, Leybourne, Newbold, 1997)
    with a correction for small samples.

    This is essential for honest model comparison -- raw metric differences
    can be misleading without statistical significance testing.

    Args:
        actual: Ground truth values.
        pred_model1: Predictions from model 1.
        pred_model2: Predictions from model 2.
        horizon: Forecast horizon (for autocorrelation correction).
        alternative: 'two-sided', 'less' (model 1 better), or 'greater' (model 2 better).

    Returns:
        Dictionary with 'dm_statistic', 'p_value', and 'significant' (at 0.05 level).
    """
    n = len(actual)

    # Loss differential (squared error)
    e1 = (actual - pred_model1) ** 2
    e2 = (actual - pred_model2) ** 2
    d = e1 - e2  # positive means model 2 is better

    # Mean and variance of loss differential
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    # Autocovariance adjustment for multi-step forecasts
    # (Diebold-Mariano use h-1 autocovariances)
    if horizon > 1:
        for k in range(1, horizon):
            autocov = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
            d_var += 2 * autocov

    # DM statistic
    if d_var < 1e-10:
        return {
            "dm_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    dm_stat = d_mean / np.sqrt(d_var / n)

    # Harvey-Leybourne-Newbold correction for small samples
    correction = np.sqrt(
        (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    )
    dm_stat_corrected = dm_stat * correction

    # P-value from t-distribution
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_corrected), df=n - 1))
    elif alternative == "less":
        p_value = stats.t.cdf(dm_stat_corrected, df=n - 1)
    elif alternative == "greater":
        p_value = 1 - stats.t.cdf(dm_stat_corrected, df=n - 1)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    return {
        "dm_statistic": float(dm_stat_corrected),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def model_comparison_table(
    results: dict[str, dict[str, float]],
    metric_names: Optional[list[str]] = None,
) -> str:
    """Generate a formatted comparison table of model results.

    Args:
        results: Dictionary mapping model names to metric dictionaries.
        metric_names: Optional list of metrics to include (default: all).

    Returns:
        Formatted string table for display.
    """
    if not results:
        return "No results to display."

    if metric_names is None:
        metric_names = list(next(iter(results.values())).keys())

    # Build header
    model_col_width = max(len(name) for name in results.keys()) + 2
    col_width = 12

    header = f"{'Model':<{model_col_width}}"
    for metric in metric_names:
        header += f"{metric:>{col_width}}"

    separator = "-" * len(header)

    # Build rows
    rows = []
    for model_name, metrics in results.items():
        row = f"{model_name:<{model_col_width}}"
        for metric in metric_names:
            value = metrics.get(metric, float("nan"))
            if np.isnan(value):
                row += f"{'N/A':>{col_width}}"
            else:
                row += f"{value:>{col_width}.4f}"
        rows.append(row)

    # Mark best (lowest) for each metric
    table = f"\n{separator}\n{header}\n{separator}\n"
    table += "\n".join(rows)
    table += f"\n{separator}\n"

    return table
