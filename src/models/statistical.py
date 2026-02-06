"""Statistical forecasting models: ARIMA, Prophet, and Holt-Winters.

These are the workhorses of production forecasting. In many real-world
scenarios, a well-tuned ARIMA or Prophet model will match or beat a
neural network -- especially with limited data or short horizons.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """Auto-ARIMA forecaster using pmdarima.

    Automatically selects (p, d, q) and optional seasonal (P, D, Q, m)
    parameters via stepwise search. This is the standard approach for
    univariate time series when you need a strong statistical baseline.

    Args:
        seasonal: Whether to fit seasonal ARIMA (SARIMA).
        m: Seasonal period (e.g., 24 for hourly data with daily seasonality).
        max_p: Maximum AR order.
        max_q: Maximum MA order.
        suppress_warnings: Whether to suppress convergence warnings.
    """

    def __init__(
        self,
        seasonal: bool = True,
        m: int = 24,
        max_p: int = 5,
        max_q: int = 5,
        suppress_warnings: bool = True,
    ):
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.suppress_warnings = suppress_warnings
        self.model = None
        self.name = "SARIMA" if seasonal else "ARIMA"

    def fit(self, y: np.ndarray) -> "ARIMAForecaster":
        """Fit auto-ARIMA on training data.

        Args:
            y: Training target values, shape (n_samples,).

        Returns:
            self for method chaining.
        """
        import pmdarima as pm

        logger.info("Fitting %s on %d observations...", self.name, len(y))

        with warnings.catch_warnings():
            if self.suppress_warnings:
                warnings.simplefilter("ignore")

            self.model = pm.auto_arima(
                y,
                seasonal=self.seasonal,
                m=self.m if self.seasonal else 1,
                max_p=self.max_p,
                max_q=self.max_q,
                stepwise=True,
                suppress_warnings=self.suppress_warnings,
                error_action="ignore",
                trace=False,
            )

        order = self.model.order
        seasonal_order = getattr(self.model, "seasonal_order", None)
        logger.info(
            "%s fitted: order=%s, seasonal_order=%s, AIC=%.2f",
            self.name,
            order,
            seasonal_order,
            self.model.aic(),
        )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Generate point forecasts.

        Args:
            horizon: Number of future timesteps to predict.

        Returns:
            Array of shape (horizon,) with point forecasts.
        """
        if self.model is None:
            raise RuntimeError("Must call fit() before predict().")
        return self.model.predict(n_periods=horizon)

    def predict_with_intervals(
        self, horizon: int, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with prediction intervals.

        Args:
            horizon: Number of future timesteps.
            alpha: Significance level for intervals (default 95% CI).

        Returns:
            Tuple of (point_forecast, lower_bound, upper_bound).
        """
        if self.model is None:
            raise RuntimeError("Must call fit() before predict_with_intervals().")
        forecast, conf_int = self.model.predict(
            n_periods=horizon, return_conf_int=True, alpha=alpha
        )
        return forecast, conf_int[:, 0], conf_int[:, 1]

    def update(self, y_new: np.ndarray) -> None:
        """Update model with new observations (for walk-forward validation).

        Args:
            y_new: New observed values to incorporate.
        """
        if self.model is None:
            raise RuntimeError("Must call fit() before update().")
        self.model.update(y_new)


class ProphetForecaster:
    """Facebook Prophet forecaster.

    Prophet excels at data with strong seasonal patterns and holiday effects.
    It is particularly useful when the data has multiple seasonalities
    (daily, weekly, yearly) and when trend changes are important.

    Args:
        yearly_seasonality: Whether to model yearly seasonality.
        weekly_seasonality: Whether to model weekly seasonality.
        daily_seasonality: Whether to model daily seasonality.
        changepoint_prior_scale: Flexibility of trend changepoints (higher = more flexible).
    """

    def __init__(
        self,
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        self.freq: Optional[str] = None
        self.name = "Prophet"

    def fit(self, y: np.ndarray, dates: Optional[np.ndarray] = None) -> "ProphetForecaster":
        """Fit Prophet on training data.

        Args:
            y: Training target values, shape (n_samples,).
            dates: Datetime values for each observation.

        Returns:
            self for method chaining.
        """
        from prophet import Prophet
        import pandas as pd

        logger.info("Fitting Prophet on %d observations...", len(y))

        # Prophet requires a DataFrame with 'ds' and 'y' columns
        if dates is not None:
            ds = pd.to_datetime(dates)
        else:
            # Generate hourly dates starting from a reference point
            ds = pd.date_range(start="2016-07-01", periods=len(y), freq="h")

        df = pd.DataFrame({"ds": ds, "y": y})
        self.freq = "h"

        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )

        # Suppress Prophet's verbose output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df)

        self._last_date = ds.max()
        logger.info("Prophet fitted successfully")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Generate point forecasts.

        Args:
            horizon: Number of future timesteps to predict.

        Returns:
            Array of shape (horizon,) with point forecasts.
        """
        import pandas as pd

        if self.model is None:
            raise RuntimeError("Must call fit() before predict().")

        future = self.model.make_future_dataframe(periods=horizon, freq=self.freq)
        forecast = self.model.predict(future)

        # Return only the forecast for future periods
        return forecast["yhat"].values[-horizon:]

    def predict_with_intervals(
        self, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with prediction intervals.

        Args:
            horizon: Number of future timesteps.

        Returns:
            Tuple of (point_forecast, lower_bound, upper_bound).
        """
        import pandas as pd

        if self.model is None:
            raise RuntimeError("Must call fit() before predict_with_intervals().")

        future = self.model.make_future_dataframe(periods=horizon, freq=self.freq)
        forecast = self.model.predict(future)

        return (
            forecast["yhat"].values[-horizon:],
            forecast["yhat_lower"].values[-horizon:],
            forecast["yhat_upper"].values[-horizon:],
        )


class HoltWintersForecaster:
    """Holt-Winters exponential smoothing forecaster.

    Triple exponential smoothing that captures level, trend, and seasonality.
    Effective for data with clear seasonal patterns and moderate trends.

    Args:
        seasonal_periods: Number of periods in a seasonal cycle.
        trend: Type of trend component ('add' or 'mul').
        seasonal: Type of seasonal component ('add' or 'mul').
        damped_trend: Whether to damp the trend (prevents unrealistic extrapolation).
    """

    def __init__(
        self,
        seasonal_periods: int = 24,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool = True,
    ):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.model = None
        self.fitted_model = None
        self.name = "Holt-Winters"

    def fit(self, y: np.ndarray) -> "HoltWintersForecaster":
        """Fit Holt-Winters on training data.

        Args:
            y: Training target values, shape (n_samples,).

        Returns:
            self for method chaining.
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        logger.info("Fitting Holt-Winters on %d observations...", len(y))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend,
            )
            self.fitted_model = self.model.fit(optimized=True)

        logger.info(
            "Holt-Winters fitted: AIC=%.2f, BIC=%.2f",
            self.fitted_model.aic,
            self.fitted_model.bic,
        )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Generate point forecasts.

        Args:
            horizon: Number of future timesteps to predict.

        Returns:
            Array of shape (horizon,) with point forecasts.
        """
        if self.fitted_model is None:
            raise RuntimeError("Must call fit() before predict().")
        return self.fitted_model.forecast(horizon)
