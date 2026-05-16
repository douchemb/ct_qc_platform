# -*- coding: utf-8 -*-
"""
modules/predictive/trend_model.py — Linear Regression Trend Model.

Fits independent linear regression trend models for multiple QC metrics.
Linear regression is appropriate for X-ray tube degradation because the
degradation process is approximately linear over the useful tube life.

Statistical foundation:
    - OLS regression via sklearn.linear_model.LinearRegression
    - Prediction intervals via Montgomery & Runger Eq. 11.21
    - Two-tailed t-test for slope significance

Standards References:
    - AAPM TG-66 Section 7: QC Trending
    - Montgomery & Runger, Applied Statistics, Section 11.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from config import PredictiveConfig

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = [
    "QCTrendModel",
    "TrendModelResult",
    "InsufficientDataError",
    "ModelFitError",
]


# ═══════════════════════════════════════════════════════════════════
# Custom Exceptions
# ═══════════════════════════════════════════════════════════════════

class InsufficientDataError(ValueError):
    """
    Raised when fewer than 2 data points are available for trend fitting.
    Minimum 2 points are required to define a line.
    At least CONFIG.predictive.min_history_points points are needed for
    a statistically reliable prediction.
    """


class ModelFitError(RuntimeError):
    """
    Raised when the regression model cannot be fitted due to
    numerical issues (e.g., all x-values identical, degenerate matrix).
    """


# ═══════════════════════════════════════════════════════════════════
# TrendModelResult Dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrendModelResult:
    """
    Result of fitting a linear trend to one QC metric time series.

    The linear model: metric_value = slope_per_day × day_ordinal + intercept
    where day_ordinal = 0 at the first observation date.

    Stored fields needed for prediction interval computation:
      x_values:  ordinal-encoded training dates (day_ordinal array)
      y_values:  training metric values
      x_mean:    mean of x_values
      ss_xx:     Σ(xi - x̄)² — used in prediction interval formula
      residual_std: sqrt(Σ(yi - ŷi)² / (n-2)) — residual standard error
    """
    metric_name: str
    n_data_points: int
    slope_per_day: float
    intercept: float
    r_squared: float
    slope_std_error: float
    p_value_slope: float
    first_date: str
    last_date: str
    fit_span_days: int
    is_reliable: bool               # r_squared >= CONFIG threshold
    has_sufficient_data: bool        # n >= CONFIG min_history_points
    trend_direction: str             # "increasing", "decreasing", "stable"
    # Stored for prediction interval computation
    x_values: np.ndarray             # shape (n,) float64
    y_values: np.ndarray             # shape (n,) float64
    x_mean: float
    ss_xx: float
    residual_std: float

    def to_dict(self) -> dict:
        """Serialize — convert numpy arrays to lists."""
        d = {
            "metric_name": self.metric_name,
            "n_data_points": self.n_data_points,
            "slope_per_day": self.slope_per_day,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "slope_std_error": self.slope_std_error,
            "p_value_slope": self.p_value_slope,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "fit_span_days": self.fit_span_days,
            "is_reliable": self.is_reliable,
            "has_sufficient_data": self.has_sufficient_data,
            "trend_direction": self.trend_direction,
            "x_values": self.x_values.tolist(),
            "y_values": self.y_values.tolist(),
            "x_mean": self.x_mean,
            "ss_xx": self.ss_xx,
            "residual_std": self.residual_std,
        }
        return d

    def predict_at_date(self, target_date: str) -> float:
        """
        Predict metric value at a future date.
        target_date: "YYYY-MM-DD" string.
        Returns predicted float.
        """
        epoch = date(1970, 1, 1)
        first_date_ordinal = (date.fromisoformat(self.first_date) - epoch).days
        target_ordinal = (date.fromisoformat(target_date) - epoch).days
        x_new = float(target_ordinal - first_date_ordinal)
        return self.slope_per_day * x_new + self.intercept

    def prediction_interval_at_date(
        self,
        target_date: str,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """
        Returns (lower, upper) prediction interval at target_date.

        Formula (OLS prediction interval):
            ŷ = slope × x_new + intercept
            SE_pred = residual_std × sqrt(1 + 1/n + (x_new - x̄)² / SS_xx)
            PI = ŷ ± t_{α/2, n-2} × SE_pred

        Reference: Montgomery & Runger, Applied Statistics, Section 11.5.
        """
        epoch = date(1970, 1, 1)
        first_date_ordinal = (date.fromisoformat(self.first_date) - epoch).days
        target_ordinal = (date.fromisoformat(target_date) - epoch).days
        x_new = float(target_ordinal - first_date_ordinal)

        y_hat = self.slope_per_day * x_new + self.intercept
        n = self.n_data_points

        if n <= 2 or self.ss_xx < 1e-12:
            # Cannot compute meaningful PI
            return (y_hat, y_hat)

        # Montgomery & Runger Eq. 11.21
        se_pred = self.residual_std * np.sqrt(
            1.0 + 1.0 / n + (x_new - self.x_mean) ** 2 / self.ss_xx
        )
        t_crit = stats.t.ppf((1.0 + confidence) / 2.0, df=n - 2)
        return (y_hat - t_crit * se_pred, y_hat + t_crit * se_pred)


# ═══════════════════════════════════════════════════════════════════
# QC Trend Model Class
# ═══════════════════════════════════════════════════════════════════

class QCTrendModel:
    """
    Fits independent linear regression trend models for multiple QC metrics.
    One model per metric, stored in an internal dict keyed by metric_name.
    """

    def __init__(self, config: PredictiveConfig) -> None:
        self._config = config
        self._models: dict[str, tuple] = {}

    def fit(
        self,
        dates: list[str],
        values: list[float],
        metric_name: str,
    ) -> TrendModelResult:
        """Fit a linear regression trend to one QC metric time series.

        Parameters
        ----------
        dates : list[str]
            Dates in "YYYY-MM-DD" format.
        values : list[float]
            Corresponding metric values.
        metric_name : str
            Name of the metric being fitted.

        Returns
        -------
        TrendModelResult

        Raises
        ------
        InsufficientDataError
            If fewer than 2 data points.
        """
        n = len(dates)
        if n < 2:
            raise InsufficientDataError(
                "At least 2 data points required for trend fitting, got %d" % n
            )

        if n < self._config.min_history_points:
            logger.warning(
                "Metric '%s': only %d data points (recommended minimum: %d). "
                "Prediction reliability is reduced.",
                metric_name, n, self._config.min_history_points,
            )

        # Convert dates to ordinals (day 0 = first observation)
        x = self._dates_to_ordinals(dates)
        y = np.array(values, dtype=np.float64)

        # Fit sklearn LinearRegression
        model = LinearRegression()
        x_reshaped = x.reshape(-1, 1)
        model.fit(x_reshaped, y)

        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        r_squared = float(model.score(x_reshaped, y))

        # Compute predictions and residual statistics
        y_pred = model.predict(x_reshaped)
        slope_std_error, residual_std = self._compute_slope_std_error(x, y, y_pred)

        # Compute p-value for slope
        p_value = self._compute_p_value(slope, slope_std_error, n)

        # Classify trend direction
        trend_direction = self._classify_trend_direction(slope, p_value)

        # Compute remaining fields
        ss_xx = float(np.sum((x - x.mean()) ** 2))
        fit_span_days = int(x[-1] - x[0])

        is_reliable = r_squared >= self._config.r2_minimum_acceptable
        has_sufficient_data = n >= self._config.min_history_points

        # Store model for potential reuse
        self._models[metric_name] = (model, x[0], x)

        result = TrendModelResult(
            metric_name=metric_name,
            n_data_points=n,
            slope_per_day=slope,
            intercept=intercept,
            r_squared=r_squared,
            slope_std_error=slope_std_error,
            p_value_slope=p_value,
            first_date=dates[0],
            last_date=dates[-1],
            fit_span_days=fit_span_days,
            is_reliable=is_reliable,
            has_sufficient_data=has_sufficient_data,
            trend_direction=trend_direction,
            x_values=x,
            y_values=y,
            x_mean=float(x.mean()),
            ss_xx=ss_xx,
            residual_std=residual_std,
        )

        logger.info(
            "Trend fitted for '%s': slope=%+.6f/day, R²=%.4f, p=%.4f, "
            "direction=%s, n=%d, span=%d days",
            metric_name, slope, r_squared, p_value,
            trend_direction, n, fit_span_days,
        )

        return result

    def fit_all_metrics(
        self,
        archive: "MetricsArchive",
        metric_names: Optional[list[str]] = None,
    ) -> dict[str, TrendModelResult]:
        """Fit trends for all tracked metrics in the archive.

        Parameters
        ----------
        archive : MetricsArchive
            Archive containing historical sessions.
        metric_names : list[str], optional
            Metrics to fit. Defaults to the six hardware-tracked metrics.

        Returns
        -------
        dict[str, TrendModelResult]
            Mapping metric name to fitted TrendModelResult.
        """
        if metric_names is None:
            metric_names = [
                self._config.tracked_metric_noise_std,
                self._config.tracked_metric_nps_peak,
                self._config.tracked_metric_mtf50,
                "hu_linearity_max_deviation_hu",
                self._config.tracked_metric_ed_soft_slope,
                self._config.tracked_metric_ed_bone_slope,
            ]

        results: dict[str, TrendModelResult] = {}

        for metric_name in metric_names:
            dates, values = archive.get_metric_series(metric_name)
            if len(dates) < 2:
                logger.warning(
                    "Metric '%s': fewer than 2 data points (%d) — skipping trend fit",
                    metric_name, len(dates),
                )
                continue

            try:
                result = self.fit(dates, values, metric_name)
                results[metric_name] = result
            except (InsufficientDataError, ModelFitError) as exc:
                logger.warning("Failed to fit trend for '%s': %s", metric_name, exc)
                continue

        logger.info(
            "Fitted trends for %d / %d metrics",
            len(results), len(metric_names),
        )
        return results

    # ─────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────

    def _dates_to_ordinals(self, dates: list[str]) -> np.ndarray:
        """Convert date strings to ordinal day numbers, centered at day 0.

        Parameters
        ----------
        dates : list[str]
            Date strings in "YYYY-MM-DD" format.

        Returns
        -------
        np.ndarray
            Ordinal day numbers, shape (n,), float64.
        """
        epoch = date(1970, 1, 1)
        ordinals = np.array([
            (date.fromisoformat(d) - epoch).days
            for d in dates
        ], dtype=np.float64)
        # Center at zero: first observation = day 0
        ordinals = ordinals - ordinals[0]
        return ordinals

    def _compute_slope_std_error(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[float, float]:
        """Compute standard error of slope and residual std.

        Reference: Montgomery & Runger, Applied Statistics, Eq. 11.15.

        Returns
        -------
        tuple[float, float]
            (slope_std_error, residual_std)
        """
        n = len(x)
        residuals = y - y_pred

        # OLS residual variance — ddof=2 (slope and intercept estimated)
        if n <= 2:
            return (0.0, 0.0)

        s_squared = float(np.sum(residuals ** 2)) / (n - 2)
        residual_std = float(np.sqrt(s_squared))

        ss_xx = float(np.sum((x - x.mean()) ** 2))
        if ss_xx < 1e-12:
            raise ModelFitError(
                "All x-values are identical — cannot estimate slope variance"
            )

        slope_std_error = float(np.sqrt(s_squared / ss_xx))
        return slope_std_error, residual_std

    def _compute_p_value(
        self,
        slope: float,
        slope_std_error: float,
        n: int,
    ) -> float:
        """Two-tailed t-test for slope != 0.

        H0: slope = 0, H1: slope != 0
        t = slope / SE(slope), df = n - 2

        Reference: Montgomery & Runger, Section 11.4.2.

        Returns
        -------
        float
            p-value in [0, 1].
        """
        if n <= 2:
            return 1.0
        # When slope is effectively zero, there is no evidence of a trend
        if abs(slope) < 1e-12:
            return 1.0
        if slope_std_error < 1e-12:
            t_stat = float("inf")
        else:
            t_stat = slope / slope_std_error

        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 2))
        return p_value

    def _classify_trend_direction(
        self,
        slope_per_day: float,
        p_value: float,
        significance_level: float = 0.05,
    ) -> str:
        """Classify trend as increasing, decreasing, or stable.

        Parameters
        ----------
        slope_per_day : float
            Slope of the linear regression.
        p_value : float
            p-value of the slope t-test.
        significance_level : float
            Threshold for statistical significance.

        Returns
        -------
        str
            "increasing", "decreasing", or "stable".
        """
        if p_value >= significance_level:
            return "stable"
        elif slope_per_day > 0:
            return "increasing"
        else:
            return "decreasing"
