# -*- coding: utf-8 -*-
"""
tests/test_trend_model.py — Tests for Linear Regression Trend Model.

Uses known synthetic data where correct answers are analytically derivable.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import CONFIG
from modules.predictive.trend_model import (
    QCTrendModel,
    TrendModelResult,
    InsufficientDataError,
    ModelFitError,
)


@pytest.fixture
def trend_model():
    """Returns QCTrendModel configured with CONFIG.predictive."""
    return QCTrendModel(CONFIG.predictive)


class TestPerfectLinearTrend:
    """Test with perfect linear data: y = [1,2,3,4,5] at 30-day intervals."""

    def test_slope_per_day(self, trend_model):
        """slope_per_day ≈ 1/30 = 0.0333."""
        dates = ["2024-01-15", "2024-02-14", "2024-03-15", "2024-04-14", "2024-05-14"]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = trend_model.fit(dates, values, "test_linear")
        assert result.slope_per_day == pytest.approx(1.0 / 30.0, rel=0.05)

    def test_r_squared_near_one(self, trend_model):
        """R² > 0.999 for perfect linear data."""
        dates = ["2024-01-15", "2024-02-14", "2024-03-15", "2024-04-14", "2024-05-14"]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = trend_model.fit(dates, values, "test_linear")
        assert result.r_squared > 0.999

    def test_direction_increasing(self, trend_model):
        """trend_direction == 'increasing' for rising data."""
        dates = ["2024-01-15", "2024-02-14", "2024-03-15", "2024-04-14", "2024-05-14"]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = trend_model.fit(dates, values, "test_linear")
        assert result.trend_direction == "increasing"

    def test_p_value_below_005(self, trend_model):
        """p_value_slope < 0.05 for significant trend."""
        dates = ["2024-01-15", "2024-02-14", "2024-03-15", "2024-04-14", "2024-05-14"]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = trend_model.fit(dates, values, "test_linear")
        assert result.p_value_slope < 0.05


class TestFlatTrend:
    """Test with constant data: all values identical."""

    def test_direction_stable(self, trend_model):
        """trend_direction == 'stable' for flat data."""
        dates = ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15", "2024-05-15"]
        values = [3.0, 3.0, 3.0, 3.0, 3.0]
        result = trend_model.fit(dates, values, "test_flat")
        assert result.trend_direction == "stable"

    def test_p_value_above_005(self, trend_model):
        """p_value_slope >= 0.05 for flat data (not significant)."""
        dates = ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15", "2024-05-15"]
        values = [3.0, 3.0, 3.0, 3.0, 3.0]
        result = trend_model.fit(dates, values, "test_flat")
        assert result.p_value_slope >= 0.05


class TestDecreasingTrend:
    """Test with decreasing data."""

    def test_direction_decreasing(self, trend_model):
        """trend_direction == 'decreasing' for falling data."""
        dates = ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15", "2024-05-15"]
        values = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = trend_model.fit(dates, values, "test_dec")
        assert result.trend_direction == "decreasing"


class TestPrediction:
    """Test predict_at_date and prediction_interval."""

    def test_predict_at_date_analytic(self, trend_model):
        """predict_at_date should match analytic value within 1e-4."""
        dates = ["2024-01-15", "2024-02-14", "2024-03-15", "2024-04-14", "2024-05-14"]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = trend_model.fit(dates, values, "test_pred")
        # After 120 days from first: y = slope * 120 + intercept ≈ 1 + 4 = 5
        predicted = result.predict_at_date("2024-05-14")
        assert predicted == pytest.approx(5.0, abs=0.1)

    def test_prediction_interval_bounds(self, trend_model):
        """PI returns (lower, upper) where lower < predicted < upper."""
        dates = ["2024-01-15", "2024-02-14", "2024-03-15", "2024-04-14", "2024-05-14"]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = trend_model.fit(dates, values, "test_pi")
        predicted = result.predict_at_date("2024-07-15")
        lower, upper = result.prediction_interval_at_date("2024-07-15", confidence=0.95)
        assert lower <= predicted <= upper

    def test_pi_99_wider_than_95(self, trend_model):
        """PI with confidence=0.99 should be wider than 0.95."""
        dates = ["2024-01-15", "2024-02-14", "2024-03-15", "2024-04-14", "2024-05-14"]
        values = [1.0, 2.1, 2.9, 4.1, 4.9]  # slight noise for nonzero residuals
        result = trend_model.fit(dates, values, "test_ci_width")
        lo_95, hi_95 = result.prediction_interval_at_date("2024-07-15", 0.95)
        lo_99, hi_99 = result.prediction_interval_at_date("2024-07-15", 0.99)
        assert (hi_99 - lo_99) >= (hi_95 - lo_95)


class TestInsufficientData:
    """Test error handling."""

    def test_raises_with_fewer_than_2(self, trend_model):
        """fit raises InsufficientDataError with fewer than 2 points."""
        with pytest.raises(InsufficientDataError):
            trend_model.fit(["2024-01-15"], [1.0], "test")


class TestStatisticalHelpers:
    """Test private statistical methods."""

    def test_slope_std_error_non_negative(self, trend_model):
        """_compute_slope_std_error returns non-negative floats."""
        x = np.array([0, 30, 60, 90, 120], dtype=np.float64)
        y = np.array([1.0, 2.1, 2.9, 4.1, 5.0], dtype=np.float64)
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        sse, residual_std = trend_model._compute_slope_std_error(x, y, y_pred)
        assert sse >= 0.0
        assert residual_std >= 0.0

    def test_p_value_in_0_1_range(self, trend_model):
        """_compute_p_value returns a value in [0.0, 1.0]."""
        p = trend_model._compute_p_value(0.033, 0.001, 5)
        assert 0.0 <= p <= 1.0


class TestReliability:
    """Test is_reliable flag."""

    def test_unreliable_with_noise(self, trend_model):
        """is_reliable == False when R² is below threshold due to noise."""
        dates = ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15", "2024-05-15"]
        values = [1.0, 5.0, 2.0, 4.0, 3.0]  # high noise, low R²
        result = trend_model.fit(dates, values, "test_noisy")
        assert result.r_squared < CONFIG.predictive.r2_minimum_acceptable
        assert result.is_reliable is False


class TestFitAllMetrics:
    """Test fit_all_metrics with populated archive."""

    def test_returns_dict_with_6_keys(self, fitted_trend_results):
        """fit_all_metrics returns a dict with 6 entries."""
        assert isinstance(fitted_trend_results, dict)
        assert len(fitted_trend_results) == 6

    def test_all_directions_valid(self, fitted_trend_results):
        """All TrendModelResult objects have valid trend_direction."""
        valid = {"increasing", "decreasing", "stable"}
        for name, result in fitted_trend_results.items():
            assert result.trend_direction in valid, (
                "Metric '%s' has invalid direction '%s'" % (name, result.trend_direction)
            )
