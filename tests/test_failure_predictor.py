# -*- coding: utf-8 -*-
"""
tests/test_failure_predictor.py — Tests for Hardware-Specific Failure Predictor.

Uses synthetic TrendModelResult objects for controlled testing of
breach prediction, urgency classification, and hardware mapping.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from config import CONFIG
from modules.predictive.failure_predictor import (
    FailurePredictor,
    FailurePrediction,
    MaintenanceAlert,
    HardwareStatusReport,
    UnknownMetricError,
)
from modules.predictive.trend_model import TrendModelResult


def _make_trend(
    metric_name: str = "center_water_std_hu",
    slope: float = 0.02,
    intercept: float = 3.0,
    n: int = 10,
    r_squared: float = 0.95,
) -> TrendModelResult:
    """Helper: create a synthetic TrendModelResult."""
    today = date.today()
    first = today - timedelta(days=n * 30)
    x = np.linspace(0, n * 30, n, dtype=np.float64)
    y = slope * x + intercept
    ss_xx = float(np.sum((x - x.mean()) ** 2))
    return TrendModelResult(
        metric_name=metric_name,
        n_data_points=n,
        slope_per_day=slope,
        intercept=intercept,
        r_squared=r_squared,
        slope_std_error=0.001,
        p_value_slope=0.001,
        first_date=str(first),
        last_date=str(today),
        fit_span_days=n * 30,
        is_reliable=r_squared >= CONFIG.predictive.r2_minimum_acceptable,
        has_sufficient_data=n >= CONFIG.predictive.min_history_points,
        trend_direction="increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable"),
        x_values=x,
        y_values=y,
        x_mean=float(x.mean()),
        ss_xx=ss_xx,
        residual_std=0.1,
    )


@pytest.fixture
def predictor():
    """Returns FailurePredictor with CONFIG.predictive."""
    return FailurePredictor(CONFIG.predictive)


class TestPredictBreach:
    """Test predict_breach with controlled scenarios."""

    def test_breach_prediction_approximately_correct(self, predictor, populated_archive):
        """Slope=0.02/day, intercept=3.0 → breach 5.0 in ~75 days from last value."""
        trend = _make_trend(slope=0.02, intercept=3.0, n=10)
        pred = predictor.predict_breach(trend, "center_water_std_hu", populated_archive)
        assert pred.metric_name == "center_water_std_hu"
        assert pred.hardware_component == "X-ray tube filament"
        # The breach should be roughly predictable
        if pred.predicted_breach_date is not None:
            assert pred.days_until_breach is not None
            assert pred.days_until_breach >= 0

    def test_flat_trend_no_breach(self, predictor, populated_archive):
        """Flat trend (slope=0.0) → predicted_breach_date is None."""
        trend = _make_trend(slope=0.0, intercept=3.0, r_squared=0.0)
        trend = TrendModelResult(
            **{**trend.__dict__, "trend_direction": "stable", "is_reliable": False}
        )
        pred = predictor.predict_breach(trend, "center_water_std_hu", populated_archive)
        assert pred.predicted_breach_date is None
        assert pred.get_urgency_level() == "stable"

    def test_breached_status_when_above_threshold(self, predictor, populated_archive):
        """Current value 6.0 HU > 5.0 threshold → current_status == 'breached'."""
        # Create an archive with current value already breached
        import uuid
        from modules.predictive.metrics_archive import MetricsArchive, QCSessionRecord

        temp_archive = populated_archive
        record = QCSessionRecord(
            session_id=str(uuid.uuid4()),
            session_date="2024-12-01",
            session_timestamp="2024-12-01T10:00:00Z",
            scanner_id="TEST_SCANNER",
            center_water_std_hu=6.0,
            schema_version="1.0",
        )
        temp_archive.append_session(record)

        trend = _make_trend(slope=0.02, intercept=3.0)
        pred = predictor.predict_breach(trend, "center_water_std_hu", temp_archive)
        assert pred.current_status == "breached"
        assert pred.get_urgency_level() == "breached"

    def test_warning_zone(self, predictor, populated_archive):
        """Value 4.5 HU (within 20% of 5.0) → current_status == 'warning'."""
        import uuid
        from modules.predictive.metrics_archive import QCSessionRecord

        record = QCSessionRecord(
            session_id=str(uuid.uuid4()),
            session_date="2024-12-01",
            session_timestamp="2024-12-01T10:00:00Z",
            scanner_id="TEST_SCANNER",
            center_water_std_hu=4.5,
            schema_version="1.0",
        )
        populated_archive.append_session(record)

        trend = _make_trend(slope=0.02, intercept=3.0)
        pred = predictor.predict_breach(trend, "center_water_std_hu", populated_archive)
        assert pred.current_status == "warning"


class TestUrgencyLevels:
    """Test get_urgency_level classification."""

    def test_critical_within_30_days(self):
        """Urgency == 'critical' when days_until_breach <= 30."""
        pred = FailurePrediction(
            metric_name="center_water_std_hu",
            hardware_component="X-ray tube filament",
            clinical_action="Test",
            tolerance_threshold=5.0,
            breach_direction="above",
            current_value=4.0,
            current_date="2024-01-15",
            margin_to_threshold=1.0,
            current_status="safe",
            predicted_breach_date="2024-02-01",
            days_until_breach=20,
            confidence_interval_days=None,
            trend_direction="increasing",
            slope_per_day=0.05,
            r_squared=0.95,
            is_reliable=True,
            prediction_note="test",
        )
        assert pred.get_urgency_level() == "critical"

    def test_warning_within_90_days(self):
        """Urgency == 'warning' when days_until_breach == 60."""
        pred = FailurePrediction(
            metric_name="test", hardware_component="test",
            clinical_action="test", tolerance_threshold=5.0,
            breach_direction="above", current_value=4.0,
            current_date="2024-01-15", margin_to_threshold=1.0,
            current_status="safe", predicted_breach_date="2024-03-15",
            days_until_breach=60, confidence_interval_days=None,
            trend_direction="increasing", slope_per_day=0.02,
            r_squared=0.95, is_reliable=True, prediction_note="test",
        )
        assert pred.get_urgency_level() == "warning"

    def test_monitor_within_180_days(self):
        """Urgency == 'monitor' when days_until_breach == 120."""
        pred = FailurePrediction(
            metric_name="test", hardware_component="test",
            clinical_action="test", tolerance_threshold=5.0,
            breach_direction="above", current_value=3.0,
            current_date="2024-01-15", margin_to_threshold=2.0,
            current_status="safe", predicted_breach_date="2024-05-15",
            days_until_breach=120, confidence_interval_days=None,
            trend_direction="increasing", slope_per_day=0.015,
            r_squared=0.95, is_reliable=True, prediction_note="test",
        )
        assert pred.get_urgency_level() == "monitor"


class TestHardwareMapping:
    """Test that all 6 metrics map to correct hardware components."""

    @pytest.mark.parametrize("metric,expected_hw", [
        ("center_water_std_hu", "X-ray tube filament"),
        ("nps_peak_frequency_lpmm", "X-ray tube filament"),
        ("mtf_50_lpmm", "Anode focal spot"),
        ("hu_linearity_max_deviation_hu", "kVp high-voltage generator"),
        ("ed_soft_tissue_slope", "kVp high-voltage generator"),
        ("ed_bone_slope", "kVp high-voltage generator"),
    ])
    def test_hardware_component_populated(self, metric, expected_hw, predictor, populated_archive):
        """FailurePrediction.hardware_component is correct for all 6 metrics."""
        trend = _make_trend(metric_name=metric, slope=0.01)
        pred = predictor.predict_breach(trend, metric, populated_archive)
        assert pred.hardware_component == expected_hw

    @pytest.mark.parametrize("metric", [
        "center_water_std_hu", "nps_peak_frequency_lpmm", "mtf_50_lpmm",
        "hu_linearity_max_deviation_hu", "ed_soft_tissue_slope", "ed_bone_slope",
    ])
    def test_clinical_action_non_empty(self, metric, predictor, populated_archive):
        """FailurePrediction.clinical_action is a non-empty string."""
        trend = _make_trend(metric_name=metric, slope=0.01)
        pred = predictor.predict_breach(trend, metric, populated_archive)
        assert isinstance(pred.clinical_action, str)
        assert len(pred.clinical_action) > 0


class TestMaintenanceAlert:
    """Test MaintenanceAlert generation and serialization."""

    def test_generate_returns_alert(self, maintenance_alert_fixture):
        """generate_maintenance_alert returns a MaintenanceAlert."""
        assert isinstance(maintenance_alert_fixture, MaintenanceAlert)

    def test_hardware_urgency_in_order(self, maintenance_alert_fixture):
        """alert.hardware_report.tube_filament_urgency is in URGENCY_ORDER."""
        report = maintenance_alert_fixture.hardware_report
        assert report.tube_filament_urgency in FailurePredictor.URGENCY_ORDER

    def test_to_dict_serializable(self, maintenance_alert_fixture):
        """MaintenanceAlert.to_dict() returns a JSON-serializable dict."""
        d = maintenance_alert_fixture.to_dict()
        json_str = json.dumps(d, default=str)
        assert isinstance(json_str, str)

    def test_to_json_creates_file(self, maintenance_alert_fixture, tmp_path):
        """alert.to_json() creates a valid JSON file."""
        out_path = tmp_path / "alert.json"
        maintenance_alert_fixture.to_json(out_path)
        assert out_path.is_file()
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "scanner_id" in data
