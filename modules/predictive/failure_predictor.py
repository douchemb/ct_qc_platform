# -*- coding: utf-8 -*-
"""
modules/predictive/failure_predictor.py — Hardware-Specific Failure Predictor.

Predicts metric threshold-crossing dates and maps predictions
to specific hardware components requiring maintenance.

Hardware mapping table encodes the supervisor requirement:
each metric drift is associated with one hardware component
and one clinical action.

Standards References:
    - AAPM TG-66: CT QC Tolerances
    - AAPM Report 204: SSDE
    - IEC 60336: Focal Spot Measurement
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from config import PredictiveConfig
from modules.predictive.trend_model import TrendModelResult

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = [
    "FailurePredictor",
    "FailurePrediction",
    "HardwareStatusReport",
    "MaintenanceAlert",
    "UnknownMetricError",
    "NoPredictionPossibleError",
]


# ═══════════════════════════════════════════════════════════════════
# Custom Exceptions
# ═══════════════════════════════════════════════════════════════════

class UnknownMetricError(KeyError):
    """Raised when a metric name is not in the METRIC_HARDWARE_MAP."""


class NoPredictionPossibleError(RuntimeError):
    """
    Raised when prediction is requested but the model is unreliable
    AND the metric is already in a breached state requiring immediate action.
    """


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FailurePrediction:
    """
    Prediction of when one QC metric will breach its tolerance threshold.

    hardware_component and clinical_action are populated from METRIC_HARDWARE_MAP,
    making each prediction directly actionable by the medical physicist.
    """
    metric_name: str
    hardware_component: str
    clinical_action: str
    tolerance_threshold: float
    breach_direction: str               # "above" or "below"

    # Current state
    current_value: float
    current_date: str
    margin_to_threshold: float
    current_status: str                 # "safe", "warning", "breached"

    # Prediction
    predicted_breach_date: Optional[str]
    days_until_breach: Optional[int]
    confidence_interval_days: Optional[tuple[int, int]]

    # Model metadata
    trend_direction: str
    slope_per_day: float
    r_squared: float
    is_reliable: bool
    prediction_note: str

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return asdict(self)

    def get_urgency_level(self) -> str:
        """
        Compute urgency level based on current status and prediction.

        Returns
        -------
        str
            One of: "breached", "critical", "warning", "monitor",
                    "improving", "stable"
        """
        if self.current_status == "breached":
            return "breached"
        if not self.is_reliable or self.predicted_breach_date is None:
            if self.trend_direction == "improving":
                return "improving"
            return "stable"
        if self.days_until_breach is not None:
            if self.days_until_breach <= 30:
                return "critical"
            elif self.days_until_breach <= 90:
                return "warning"
            elif self.days_until_breach <= 180:
                return "monitor"
        if self.trend_direction in ("decreasing", "increasing"):
            return "monitor"
        return "stable"


@dataclass
class HardwareStatusReport:
    """
    Hardware-grouped summary of all failure predictions.
    Groups predictions by hardware component for clear clinical communication.
    """
    scanner_id: str
    report_date: str
    # Grouped by hardware component
    tube_filament_predictions: list[FailurePrediction]
    focal_spot_predictions: list[FailurePrediction]
    kvp_generator_predictions: list[FailurePrediction]
    # Overall per-component urgency
    tube_filament_urgency: str
    focal_spot_urgency: str
    kvp_generator_urgency: str
    # Overall scanner status
    overall_urgency: str
    recommended_action: str
    recommended_maintenance_date: Optional[str]

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "scanner_id": self.scanner_id,
            "report_date": self.report_date,
            "tube_filament_predictions": [p.to_dict() for p in self.tube_filament_predictions],
            "focal_spot_predictions": [p.to_dict() for p in self.focal_spot_predictions],
            "kvp_generator_predictions": [p.to_dict() for p in self.kvp_generator_predictions],
            "tube_filament_urgency": self.tube_filament_urgency,
            "focal_spot_urgency": self.focal_spot_urgency,
            "kvp_generator_urgency": self.kvp_generator_urgency,
            "overall_urgency": self.overall_urgency,
            "recommended_action": self.recommended_action,
            "recommended_maintenance_date": self.recommended_maintenance_date,
        }

    def to_json(self, output_path: Path) -> None:
        """Atomic JSON write."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(output_path)
        logger.info("Hardware status report saved to: %s", output_path)


@dataclass
class MaintenanceAlert:
    """
    Top-level alert object returned by generate_maintenance_alert().
    Contains all predictions and the hardware status report.
    """
    scanner_id: str
    alert_date: str
    overall_urgency: str
    predictions: list[FailurePrediction]
    hardware_report: HardwareStatusReport
    recommended_action: str
    recommended_maintenance_date: Optional[str]

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "scanner_id": self.scanner_id,
            "alert_date": self.alert_date,
            "overall_urgency": self.overall_urgency,
            "predictions": [p.to_dict() for p in self.predictions],
            "hardware_report": self.hardware_report.to_dict(),
            "recommended_action": self.recommended_action,
            "recommended_maintenance_date": self.recommended_maintenance_date,
        }

    def to_json(self, output_path: Path) -> None:
        """Atomic JSON write."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(output_path)
        logger.info("Maintenance alert saved to: %s", output_path)


# ═══════════════════════════════════════════════════════════════════
# Failure Predictor Class
# ═══════════════════════════════════════════════════════════════════

class FailurePredictor:
    """
    Predicts metric threshold-crossing dates and maps predictions
    to specific hardware components requiring maintenance.

    Hardware mapping table encodes the supervisor requirement:
    each metric drift is associated with one hardware component
    and one clinical action.
    """

    # (threshold, breach_direction, hardware_component, clinical_action)
    # All thresholds sourced from AAPM TG-66 or clinical convention
    METRIC_HARDWARE_MAP: dict[str, tuple[float, str, str, str]] = {
        "center_water_std_hu": (
            5.0, "above",
            "X-ray tube filament",
            "Schedule filament inspection and tube output consistency measurement. "
            "Increasing noise SD indicates non-uniform electron emission from aging filament.",
        ),
        "nps_peak_frequency_lpmm": (
            0.15, "below",
            "X-ray tube filament",
            "Low-frequency NPS peak shift indicates increasing low-frequency noise "
            "from filament emission non-uniformity. Schedule tube evaluation.",
        ),
        "mtf_50_lpmm": (
            0.3, "below",
            "Anode focal spot",
            "Decreasing MTF50 indicates focal spot blooming from anode surface "
            "roughening. Schedule focal spot size measurement (IEC 60336).",
        ),
        "hu_linearity_max_deviation_hu": (
            4.0, "above",
            "kVp high-voltage generator",
            "Growing HU linearity deviation indicates inconsistent kVp output. "
            "Schedule kVp accuracy measurement and generator calibration.",
        ),
        "ed_soft_tissue_slope": (
            0.02, "above",
            "kVp high-voltage generator",
            "Soft-tissue ED calibration slope drift indicates beam quality change "
            "at low energies. Verify kVp accuracy and repeat stoichiometric calibration.",
        ),
        "ed_bone_slope": (
            0.02, "above",
            "kVp high-voltage generator",
            "Bone ED calibration slope drift indicates beam quality change "
            "at high energies. Verify kVp accuracy and repeat stoichiometric calibration.",
        ),
    }

    # Urgency ordering for comparison
    URGENCY_ORDER = ["stable", "improving", "monitor", "warning", "critical", "breached"]

    def __init__(self, config: PredictiveConfig) -> None:
        self._config = config

    def predict_breach(
        self,
        trend_result: TrendModelResult,
        metric_name: str,
        archive: "MetricsArchive",
    ) -> FailurePrediction:
        """Predict when a metric will breach its tolerance threshold.

        Parameters
        ----------
        trend_result : TrendModelResult
            Fitted trend for this metric.
        metric_name : str
            Name of the metric.
        archive : MetricsArchive
            Archive for retrieving current value.

        Returns
        -------
        FailurePrediction

        Raises
        ------
        UnknownMetricError
            If metric_name is not in METRIC_HARDWARE_MAP.
        """
        # Look up hardware info
        if metric_name not in self.METRIC_HARDWARE_MAP:
            raise UnknownMetricError(
                "Metric '%s' is not in METRIC_HARDWARE_MAP" % metric_name
            )

        threshold, direction, hardware, action = self.METRIC_HARDWARE_MAP[metric_name]

        # Get latest value
        latest = archive.get_latest_session()
        if latest is not None:
            current_value = latest.get_metric(metric_name)
            current_date = latest.session_date
        else:
            current_value = None
            current_date = str(date.today())

        if current_value is None:
            current_value = trend_result.intercept + trend_result.slope_per_day * trend_result.x_values[-1]
            current_date = trend_result.last_date

        # Classify current status
        status, margin = self._classify_current_status(
            current_value, threshold, direction, metric_name
        )

        # Prediction
        predicted_breach_date = None
        days_until_breach = None
        ci_days = None

        if trend_result.is_reliable:
            predicted_breach_date = self._solve_for_breach_date(
                trend_result, threshold, direction, metric_name
            )
            if predicted_breach_date is not None:
                today = date.today()
                breach_date = date.fromisoformat(predicted_breach_date)
                days_until_breach = (breach_date - today).days
                if days_until_breach < 0:
                    days_until_breach = 0
                ci_days = self._compute_breach_confidence_interval(
                    trend_result, threshold, predicted_breach_date
                )
        else:
            logger.warning(
                "Metric '%s': model unreliable (R²=%.4f, n=%d) — "
                "no prediction generated",
                metric_name, trend_result.r_squared, trend_result.n_data_points,
            )

        # Determine trend direction relative to threshold for "improving" check
        actual_trend = trend_result.trend_direction
        if metric_name in ("ed_soft_tissue_slope", "ed_bone_slope"):
            # For ED slopes, "improving" = trending toward 1.0
            if current_value > 1.0 and trend_result.slope_per_day < 0:
                actual_trend = "improving"
            elif current_value < 1.0 and trend_result.slope_per_day > 0:
                actual_trend = "improving"
        else:
            if direction == "above" and trend_result.slope_per_day < 0:
                actual_trend = "improving"
            elif direction == "below" and trend_result.slope_per_day > 0:
                actual_trend = "improving"

        # Build prediction note
        prediction = FailurePrediction(
            metric_name=metric_name,
            hardware_component=hardware,
            clinical_action=action,
            tolerance_threshold=threshold,
            breach_direction=direction,
            current_value=current_value,
            current_date=current_date,
            margin_to_threshold=margin,
            current_status=status,
            predicted_breach_date=predicted_breach_date,
            days_until_breach=days_until_breach,
            confidence_interval_days=ci_days,
            trend_direction=actual_trend,
            slope_per_day=trend_result.slope_per_day,
            r_squared=trend_result.r_squared,
            is_reliable=trend_result.is_reliable,
            prediction_note="",
        )

        prediction.prediction_note = self._build_prediction_note(
            metric_name, trend_result, prediction, hardware
        )

        return prediction

    def generate_maintenance_alert(
        self,
        trend_results: dict[str, TrendModelResult],
        archive: "MetricsArchive",
        scanner_id: str,
    ) -> MaintenanceAlert:
        """Generate a complete maintenance alert from all trend results.

        Parameters
        ----------
        trend_results : dict[str, TrendModelResult]
            Fitted trends for each metric.
        archive : MetricsArchive
            Archive for current values.
        scanner_id : str
            Scanner identifier.

        Returns
        -------
        MaintenanceAlert
        """
        today_str = str(date.today())

        predictions: list[FailurePrediction] = []
        for name, tr in trend_results.items():
            try:
                pred = self.predict_breach(tr, name, archive)
                predictions.append(pred)
            except UnknownMetricError:
                logger.warning("Skipping unknown metric '%s'", name)
                continue

        hardware_report = self._build_hardware_report(predictions, scanner_id)

        overall_urgency = hardware_report.overall_urgency
        recommended_action = self._select_recommended_action(overall_urgency)
        recommended_date = hardware_report.recommended_maintenance_date

        alert = MaintenanceAlert(
            scanner_id=scanner_id,
            alert_date=today_str,
            overall_urgency=overall_urgency,
            predictions=predictions,
            hardware_report=hardware_report,
            recommended_action=recommended_action,
            recommended_maintenance_date=recommended_date,
        )

        logger.info(
            "Maintenance alert generated: urgency=%s, predictions=%d, scanner=%s",
            overall_urgency, len(predictions), scanner_id,
        )

        return alert

    # ─────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────

    def _classify_current_status(
        self,
        current_value: float,
        threshold: float,
        breach_direction: str,
        metric_name: str,
    ) -> tuple[str, float]:
        """Classify the current status of a metric.

        Returns
        -------
        tuple[str, float]
            (status_string, margin_value)
        """
        # Special handling for ED slope metrics (drift from 1.0)
        if metric_name in ("ed_soft_tissue_slope", "ed_bone_slope"):
            deviation = abs(current_value - 1.0)
            margin = threshold - deviation  # positive = safe
            if deviation >= threshold:
                return "breached", margin
            elif deviation >= 0.8 * threshold:
                return "warning", margin
            else:
                return "safe", margin

        # Standard directional check
        if breach_direction == "above":
            margin = threshold - current_value  # positive = safe
            if current_value >= threshold:
                return "breached", margin
            elif abs(margin) < 0.2 * threshold:
                return "warning", margin
            else:
                return "safe", margin
        else:  # "below"
            margin = current_value - threshold  # positive = safe
            if current_value <= threshold:
                return "breached", margin
            elif abs(margin) < 0.2 * threshold:
                return "warning", margin
            else:
                return "safe", margin

    def _solve_for_breach_date(
        self,
        trend: TrendModelResult,
        threshold: float,
        breach_direction: str,
        metric_name: str = "",
    ) -> Optional[str]:
        """Solve for the date when the trend crosses the threshold.

        Returns
        -------
        Optional[str]
            "YYYY-MM-DD" or None.
        """
        slope = trend.slope_per_day
        intercept = trend.intercept

        if abs(slope) < 1e-12:
            return None  # Flat trend, never crosses

        # For ED slope metrics: solve for when |predicted - 1.0| = threshold
        if metric_name in ("ed_soft_tissue_slope", "ed_bone_slope"):
            # predicted(x) = slope * x + intercept
            # We want |slope*x + intercept - 1.0| = threshold
            # Two solutions: slope*x + intercept = 1.0 + threshold
            #             or slope*x + intercept = 1.0 - threshold
            x1 = (1.0 + threshold - intercept) / slope
            x2 = (1.0 - threshold - intercept) / slope
            # Choose the earliest positive solution
            candidates = [x for x in [x1, x2] if x > 0]
            if not candidates:
                return None
            x_breach = min(candidates)
        else:
            # Standard: solve slope * x + intercept = threshold
            x_breach = (threshold - intercept) / slope

            # Check if trend is moving toward threshold
            if breach_direction == "above" and slope <= 0:
                return None  # Moving away
            if breach_direction == "below" and slope >= 0:
                return None  # Moving away

        if x_breach < 0:
            return None  # Already breached in the past

        # Convert to calendar date
        first_date = date.fromisoformat(trend.first_date)
        breach_date = first_date + timedelta(days=int(x_breach))

        # Check against forecast horizon
        today = date.today()
        days_from_today = (breach_date - today).days
        if days_from_today > self._config.forecast_horizon_days:
            logger.debug(
                "Predicted breach for '%s' is %d days out (beyond %d-day horizon)",
                metric_name if metric_name else "metric",
                days_from_today,
                self._config.forecast_horizon_days,
            )
            return None

        return str(breach_date)

    def _compute_breach_confidence_interval(
        self,
        trend: TrendModelResult,
        threshold: float,
        predicted_breach_date: str,
    ) -> Optional[tuple[int, int]]:
        """Compute confidence interval for breach date.

        Returns
        -------
        Optional[tuple[int, int]]
            (days_lower, days_upper) relative to today.
        """
        if trend.n_data_points <= 2 or trend.ss_xx < 1e-12:
            return None

        try:
            lower, upper = trend.prediction_interval_at_date(
                predicted_breach_date, confidence=0.95
            )
        except Exception:
            return None

        # Solve for when the PI bounds cross the threshold
        # Use simple linear extrapolation of bounds
        slope = trend.slope_per_day
        if abs(slope) < 1e-12:
            return None

        today = date.today()
        breach_date = date.fromisoformat(predicted_breach_date)
        center_days = (breach_date - today).days

        # Estimate spread: how many days earlier/later could breach occur
        # based on the PI width at the breach point
        pi_half_width = (upper - lower) / 2.0
        if abs(slope) > 1e-12:
            day_spread = int(pi_half_width / abs(slope))
        else:
            day_spread = 0

        lower_days = max(0, center_days - day_spread)
        upper_days = center_days + day_spread

        return (lower_days, upper_days)

    def _build_prediction_note(
        self,
        metric_name: str,
        trend: TrendModelResult,
        prediction: FailurePrediction,
        hardware_component: str,
    ) -> str:
        """Generate a clinically actionable prediction note.

        Returns
        -------
        str
            Human-readable, actionable explanation.
        """
        parts = []

        # Direction description
        direction_str = trend.trend_direction.upper()
        parts.append(
            "%s is %s at %+.6f/day (R²=%.4f)."
            % (metric_name, direction_str, trend.slope_per_day, trend.r_squared)
        )

        if not trend.is_reliable:
            parts.append(
                "Insufficient data for reliable prediction (R²=%.4f, "
                "n=%d points). Collect at least %d sessions before acting "
                "on this forecast."
                % (trend.r_squared, trend.n_data_points, self._config.min_history_points)
            )
        elif prediction.predicted_breach_date is not None:
            parts.append(
                "At this rate, the %.3f limit will be reached in approximately "
                "%d days (around %s)."
                % (
                    prediction.tolerance_threshold,
                    prediction.days_until_breach or 0,
                    prediction.predicted_breach_date,
                )
            )
            parts.append(
                "Hardware component at risk: %s." % hardware_component
            )
            parts.append(
                "Recommended action: %s" % prediction.clinical_action
            )
        elif prediction.current_status == "breached":
            parts.append(
                "ALERT: Metric has ALREADY EXCEEDED the tolerance threshold of %.3f. "
                "IMMEDIATE ACTION REQUIRED."
                % prediction.tolerance_threshold
            )
            parts.append(
                "Hardware component: %s." % hardware_component
            )
        else:
            parts.append(
                "Current trend is stable (p=%.4f). No breach predicted "
                "within %d days. Continue standard QC schedule."
                % (trend.p_value_slope, self._config.forecast_horizon_days)
            )

        return " ".join(parts)

    def _build_hardware_report(
        self,
        predictions: list[FailurePrediction],
        scanner_id: str,
    ) -> HardwareStatusReport:
        """Group predictions by hardware component.

        Returns
        -------
        HardwareStatusReport
        """
        today_str = str(date.today())

        tube_preds = [p for p in predictions if p.hardware_component == "X-ray tube filament"]
        focal_preds = [p for p in predictions if p.hardware_component == "Anode focal spot"]
        kvp_preds = [p for p in predictions if p.hardware_component == "kVp high-voltage generator"]

        tube_urgency = self._highest_urgency(
            [p.get_urgency_level() for p in tube_preds]
        ) if tube_preds else "stable"

        focal_urgency = self._highest_urgency(
            [p.get_urgency_level() for p in focal_preds]
        ) if focal_preds else "stable"

        kvp_urgency = self._highest_urgency(
            [p.get_urgency_level() for p in kvp_preds]
        ) if kvp_preds else "stable"

        overall = self._highest_urgency([tube_urgency, focal_urgency, kvp_urgency])
        recommended_action = self._select_recommended_action(overall)

        # Find earliest critical breach date
        critical_dates = [
            p.predicted_breach_date for p in predictions
            if p.get_urgency_level() in ("critical", "breached")
            and p.predicted_breach_date is not None
        ]
        recommended_date = min(critical_dates) if critical_dates else None

        return HardwareStatusReport(
            scanner_id=scanner_id,
            report_date=today_str,
            tube_filament_predictions=tube_preds,
            focal_spot_predictions=focal_preds,
            kvp_generator_predictions=kvp_preds,
            tube_filament_urgency=tube_urgency,
            focal_spot_urgency=focal_urgency,
            kvp_generator_urgency=kvp_urgency,
            overall_urgency=overall,
            recommended_action=recommended_action,
            recommended_maintenance_date=recommended_date,
        )

    def _select_recommended_action(self, overall_urgency: str) -> str:
        """Return recommended action string for the given urgency level.

        Returns
        -------
        str
            Clinical action string.
        """
        actions = {
            "breached": (
                "IMMEDIATE ACTION REQUIRED: One or more QC metrics have exceeded "
                "normative tolerances. The CT simulator must be suspended from "
                "clinical service until recalibration is verified and documented."
            ),
            "critical": (
                "URGENT: Predictive model forecasts threshold breach within 30 days. "
                "Contact service engineer and schedule preventive maintenance immediately. "
                "Do not wait for next scheduled QC session."
            ),
            "warning": (
                "ADVISORY: Degradation trend detected across one or more hardware "
                "components. Schedule maintenance within 60 days. Increase QC session "
                "frequency to weekly to monitor trend acceleration."
            ),
            "monitor": (
                "MONITOR: Slow degradation trend detected. No immediate action required. "
                "Ensure next QC session occurs within 30 days to update the predictive model."
            ),
        }
        return actions.get(
            overall_urgency,
            "NO ACTION REQUIRED: All tracked hardware metrics are stable or improving. "
            "Continue the standard monthly QC schedule.",
        )

    def _highest_urgency(self, urgencies: list[str]) -> str:
        """Return the highest urgency from a list.

        Returns
        -------
        str
            Highest urgency level.
        """
        if not urgencies:
            return "stable"
        return max(urgencies, key=lambda u: self.URGENCY_ORDER.index(u)
                   if u in self.URGENCY_ORDER else 0)
