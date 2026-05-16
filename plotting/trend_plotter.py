# -*- coding: utf-8 -*-
"""
plotting/trend_plotter.py — QC Trend Visualization.

All methods save to file and return Path. Never calls plt.show().
Save at 150 DPI. Platform runs headless.

Standards References:
    - AAPM TG-66 Section 7: QC Record Keeping and Trending
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from config import CONFIG
from modules.predictive.trend_model import TrendModelResult
from modules.predictive.failure_predictor import (
    FailurePrediction,
    HardwareStatusReport,
    MaintenanceAlert,
)

matplotlib.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.labelsize':    12,
    'axes.titlesize':    13,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'grid.alpha':        0.3,
    'grid.color':        '#888888',
    'grid.linestyle':    '--',
    'lines.linewidth':   2.0,
})

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = ["TrendPlotter"]


class TrendPlotter:
    """Generates metric trend plots with regression lines and breach forecasts."""

    def plot_metric_trend(
        self,
        trend: TrendModelResult,
        archive: "MetricsArchive",
        metric_name: str,
        prediction: FailurePrediction,
        output_path: Path = None,
    ) -> Path:
        """Plot a single metric trend with regression line and breach forecast.

        Parameters
        ----------
        trend : TrendModelResult
        archive : MetricsArchive
        metric_name : str
        prediction : FailurePrediction
        output_path : Path, optional

        Returns
        -------
        Path
        """
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / ("trend_%s.png" % metric_name)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 8))

        # ── Data retrieval ──
        dates, values = archive.get_metric_series(metric_name)
        date_objects = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        today = datetime.now()

        # ── 1. Safe/danger background zones ──
        threshold = prediction.tolerance_threshold
        breach_dir = prediction.breach_direction
        y_min = min(values) - abs(min(values)) * 0.3 if values else 0
        y_max = max(values) + abs(max(values)) * 0.3 if values else 10

        if breach_dir == "above":
            ax.axhspan(y_min, threshold, alpha=0.05, color='green', zorder=0)
            ax.axhspan(threshold, y_max * 1.5, alpha=0.05, color='red', zorder=0)
        else:
            ax.axhspan(threshold, y_max * 1.5, alpha=0.05, color='green', zorder=0)
            ax.axhspan(y_min, threshold, alpha=0.05, color='red', zorder=0)

        # ── 2. Historical data scatter ──
        ax.scatter(date_objects, values, c='blue', s=60, zorder=5, label="Observed")

        # ── 3. Regression line extended to forecast horizon ──
        first_date = datetime.strptime(trend.first_date, "%Y-%m-%d")
        forecast_end = today + timedelta(days=CONFIG.predictive.forecast_horizon_days)
        n_line_points = 200
        line_dates = [
            first_date + timedelta(days=i * (forecast_end - first_date).days / n_line_points)
            for i in range(n_line_points + 1)
        ]
        line_x_ordinals = [(d - first_date).days for d in line_dates]
        line_y = [trend.slope_per_day * x + trend.intercept for x in line_x_ordinals]
        ax.plot(line_dates, line_y, 'b-', linewidth=1.5, label="Regression line", zorder=3)

        # ── 4. 95% prediction interval band ──
        pi_lower = []
        pi_upper = []
        for d in line_dates:
            d_str = d.strftime("%Y-%m-%d")
            try:
                lo, hi = trend.prediction_interval_at_date(d_str, confidence=0.95)
                pi_lower.append(lo)
                pi_upper.append(hi)
            except Exception:
                pi_lower.append(np.nan)
                pi_upper.append(np.nan)
        ax.fill_between(line_dates, pi_lower, pi_upper, alpha=0.15, color='blue', zorder=2,
                        label="95% PI")

        # ── 5. Threshold line ──
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.6,
                   label="Threshold: %.3f" % threshold, zorder=4)

        # ── 6. Today vertical line ──
        ax.axvline(today, color='gray', linestyle='--', alpha=0.5,
                   label="Today", zorder=4)

        # ── 7. Predicted breach line ──
        if prediction.predicted_breach_date is not None:
            breach_dt = datetime.strptime(prediction.predicted_breach_date, "%Y-%m-%d")
            ax.axvline(breach_dt, color='orange', linestyle='--', alpha=0.7,
                       label="Breach: %s (%dd)"
                       % (prediction.predicted_breach_date,
                          prediction.days_until_breach or 0),
                       zorder=4)

        # ── 8. CI shading on breach date ──
        if prediction.confidence_interval_days is not None:
            ci_lo_days, ci_hi_days = prediction.confidence_interval_days
            ci_lo_dt = today + timedelta(days=ci_lo_days)
            ci_hi_dt = today + timedelta(days=ci_hi_days)
            ax.axvspan(ci_lo_dt, ci_hi_dt, alpha=0.2, color='orange', zorder=1)

        # ── Annotations ──
        ann_text = (
            "Slope: %+.4f/day\nR² = %.3f\nn = %d sessions\nTrend: %s"
            % (trend.slope_per_day, trend.r_squared,
               trend.n_data_points, trend.trend_direction.upper())
        )
        ax.text(
            0.02, 0.98, ann_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        )

        # Hardware annotation at bottom
        hw_text = (
            "Hardware: %s\nAction: %s\nUrgency: %s"
            % (prediction.hardware_component,
               prediction.clinical_action[:80] + "...",
               prediction.get_urgency_level().upper())
        )
        fig.text(
            0.5, 0.01, hw_text, ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
        )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate(rotation=30)

        ax.set_xlabel("Date")
        ax.set_ylabel(metric_name)
        ax.set_title(
            "QC Trend — %s — %s (%+.4f/day)"
            % (metric_name, trend.trend_direction.upper(), trend.slope_per_day)
        )
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info("Trend plot saved to: %s", output_path)
        return output_path

    def plot_hardware_dashboard(
        self,
        hardware_report: HardwareStatusReport,
        archive: "MetricsArchive",
        trend_results: dict[str, TrendModelResult],
        predictions: list[FailurePrediction],
        output_path: Path = None,
    ) -> Path:
        """3-panel hardware dashboard.

        Row 1: X-ray tube filament — center_water_std_hu
        Row 2: Anode focal spot — mtf_50_lpmm
        Row 3: kVp generator — ed_soft_tissue_slope

        Parameters
        ----------
        hardware_report : HardwareStatusReport
        archive : MetricsArchive
        trend_results : dict[str, TrendModelResult]
        predictions : list[FailurePrediction]
        output_path : Path, optional

        Returns
        -------
        Path
        """
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "hardware_dashboard.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        panel_metrics = [
            ("center_water_std_hu", "X-ray Tube Filament", hardware_report.tube_filament_urgency),
            ("mtf_50_lpmm", "Anode Focal Spot", hardware_report.focal_spot_urgency),
            ("ed_soft_tissue_slope", "kVp Generator", hardware_report.kvp_generator_urgency),
        ]

        urgency_colors = {
            "breached": "red", "critical": "red",
            "warning": "orange", "monitor": "gold",
            "stable": "green", "improving": "green",
        }

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

        for idx, (metric, hw_name, urgency) in enumerate(panel_metrics):
            ax = axes[idx]

            # Colored left spine
            spine_color = urgency_colors.get(urgency, "gray")
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_linewidth(4)
            ax.spines['left'].set_color(spine_color)

            dates, values = archive.get_metric_series(metric)
            if not dates or metric not in trend_results:
                ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                        ha='center', fontsize=12)
                ax.set_title(
                    "%s — %s [%s]" % (hw_name, metric, urgency.upper()),
                    fontsize=10,
                )
                continue

            date_objects = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
            trend = trend_results[metric]

            # Find matching prediction
            pred = next((p for p in predictions if p.metric_name == metric), None)

            ax.scatter(date_objects, values, c='blue', s=30, zorder=5)

            # Regression line
            first_date = datetime.strptime(trend.first_date, "%Y-%m-%d")
            today = datetime.now()
            forecast_end = today + timedelta(days=90)
            n_pts = 100
            ld = [
                first_date + timedelta(days=i * (forecast_end - first_date).days / n_pts)
                for i in range(n_pts + 1)
            ]
            ly = [trend.slope_per_day * (d - first_date).days + trend.intercept for d in ld]
            ax.plot(ld, ly, 'b-', linewidth=1.2)

            # Threshold
            if pred is not None:
                ax.axhline(pred.tolerance_threshold, color='red', linestyle='--', alpha=0.5)

            ax.set_title(
                "%s — %s [%s]"
                % (hw_name, metric, urgency.upper()),
                fontsize=10,
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.grid(True)
            ax.set_ylabel(metric, fontsize=9)

        fig.suptitle(
            "Hardware Status Dashboard — %s — %s"
            % (hardware_report.scanner_id, hardware_report.report_date),
            fontsize=14, fontweight='bold',
        )

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info("Hardware dashboard saved to: %s", output_path)
        return output_path

    def plot_maintenance_timeline(
        self,
        alert: MaintenanceAlert,
        output_path: Path = None,
    ) -> Path:
        """Horizontal Gantt-style maintenance timeline.

        Parameters
        ----------
        alert : MaintenanceAlert
        output_path : Path, optional

        Returns
        -------
        Path
        """
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "maintenance_timeline.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, max(6, len(alert.predictions) * 1.2)))

        today = datetime.now()
        horizon_end = today + timedelta(days=CONFIG.predictive.forecast_horizon_days)

        urgency_colors = {
            "breached": "#D32F2F", "critical": "#F44336",
            "warning": "#FF9800", "monitor": "#FFC107",
            "stable": "#4CAF50", "improving": "#66BB6A",
        }

        # Collect metrics with breach dates
        bar_metrics = []
        for pred in alert.predictions:
            if pred.predicted_breach_date is not None:
                bar_metrics.append(pred)

        if not bar_metrics:
            ax.text(0.5, 0.5, "No breach dates predicted\nAll metrics stable",
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
        else:
            for i, pred in enumerate(bar_metrics):
                breach_dt = datetime.strptime(pred.predicted_breach_date, "%Y-%m-%d")
                urgency = pred.get_urgency_level()
                color = urgency_colors.get(urgency, "#9E9E9E")

                # Main bar from today to breach
                bar_start = mdates.date2num(today)
                bar_end = mdates.date2num(breach_dt)
                bar_width = bar_end - bar_start

                ax.barh(i, bar_width, left=bar_start, height=0.6,
                        color=color, alpha=0.7, zorder=3)

                # CI extension
                if pred.confidence_interval_days is not None:
                    ci_lo, ci_hi = pred.confidence_interval_days
                    ci_lo_dt = today + timedelta(days=ci_lo)
                    ci_hi_dt = today + timedelta(days=ci_hi)
                    ci_start = mdates.date2num(ci_lo_dt)
                    ci_width = mdates.date2num(ci_hi_dt) - ci_start
                    ax.barh(i, ci_width, left=ci_start, height=0.6,
                            color=color, alpha=0.2, zorder=2)

                # Label
                label = "%s: ~%s" % (pred.metric_name, pred.predicted_breach_date)
                ax.text(bar_start - 1, i, label, ha='right', va='center', fontsize=8)

            ax.set_yticks(range(len(bar_metrics)))
            ax.set_yticklabels([""] * len(bar_metrics))

        # Recommended maintenance date
        if alert.recommended_maintenance_date is not None:
            maint_dt = datetime.strptime(alert.recommended_maintenance_date, "%Y-%m-%d")
            ax.axvline(mdates.date2num(maint_dt), color='red', linestyle='--',
                       linewidth=2, label="Recommended maintenance", zorder=5)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        fig.autofmt_xdate(rotation=30)

        # Legend for urgency colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, alpha=0.7, label=k.capitalize())
            for k, c in urgency_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        ax.set_xlabel("Date")
        ax.set_title(
            "Predicted Maintenance Timeline — %s" % alert.scanner_id,
            fontsize=14, fontweight='bold',
        )
        ax.grid(True, axis='x')

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info("Maintenance timeline saved to: %s", output_path)
        return output_path
