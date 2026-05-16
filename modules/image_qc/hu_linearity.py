# -*- coding: utf-8 -*-
"""
modules/image_qc/hu_linearity.py — HU Linearity Validator.

Measures HU accuracy across multiple phantom materials per AAPM TG-66 Section 5.2.

Hardware failure mapping:
    max_deviation_hu — primary indicator of kVp generator instability.
    slope deviation from 1.0 — systematic kVp shift.
Reference: AAPM TG-66 Section 5.2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from config import CONFIG, ImageQCConfig
from modules.image_qc.roi_stats import SliceAnalysisResult, ROIDescriptor

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = ["HULinearityAnalyzer", "HULinearityResult"]


@dataclass
class HULinearityResult:
    """
    HU linearity analysis result.

    Hardware failure mapping:
      max_deviation_hu — primary indicator of kVp generator instability.
        Growing deviation across the HU range (especially at extremes)
        indicates inconsistent kVp output from the high-voltage generator.
      slope deviation from 1.0 — systematic kVp shift.
        slope < 1.0 indicates kVp lower than nominal; slope > 1.0 indicates higher.
    Reference: AAPM TG-66 Section 5.2.
    """
    acquisition_date: str
    series_description: str
    measurements: list[tuple[str, float, float, float, bool]]
    max_deviation_hu: float
    all_passed: bool
    r_squared: float
    slope: float
    intercept: float

    def to_dict(self) -> dict:
        return asdict(self)

    def passes_tg66(self) -> bool:
        """Returns True if all materials are within TG-66 linearity tolerance."""
        return self.all_passed


class HULinearityAnalyzer:
    """Verifies HU accuracy across CatPhan phantom inserts.

    Reference: AAPM TG-66 Section 5.2
    """

    def __init__(self, config: ImageQCConfig) -> None:
        self._config = config
        self._nominal_values = {
            "water": config.hu_water_nominal,
            "air": config.hu_air_nominal,
            "acrylic": config.hu_acrylic_nominal,
        }

    def analyze(
        self,
        slice_results: list[SliceAnalysisResult],
        material_rois: dict[str, ROIDescriptor],
    ) -> HULinearityResult:
        """Analyze HU linearity from slice analysis results.

        For each material, averages mean_hu across all slices.
        Compares to nominal values. Tolerance: ±hu_linearity_tolerance.
        """
        tolerance = self._config.hu_linearity_tolerance
        measurements = []
        measured_values = []
        nominal_values = []

        for material_name, roi in material_rois.items():
            # Get nominal HU from config or from the dict
            nominal_hu = self._nominal_values.get(material_name, None)
            if nominal_hu is None:
                logger.warning("No nominal HU for material '%s', skipping", material_name)
                continue

            # Average measured HU across all slices
            hu_values = []
            for result in slice_results:
                try:
                    stat = result.get_roi_stat(roi.label)
                    hu_values.append(stat.mean_hu)
                except KeyError:
                    continue

            if not hu_values:
                logger.warning("No ROI data for material '%s'", material_name)
                continue

            measured_mean = float(np.mean(hu_values))
            deviation = abs(measured_mean - nominal_hu)
            passed = deviation <= tolerance

            measurements.append((material_name, nominal_hu, measured_mean, deviation, passed))
            measured_values.append(measured_mean)
            nominal_values.append(nominal_hu)

        # Fit linear regression
        if len(nominal_values) >= 2:
            coeffs = np.polyfit(nominal_values, measured_values, 1)
            slope = float(coeffs[0])
            intercept = float(coeffs[1])
            # R-squared
            predicted = np.polyval(coeffs, nominal_values)
            ss_res = np.sum((np.array(measured_values) - predicted) ** 2)
            ss_tot = np.sum((np.array(measured_values) - np.mean(measured_values)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        else:
            slope, intercept, r_squared = 1.0, 0.0, 1.0

        all_passed = all(m[4] for m in measurements) if measurements else False
        max_dev = max(m[3] for m in measurements) if measurements else 0.0

        acq_date = slice_results[0].metadata.acquisition_date if slice_results else ""
        series = slice_results[0].metadata.series_description if slice_results else ""

        return HULinearityResult(
            acquisition_date=acq_date, series_description=series,
            measurements=measurements, max_deviation_hu=max_dev,
            all_passed=all_passed, r_squared=float(r_squared),
            slope=slope, intercept=intercept,
        )

    def plot_linearity(self, result: HULinearityResult, output_path: Path = None) -> Path:
        """Scatter plot of measured vs nominal HU with linear fit."""
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "hu_linearity_%s.png" % result.acquisition_date

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        nominals = [m[1] for m in result.measurements]
        measured = [m[2] for m in result.measurements]
        passed = [m[4] for m in result.measurements]
        names = [m[0] for m in result.measurements]

        hu_range = [min(nominals) - 100, max(nominals) + 100]
        ax.plot(hu_range, hu_range, 'k--', alpha=0.4, label='Identity (y=x)')

        for i, (n, m, p, name) in enumerate(zip(nominals, measured, passed, names)):
            color = 'green' if p else 'red'
            ax.scatter(n, m, c=color, s=80, zorder=5, edgecolors='black', linewidth=0.5)
            ax.annotate(name, (n, m), fontsize=7, rotation=30,
                       xytext=(5, 5), textcoords='offset points')

        # Fit line
        x_fit = np.linspace(hu_range[0], hu_range[1], 100)
        y_fit = result.slope * x_fit + result.intercept
        ax.plot(x_fit, y_fit, 'b-', linewidth=1.5, alpha=0.7,
                label='Fit: y=%.4fx%+.2f' % (result.slope, result.intercept))

        ax.set_xlabel("Nominal HU")
        ax.set_ylabel("Measured HU")
        ax.set_title("HU Linearity — %s (%s)" % (result.series_description, result.acquisition_date))
        ax.legend(loc='upper left', fontsize=9)

        status = "PASS" if result.all_passed else "FAIL"
        ax.text(0.95, 0.05, "R²=%.4f\nkVp Generator: %s" % (result.r_squared, status),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)

        logger.info("HU linearity plot saved to: %s", output_path)
        return output_path
