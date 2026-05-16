# -*- coding: utf-8 -*-
"""plotting/ed_calibration_plotter.py — ED Calibration Visualization.

All methods save to file and return Path. Never calls plt.show().
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from config import CONFIG
from modules.image_qc.ed_calibration import EDCalibrationResult

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
__all__ = ["EDCalibrationPlotter"]


class EDCalibrationPlotter:
    """Generates ED calibration curve plots."""

    def plot_calibration_curve(self, result: EDCalibrationResult,
                                output_path: Path = None) -> Path:
        """Calibration curve with material scatter and segment annotations."""
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "ed_calibration_%s.png" % result.acquisition_date

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Dense calibration curve
        ax.plot(result.hu_curve, result.red_curve, 'b-', linewidth=2, label='Calibration curve')

        # Segment join boundary
        ax.axvline(result.segment_join_hu, color='gray', linestyle='--', alpha=0.5,
                   label="Segment join (%.0f HU)" % result.segment_join_hu)

        # Material scatter points
        for m in result.measurements:
            color = 'green' if m.passed else 'red'
            ax.scatter(m.measured_mean_hu, m.computed_red, c=color, s=80,
                       edgecolors='black', linewidth=0.5, zorder=5)
            ax.errorbar(m.measured_mean_hu, m.computed_red,
                        xerr=m.measured_std_hu, fmt='none', ecolor=color, alpha=0.5)
            ax.annotate(m.material_name, (m.measured_mean_hu, m.computed_red),
                       fontsize=6, rotation=45, xytext=(5, 5), textcoords='offset points')

        # Reference markers
        for m in result.measurements:
            ax.scatter(m.nominal_hu, m.reference_red, marker='x', c='gray', s=50, alpha=0.6)

        # Segment annotations
        seg_text = ("Soft tissue: slope=%.4f, R^2=%.4f\nBone: slope=%.4f, R^2=%.4f"
                    % (result.soft_tissue_slope, result.soft_tissue_r_squared,
                       result.bone_slope, result.bone_r_squared))
        ax.text(0.05, 0.95, seg_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Hardware status
        status = "PASS" if result.all_passed else "FAIL"
        ax.text(0.95, 0.05, "kVp Generator Status: %s" % status,
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel("CT Number (HU)")
        ax.set_ylabel("Relative Electron Density (RED)")
        ax.set_xlim(-1100, 1600)
        ax.set_ylim(-0.1, 2.0)
        ax.set_title("HU -> Electron Density Calibration Curve\n%s (%s)"
                      % (result.series_description, result.acquisition_date))
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info("ED calibration plot saved to: %s", output_path)
        return output_path

    def plot_calibration_comparison(self, results: list[EDCalibrationResult],
                                     labels: list[str],
                                     output_path: Path = None) -> Path:
        """Overlay multiple calibration curves from different sessions."""
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "ed_comparison.png"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = matplotlib.cm.viridis
        n = len(results)

        for i, (result, label) in enumerate(zip(results, labels)):
            color = cmap(float(i) / max(n - 1, 1))
            ax.plot(result.hu_curve, result.red_curve, color=color, label=label, alpha=0.8)

        # Scatter points only for first session
        if results:
            for m in results[0].measurements:
                ax.scatter(m.measured_mean_hu, m.computed_red, marker='o', c='black', s=30, zorder=5)

        ax.set_xlabel("CT Number (HU)")
        ax.set_ylabel("Relative Electron Density (RED)")
        ax.set_title("Calibration Curve Drift -- %d Sessions" % n)
        ax.legend(fontsize=8)
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return output_path

    def plot_calibration_trend(self, dates: list[str], soft_slopes: list[float],
                                bone_slopes: list[float], max_deviations: list[float],
                                output_path: Path = None) -> Path:
        """Three-panel trend: soft-tissue slope, bone slope, max RED deviation."""
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "ed_trend.png"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        parsed_dates = [datetime.strptime(d, "%Y%m%d") for d in dates]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        # Panel 1: soft-tissue slope
        ax1.plot(parsed_dates, soft_slopes, 'bo-', markersize=5)
        if soft_slopes:
            ax1.axhline(soft_slopes[0], color='blue', linestyle='--', alpha=0.3)
        ax1.set_ylabel("Slope")
        ax1.set_title("Soft-tissue slope -- Hardware: kVp Generator (low range)")
        ax1.grid(True)

        # Panel 2: bone slope
        ax2.plot(parsed_dates, bone_slopes, 'go-', markersize=5)
        if bone_slopes:
            ax2.axhline(bone_slopes[0], color='green', linestyle='--', alpha=0.3)
        ax2.set_ylabel("Slope")
        ax2.set_title("Bone slope -- Hardware: kVp Generator (high range)")
        ax2.grid(True)

        # Panel 3: max RED deviation
        ax3.plot(parsed_dates, max_deviations, 'rs-', markersize=5)
        ax3.axhline(CONFIG.ed_calibration.max_red_deviation, color='red', linestyle='--', alpha=0.6,
                     label="Max RED deviation: %.3f" % CONFIG.ed_calibration.max_red_deviation)
        ax3.set_ylabel("Max RED Deviation")
        ax3.set_xlabel("Date")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax3.legend(fontsize=9)
        ax3.grid(True)

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return output_path
