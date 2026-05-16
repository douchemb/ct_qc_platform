# -*- coding: utf-8 -*-
"""plotting/nps_plotter.py — NPS Visualization.

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
from modules.image_qc.nps_calculator import NPSResult

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
__all__ = ["NPSPlotter"]


class NPSPlotter:
    """Generates NPS plots. All methods save to file and return Path."""

    def plot_nps_1d(self, result: NPSResult, output_path: Path = None) -> Path:
        """1D radially-averaged NPS plot with peak frequency and TG-66 limit."""
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "nps_1d_%s.png" % result.acquisition_date

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(result.freq_axis_lpmm, result.nps_1d, 'b-', linewidth=2)
        ax.axvline(result.nps_peak_frequency_lpmm, color='blue', linestyle='--', alpha=0.6,
                   label="f_peak = %.3f lp/mm" % result.nps_peak_frequency_lpmm)

        # TG-66 variance limit: noise_tolerance^2
        tg66_limit = CONFIG.image_qc.noise_tolerance_hu ** 2
        ax.axhline(tg66_limit, color='red', linestyle='--', alpha=0.6,
                   label="TG-66 variance limit (%.0f HU^2)" % tg66_limit)

        # Annotation box
        textstr = ("Noise variance: %.2f HU^2\nNoise SD (NPS): %.2f HU\nSlices used: %d"
                   % (result.nps_integral, result.noise_std_from_nps, result.n_slices_used))
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel("Spatial Frequency (cycles/mm)")
        ax.set_ylabel("NPS (HU^2 * mm^2)")
        ax.set_title("Noise Power Spectrum -- %s (%s)" % (result.series_description, result.acquisition_date))
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info("NPS 1D plot saved to: %s", output_path)
        return output_path

    def plot_nps_comparison(self, results: list[NPSResult], labels: list[str],
                            output_path: Path = None) -> Path:
        """Overlay multiple NPS curves."""
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "nps_comparison.png"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = matplotlib.cm.tab10

        for i, (result, label) in enumerate(zip(results, labels)):
            color = cmap(i % 10)
            ax.plot(result.freq_axis_lpmm, result.nps_1d, color=color, label=label)
            ax.axvline(result.nps_peak_frequency_lpmm, color=color, linestyle='--', alpha=0.4)

        ax.set_xlabel("Spatial Frequency (cycles/mm)")
        ax.set_ylabel("NPS (HU^2 * mm^2)")
        ax.set_title("NPS Comparison -- %d Series" % len(results))
        ax.legend(fontsize=9)
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return output_path

    def plot_nps_2d(self, result: NPSResult, output_path: Path = None) -> Path:
        """2D NPS heatmap with viridis colormap."""
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "nps_2d_%s.png" % result.acquisition_date

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(result.nps_2d, cmap='viridis', origin='lower')
        cbar = fig.colorbar(im, ax=ax, label="NPS (HU^2 * mm^2)")

        # Crosshair at DC center
        cx, cy = result.nps_2d.shape[1] // 2, result.nps_2d.shape[0] // 2
        ax.axhline(cy, color='white', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.axvline(cx, color='white', linestyle='--', linewidth=0.5, alpha=0.6)

        ax.set_title("2D NPS -- %s" % result.series_description)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return output_path

    def plot_nps_trend(self, dates: list[str], peak_freqs: list[float],
                       noise_stds: list[float], output_path: Path = None) -> Path:
        """Two-panel trend plot: NPS peak frequency and noise SD over time."""
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "nps_trend.png"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        parsed_dates = [datetime.strptime(d, "%Y%m%d") for d in dates]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Top: peak frequency
        ax1.plot(parsed_dates, peak_freqs, 'bo-', markersize=5)
        if peak_freqs:
            ref = peak_freqs[0]
            ax1.axhline(ref + 0.05, color='orange', linestyle='--', alpha=0.5)
            ax1.axhline(ref - 0.05, color='orange', linestyle='--', alpha=0.5)
        ax1.set_ylabel("Peak Frequency (lp/mm)")
        ax1.set_title("NPS Peak Frequency -- Hardware: X-ray Tube Filament Wear")
        ax1.grid(True)

        # Bottom: noise SD
        ax2.plot(parsed_dates, noise_stds, 'rs-', markersize=5)
        ax2.axhline(CONFIG.image_qc.noise_tolerance_hu, color='red', linestyle='--', alpha=0.6,
                     label="TG-66 Tolerance: %.1f HU" % CONFIG.image_qc.noise_tolerance_hu)
        if hasattr(CONFIG.image_qc, 'noise_warning_hu'):
            ax2.axhline(CONFIG.image_qc.noise_warning_hu, color='orange', linestyle='--', alpha=0.6,
                         label="Warning: %.1f HU" % CONFIG.image_qc.noise_warning_hu)
        ax2.set_ylabel("Noise SD (HU)")
        ax2.set_xlabel("Date")
        ax2.set_title("Noise SD -- TG-66 Tolerance: %.1f HU" % CONFIG.image_qc.noise_tolerance_hu)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.legend(fontsize=9)
        ax2.grid(True)

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return output_path
