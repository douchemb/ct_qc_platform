# -*- coding: utf-8 -*-
"""plotting/mtf_plotter.py — MTF Visualization.

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
from modules.image_qc.mtf_calculator import MTFResult

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
__all__ = ["MTFPlotter"]


class MTFPlotter:
    """Generates MTF curve plots. All methods save to file and return Path."""

    def plot_mtf_curve(self, result: MTFResult, output_path: Path = None) -> Path:
        """Two-panel: MTF curve (top) and LSF (bottom)."""
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "mtf_%s.png" % result.acquisition_date

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                         gridspec_kw={'height_ratios': [3, 1]})

        # Top: MTF curve
        ax1.plot(result.freq_axis_lpmm, result.mtf_values, 'b-', linewidth=2)
        ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
        ax1.axhline(0.1, color='gray', linestyle='--', alpha=0.4)

        if not np.isnan(result.mtf_50_lpmm):
            ax1.axvline(result.mtf_50_lpmm, color='blue', linestyle='--', alpha=0.6,
                        label="MTF50 = %.3f lp/mm" % result.mtf_50_lpmm)
        if not np.isnan(result.mtf_10_lpmm):
            ax1.axvline(result.mtf_10_lpmm, color='gray', linestyle='--', alpha=0.6,
                        label="MTF10 = %.3f lp/mm" % result.mtf_10_lpmm)

        ax1.text(0.95, 0.95, "Hardware indicator:\nFocal spot blooming\n(anode wear)",
                 transform=ax1.transAxes, fontsize=8, verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        ax1.set_ylabel("MTF (normalized)")
        ax1.set_ylim(-0.05, 1.1)
        ax1.set_title("MTF -- %s method -- %s" % (result.method, result.series_description))
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True)

        # Bottom: LSF
        ax2.plot(result.lsf_axis_mm, result.lsf, 'r-', linewidth=1.5)
        ax2.set_xlabel("Position (mm)")
        ax2.set_ylabel("LSF")
        ax2.set_title("Line Spread Function")
        ax2.grid(True)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info("MTF plot saved to: %s", output_path)
        return output_path

    def plot_mtf_comparison(self, results: list[MTFResult], labels: list[str],
                             output_path: Path = None) -> Path:
        """Overlaid MTF curves with summary table."""
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "mtf_comparison.png"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = matplotlib.cm.tab10

        table_data = []
        for i, (result, label) in enumerate(zip(results, labels)):
            color = cmap(i % 10)
            ax.plot(result.freq_axis_lpmm, result.mtf_values, color=color, label=label)
            passed = result.passes_resolution_check()
            table_data.append([label, "%.3f" % result.mtf_50_lpmm,
                               "%.3f" % result.mtf_10_lpmm,
                               "%.3f" % result.mtf_at_nyquist])

        ax.set_xlabel("Spatial Frequency (lp/mm)")
        ax.set_ylabel("MTF (normalized)")
        ax.set_title("MTF Comparison")
        ax.legend(fontsize=9)
        ax.grid(True)

        # Add table
        if table_data:
            col_labels = ["Label", "MTF50 (lp/mm)", "MTF10 (lp/mm)", "MTF@Nyquist"]
            table = ax.table(cellText=table_data, colLabels=col_labels,
                             loc='bottom', bbox=[0.0, -0.35, 1.0, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            fig.subplots_adjust(bottom=0.3)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path

    def plot_mtf_trend(self, dates: list[str], mtf50_values: list[float],
                        mtf10_values: list[float], output_path: Path = None) -> Path:
        """MTF50 and MTF10 over time with minimum threshold."""
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "mtf_trend.png"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        parsed_dates = [datetime.strptime(d, "%Y%m%d") for d in dates]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(parsed_dates, mtf50_values, 'bo-', label="MTF50", markersize=5)
        ax.plot(parsed_dates, mtf10_values, 'gs-', label="MTF10", markersize=5)
        ax.axhline(0.4, color='red', linestyle='--', alpha=0.6, label="Min MTF50 (0.4 lp/mm)")

        ax.set_xlabel("Date")
        ax.set_ylabel("Frequency (lp/mm)")
        ax.set_title("MTF Trend -- Hardware: Anode Focal Spot Blooming")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(fontsize=9)
        ax.grid(True)

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return output_path
