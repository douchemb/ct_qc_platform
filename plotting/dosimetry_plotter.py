# -*- coding: utf-8 -*-
"""
plotting/dosimetry_plotter.py — Dosimetry Visualization.

All methods save to file and return Path. Never calls plt.show().
Save at 150 DPI. Platform runs headless.

Standards References:
    - AAPM Report 204 (2011): SSDE f-factor curve
    - AAPM TG-220 (2014): D_w profile
    - EUR 16262: Diagnostic Reference Levels (DRL)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG
from modules.dosimetry.dw_calculator import DwSeriesResult, DwLocalizerResult
from modules.dosimetry.ssde_calculator import SSDESeriesResult
from modules.dosimetry.dosimetry_report import DosimetryReport

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
__all__ = ["DosimetryPlotter"]


class DosimetryPlotter:
    """Generates dosimetry plots for D_w profiles, SSDE, and f-factor curves."""

    def plot_dw_profile(
        self,
        dw_result: DwSeriesResult,
        localizer_result: Optional[DwLocalizerResult] = None,
        output_path: Path = None,
    ) -> Path:
        """Plot D_w profile along the axial axis.

        Parameters
        ----------
        dw_result : DwSeriesResult
            D_w results from axial series.
        localizer_result : DwLocalizerResult, optional
            D_w estimates from localizer, overlaid if provided.
        output_path : Path, optional
            Output path. Defaults to plots_dir/dw_profile.png.

        Returns
        -------
        Path
            Path to saved plot.
        """
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "dw_profile_%s.png" % dw_result.acquisition_date
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 7))

        # Extract per-slice data
        positions = [r.slice_position_mm for r in dw_result.slice_results]
        dw_values = [r.dw_cm for r in dw_result.slice_results]

        # Primary curve: blue solid line
        ax.plot(positions, dw_values, 'b-o', markersize=4, label="Axial D_w", zorder=5)

        # TG-220 table range boundaries
        ax.axhline(
            CONFIG.dosimetry.dw_min_cm, color='gray', linestyle='--', alpha=0.6,
            label="TG-220 table range (%.0f–%.0f cm)" % (
                CONFIG.dosimetry.dw_min_cm, CONFIG.dosimetry.dw_max_cm
            ),
        )
        ax.axhline(
            CONFIG.dosimetry.dw_max_cm, color='gray', linestyle='--', alpha=0.6,
        )

        # Isocenter line
        ax.axvline(0.0, color='gray', linestyle='--', alpha=0.4, label="Isocenter")

        # Mark D_w at isocenter
        iso_positions = np.array(positions)
        iso_idx = int(np.argmin(np.abs(iso_positions)))
        ax.scatter(
            [positions[iso_idx]], [dw_values[iso_idx]],
            marker='*', s=200, c='red', zorder=10,
            label="D_w @ isocenter = %.2f cm" % dw_result.dw_at_isocenter_cm,
        )

        # Localizer overlay if available
        if localizer_result is not None:
            ax.plot(
                localizer_result.row_positions_mm,
                localizer_result.dw_per_row_cm,
                '--', color='orange', alpha=0.7, linewidth=1.5,
                label="Localizer estimate",
            )

        # Annotation box
        ann_text = (
            "D_w mean: %.2f cm\n"
            "D_w std: %.2f cm\n"
            "Slices: %d"
            % (dw_result.dw_mean_cm, dw_result.dw_std_cm, dw_result.n_slices)
        )
        ax.text(
            0.97, 0.97, ann_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        )

        ax.set_xlabel("Axial Position (mm)")
        ax.set_ylabel("Water-Equivalent Diameter D_w (cm)")
        ax.set_title("Water-Equivalent Diameter Profile — %s" % dw_result.series_description)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info("D_w profile plot saved to: %s", output_path)
        return output_path

    def plot_ssde_profile(
        self,
        ssde_result: SSDESeriesResult,
        output_path: Path = None,
    ) -> Path:
        """Plot SSDE profile with two panels: dose and f-factor.

        Top panel: SSDE per slice, DRL line, CTDIvol line, shaded region.
        Bottom panel: f-factor per slice, f=1.0 reference line.

        Parameters
        ----------
        ssde_result : SSDESeriesResult
        output_path : Path, optional

        Returns
        -------
        Path
        """
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "ssde_profile_%s.png" % ssde_result.acquisition_date
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 9), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]},
        )

        positions = [s.slice_position_mm for s in ssde_result.slice_results]
        ssde_values = [s.ssde_mgy for s in ssde_result.slice_results]
        f_values = [s.f_factor for s in ssde_result.slice_results]

        # ── Top panel: Dose ──
        ax1.plot(positions, ssde_values, 'b-o', markersize=4, label="SSDE", zorder=5)

        # DRL line at 25 mGy
        ax1.axhline(25.0, color='red', linestyle='--', alpha=0.6, label="DRL (25 mGy)")

        # CTDIvol line
        ax1.axhline(
            ssde_result.ctdi_vol_mgy, color='gray', linestyle='--', alpha=0.6,
            label="CTDIvol = %.2f mGy" % ssde_result.ctdi_vol_mgy,
        )

        # Shaded region between CTDIvol and SSDE
        ctdi_line = np.full(len(positions), ssde_result.ctdi_vol_mgy)
        ssde_arr = np.array(ssde_values)
        positions_arr = np.array(positions)

        # Green where SSDE < CTDIvol (larger patient), red where SSDE > CTDIvol (smaller patient)
        ax1.fill_between(
            positions_arr, ctdi_line, ssde_arr,
            where=ssde_arr <= ctdi_line,
            interpolate=True, alpha=0.1, color='green',
        )
        ax1.fill_between(
            positions_arr, ctdi_line, ssde_arr,
            where=ssde_arr > ctdi_line,
            interpolate=True, alpha=0.1, color='red',
        )

        ax1.set_ylabel("Dose (mGy)")
        ax1.set_title(
            "SSDE Profile — %s — CTDIvol = %.2f mGy"
            % (ssde_result.series_description, ssde_result.ctdi_vol_mgy)
        )
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True)

        # ── Bottom panel: f-factor ──
        ax2.plot(positions, f_values, '-o', color='teal', markersize=4, label="f-factor")

        # f=1.0 reference line
        ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.6, label="f = 1.0 (reference)")

        # Shaded regions
        f_arr = np.array(f_values)
        ones = np.ones(len(positions))

        # Light red above f=1.0: patient receives more dose than CTDIvol
        ax2.fill_between(
            positions_arr, ones, f_arr,
            where=f_arr > ones,
            interpolate=True, alpha=0.08, color='red',
        )
        # Light blue below f=1.0: patient receives less dose
        ax2.fill_between(
            positions_arr, ones, f_arr,
            where=f_arr < ones,
            interpolate=True, alpha=0.08, color='blue',
        )

        ax2.set_xlabel("Axial Position (mm)")
        ax2.set_ylabel("f-factor (dimensionless)")
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info("SSDE profile plot saved to: %s", output_path)
        return output_path

    def plot_f_factor_curve(
        self,
        ssde_result: SSDESeriesResult,
        output_path: Path = None,
    ) -> Path:
        """Plot the complete f(D_w) conversion factor curve.

        Shows the AAPM Report 204 Table A curve with patient marker.

        Parameters
        ----------
        ssde_result : SSDESeriesResult
        output_path : Path, optional

        Returns
        -------
        Path
        """
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "f_factor_curve_%s.png" % ssde_result.acquisition_date
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Table data from CONFIG
        dw_table = np.array([row[0] for row in CONFIG.dosimetry.ssde_conversion_table])
        f_table = np.array([row[1] for row in CONFIG.dosimetry.ssde_conversion_table])

        # Dense interpolation for smooth curve
        dw_dense = np.linspace(8, 42, 200)
        f_dense = np.interp(dw_dense, dw_table, f_table)

        # Blue curve and table points
        ax.plot(dw_dense, f_dense, 'b-', linewidth=2, label="f(D_w) curve")
        ax.plot(dw_table, f_table, 'bo', markersize=6, label="Table A data points")

        # Patient's mean D_w
        # Compute mean D_w from slice results
        patient_dw = np.mean([s.dw_cm for s in ssde_result.slice_results])
        patient_f = np.interp(patient_dw, dw_table, f_table)

        # Vertical dashed orange line at patient D_w
        ax.axvline(
            patient_dw, color='orange', linestyle='--', alpha=0.7,
            label="Patient D_w = %.2f cm" % patient_dw,
        )
        # Horizontal dashed orange line at patient f
        ax.axhline(
            patient_f, color='orange', linestyle='--', alpha=0.7,
            label="f = %.3f" % patient_f,
        )
        # Intersection star
        ax.scatter([patient_dw], [patient_f], marker='*', s=200, c='orange', zorder=10)

        # f=1.0 reference
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4)

        # Validated range shading
        ax.axvspan(10, 40, alpha=0.05, color='gray', label="Validated range (AAPM Report 204)")

        ax.set_xlabel("Water-Equivalent Diameter D_w (cm)")
        ax.set_ylabel("Conversion Factor f(D_w)")
        ax.set_xlim(8, 42)
        ax.set_title("TG-220 SSDE Conversion Factor f(D_w) — Body Protocol 120 kVp")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info("f-factor curve plot saved to: %s", output_path)
        return output_path

    def plot_dosimetry_summary(
        self,
        report: DosimetryReport,
        output_path: Path = None,
    ) -> Path:
        """2×2 grid summary combining key dosimetry visuals.

        Top-left: D_w profile (compact)
        Top-right: f(D_w) curve with patient marker
        Bottom-left: SSDE profile (compact, top panel only)
        Bottom-right: Compliance summary text panel

        Parameters
        ----------
        report : DosimetryReport
        output_path : Path, optional

        Returns
        -------
        Path
        """
        if output_path is None:
            output_path = CONFIG.paths.plots_dir / "dosimetry_summary_%s.png" % report.ssde_result.acquisition_date
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax_dw, ax_f, ax_ssde, ax_compliance = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        dw_result = report.dw_result
        ssde_result = report.ssde_result

        # ── Top-left: D_w profile (compact) ──
        positions = [r.slice_position_mm for r in dw_result.slice_results]
        dw_values = [r.dw_cm for r in dw_result.slice_results]
        ax_dw.plot(positions, dw_values, 'b-o', markersize=3)
        ax_dw.axhline(CONFIG.dosimetry.dw_min_cm, color='gray', linestyle='--', alpha=0.5)
        ax_dw.axhline(CONFIG.dosimetry.dw_max_cm, color='gray', linestyle='--', alpha=0.5)
        ax_dw.set_xlabel("Axial Position (mm)")
        ax_dw.set_ylabel("D_w (cm)")
        ax_dw.set_title("D_w Profile")
        ax_dw.grid(True)

        # ── Top-right: f(D_w) curve ──
        dw_table = np.array([row[0] for row in CONFIG.dosimetry.ssde_conversion_table])
        f_table = np.array([row[1] for row in CONFIG.dosimetry.ssde_conversion_table])
        dw_dense = np.linspace(8, 42, 200)
        f_dense = np.interp(dw_dense, dw_table, f_table)
        ax_f.plot(dw_dense, f_dense, 'b-', linewidth=1.5)
        ax_f.plot(dw_table, f_table, 'bo', markersize=4)

        patient_dw = np.mean([s.dw_cm for s in ssde_result.slice_results])
        patient_f = np.interp(patient_dw, dw_table, f_table)
        ax_f.scatter([patient_dw], [patient_f], marker='*', s=150, c='orange', zorder=10)
        ax_f.axvline(patient_dw, color='orange', linestyle='--', alpha=0.5)
        ax_f.axvspan(10, 40, alpha=0.05, color='gray')
        ax_f.set_xlabel("D_w (cm)")
        ax_f.set_ylabel("f(D_w)")
        ax_f.set_title("f(D_w) Curve")
        ax_f.set_xlim(8, 42)
        ax_f.grid(True)

        # ── Bottom-left: SSDE profile (compact) ──
        ssde_positions = [s.slice_position_mm for s in ssde_result.slice_results]
        ssde_values = [s.ssde_mgy for s in ssde_result.slice_results]
        ax_ssde.plot(ssde_positions, ssde_values, 'b-o', markersize=3)
        ax_ssde.axhline(25.0, color='red', linestyle='--', alpha=0.5, label="DRL")
        ax_ssde.axhline(ssde_result.ctdi_vol_mgy, color='gray', linestyle='--', alpha=0.5)
        ax_ssde.set_xlabel("Axial Position (mm)")
        ax_ssde.set_ylabel("Dose (mGy)")
        ax_ssde.set_title("SSDE Profile")
        ax_ssde.grid(True)

        # ── Bottom-right: Compliance summary text ──
        ax_compliance.axis('off')
        flags = report.get_compliance_flags()

        text_lines = []
        text_lines.append("COMPLIANCE SUMMARY")
        text_lines.append("─" * 30)
        for key, value in flags.items():
            status = "PASS" if value else "FAIL"
            color = "green" if value else "red"
            text_lines.append("%s: %s" % (key, status))

        text_lines.append("")
        text_lines.append("CTDIvol: %.2f mGy" % ssde_result.ctdi_vol_mgy)
        text_lines.append("SSDE mean: %.2f mGy" % ssde_result.ssde_mean_mgy)
        text_lines.append("DLP: %.1f mGy·cm" % ssde_result.dlp_mgy_cm)
        text_lines.append("Effective dose: %.2f mSv" % ssde_result.effective_dose_msv)

        summary_text = "\n".join(text_lines)

        # Color compliance flags in the text
        ax_compliance.text(
            0.1, 0.9, summary_text, transform=ax_compliance.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        )

        fig.suptitle(
            "Dosimetry Report — %s — %s" % (
                ssde_result.series_description, ssde_result.acquisition_date
            ),
            fontsize=14, fontweight='bold',
        )

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info("Dosimetry summary plot saved to: %s", output_path)
        return output_path
