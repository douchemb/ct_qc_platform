# -*- coding: utf-8 -*-
"""
modules/image_qc/mtf_calculator.py — Modulation Transfer Function Calculator.

Computes MTF from edge or bead phantoms via the ESF→LSF→MTF chain.

Hardware failure mapping:
    mtf_50_lpmm decreasing → anode focal spot blooming
    mtf_10_lpmm decreasing → severe resolution loss

References:
    - ICRU Report 54 (1996)
    - Samei et al. Med. Phys. 25(1) 1998
    - AAPM TG-66 Section 5.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from config import ImageQCConfig
from core.dicom_loader import DicomMetadata
from modules.image_qc.roi_stats import ROIDescriptor

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = ["MTFCalculator", "MTFResult"]


@dataclass
class MTFResult:
    """
    Complete MTF analysis result.

    Hardware failure mapping:
      mtf_50_lpmm — primary indicator of anode focal spot blooming.
        Monotonic decrease over months indicates progressive focal spot
        enlargement due to anode surface roughening.
      mtf_10_lpmm — severe resolution loss threshold.
        Falling below 0.5 lp/mm warrants immediate intervention.
    Reference: AAPM TG-66 Section 5.3; Samei et al. Med Phys 1998.
    """
    series_description: str
    acquisition_date: str
    pixel_spacing_mm: float
    method: str
    freq_axis_lpmm: np.ndarray
    mtf_values: np.ndarray
    mtf_50_lpmm: float
    mtf_10_lpmm: float
    mtf_at_nyquist: float
    lsf: np.ndarray
    lsf_axis_mm: np.ndarray

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    def passes_resolution_check(self, min_mtf50_lpmm: float = 0.4) -> bool:
        """Returns True if MTF50 is above the minimum threshold."""
        return self.mtf_50_lpmm >= min_mtf50_lpmm


class MTFCalculator:
    """Computes Modulation Transfer Function from CT phantom data.

    Reference: AAPM TG-66 Section 5.3
    """

    def __init__(self, config: ImageQCConfig) -> None:
        self._config = config

    def compute_from_edge(
        self, hu_array: np.ndarray, pixel_spacing_mm: float,
        metadata: DicomMetadata, edge_roi: ROIDescriptor,
    ) -> MTFResult:
        """Compute MTF from an edge phantom image."""
        roi_hu = hu_array[
            edge_roi.row_start:edge_roi.row_end,
            edge_roi.col_start:edge_roi.col_end,
        ].copy().astype(np.float64)

        oversampling = 4
        esf, esf_axis = self._extract_esf_from_edge(roi_hu, pixel_spacing_mm, oversampling)
        lsf, lsf_axis = self._differentiate_esf_to_lsf(esf, esf_axis)
        freq_axis, mtf_values = self._lsf_to_mtf(lsf, lsf_axis)

        mtf_50 = self._find_mtf_at_value(freq_axis, mtf_values, 0.5)
        mtf_10 = self._find_mtf_at_value(freq_axis, mtf_values, 0.1)
        mtf_nyq = float(mtf_values[-1]) if len(mtf_values) > 0 else 0.0

        return MTFResult(
            series_description=metadata.series_description,
            acquisition_date=metadata.acquisition_date,
            pixel_spacing_mm=pixel_spacing_mm,
            method="edge", freq_axis_lpmm=freq_axis, mtf_values=mtf_values,
            mtf_50_lpmm=mtf_50, mtf_10_lpmm=mtf_10, mtf_at_nyquist=mtf_nyq,
            lsf=lsf, lsf_axis_mm=lsf_axis,
        )

    def compute_from_bead(
        self, hu_array: np.ndarray, pixel_spacing_mm: float,
        metadata: DicomMetadata, bead_roi: ROIDescriptor,
    ) -> MTFResult:
        """Compute MTF from a bead/wire phantom image."""
        roi_hu = hu_array[
            bead_roi.row_start:bead_roi.row_end,
            bead_roi.col_start:bead_roi.col_end,
        ].copy().astype(np.float64)

        centroid = self._locate_bead_centroid(roi_hu, pixel_spacing_mm)
        oversampling = 4
        psf, radial_axis = self._extract_psf_from_bead(
            roi_hu, centroid, pixel_spacing_mm, oversampling
        )

        # PSF is effectively the LSF for radial symmetry
        lsf = psf.copy()
        lsf_axis = radial_axis.copy()

        freq_axis, mtf_values = self._lsf_to_mtf(lsf, lsf_axis)

        mtf_50 = self._find_mtf_at_value(freq_axis, mtf_values, 0.5)
        mtf_10 = self._find_mtf_at_value(freq_axis, mtf_values, 0.1)
        mtf_nyq = float(mtf_values[-1]) if len(mtf_values) > 0 else 0.0

        return MTFResult(
            series_description=metadata.series_description,
            acquisition_date=metadata.acquisition_date,
            pixel_spacing_mm=pixel_spacing_mm,
            method="bead", freq_axis_lpmm=freq_axis, mtf_values=mtf_values,
            mtf_50_lpmm=mtf_50, mtf_10_lpmm=mtf_10, mtf_at_nyquist=mtf_nyq,
            lsf=lsf, lsf_axis_mm=lsf_axis,
        )

    def _extract_esf_from_edge(
        self, roi_hu: np.ndarray, pixel_spacing_mm: float,
        oversampling_factor: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract oversampled ESF from an edge image.

        1. Detect edge orientation via gradient magnitude
        2. For each line perpendicular to edge, find sub-pixel position
        3. Accumulate into oversampled ESF bins
        """
        # Detect edge orientation
        grad_row = np.abs(np.gradient(roi_hu, axis=0))
        grad_col = np.abs(np.gradient(roi_hu, axis=1))
        vertical_edge = np.sum(grad_col) > np.sum(grad_row)

        if vertical_edge:
            lines = roi_hu.copy()  # Each row is a profile across the edge
        else:
            lines = roi_hu.T.copy()

        n_lines, n_pixels = lines.shape
        sub_pixel = pixel_spacing_mm / oversampling_factor
        n_bins = n_pixels * oversampling_factor
        half = n_bins // 2

        esf_accum = np.zeros(n_bins)
        esf_count = np.zeros(n_bins)

        for line_idx in range(n_lines):
            line = lines[line_idx]
            # Find approximate edge position
            grad_line = np.abs(np.gradient(line))
            edge_approx = np.argmax(grad_line)

            # Fit sigmoid for sub-pixel edge position
            try:
                x = np.arange(len(line), dtype=np.float64)
                p0 = [np.ptp(line), 2.0, float(edge_approx), np.min(line)]
                popt, _ = curve_fit(
                    lambda x, a, k, x0, b: a / (1.0 + np.exp(-k * (x - x0))) + b,
                    x, line, p0=p0, maxfev=500,
                )
                x0_subpx = popt[2]
            except (RuntimeError, ValueError):
                x0_subpx = float(edge_approx)

            # Shift and accumulate into oversampled bins
            for px_idx in range(n_pixels):
                shifted = (px_idx - x0_subpx) * oversampling_factor + half
                bin_idx = int(round(shifted))
                if 0 <= bin_idx < n_bins:
                    esf_accum[bin_idx] += line[px_idx]
                    esf_count[bin_idx] += 1

        # Average bins
        valid = esf_count > 0
        esf = np.zeros(n_bins)
        esf[valid] = esf_accum[valid] / esf_count[valid]

        # Interpolate zeros
        if np.any(~valid):
            valid_idx = np.where(valid)[0]
            if len(valid_idx) > 2:
                esf = np.interp(np.arange(n_bins), valid_idx, esf[valid_idx])

        # Smooth — savgol_filter(window=5, polyorder=3)
        if len(esf) > 5:
            esf = savgol_filter(esf, window_length=5, polyorder=3)

        esf_axis = (np.arange(n_bins) - half) * sub_pixel

        return esf, esf_axis

    def _differentiate_esf_to_lsf(
        self, esf: np.ndarray, esf_axis_mm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """LSF = d(ESF)/dx using central differences. Apply Hanning window."""
        lsf = np.gradient(esf, esf_axis_mm)
        # Apply Hanning window to suppress sidelobes before FFT
        window = np.hanning(len(lsf))
        lsf_windowed = lsf * window
        return lsf_windowed, esf_axis_mm.copy()

    def _lsf_to_mtf(
        self, lsf: np.ndarray, lsf_axis_mm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute MTF from LSF via FFT. MTF(0) = 1.0 by normalization.

        MTF(f) = |FFT(lsf)| / |FFT(lsf)|_{f=0}
        """
        n = len(lsf)
        if n == 0:
            return np.array([]), np.array([])

        # Spacing in mm
        if len(lsf_axis_mm) > 1:
            spacing = float(np.abs(lsf_axis_mm[1] - lsf_axis_mm[0]))
        else:
            spacing = 1.0

        # FFT
        f_transform = np.fft.fft(lsf)
        mtf_full = np.abs(f_transform)

        # Normalize so MTF(0) = 1.0
        dc_val = mtf_full[0]
        if dc_val > 1e-12:
            mtf_full = mtf_full / dc_val

        # Positive frequency half only
        n_half = n // 2
        mtf_values = mtf_full[:n_half]
        freq_axis = np.fft.fftfreq(n, d=spacing)[:n_half]

        return freq_axis, mtf_values

    def _locate_bead_centroid(
        self, roi_hu: np.ndarray, pixel_spacing_mm: float,
    ) -> tuple[float, float]:
        """Intensity-weighted centroid for pixels above 0.5 * max.

        Returns (row_centroid, col_centroid) with sub-pixel precision.
        """
        roi = roi_hu.copy()
        threshold = 0.5 * np.max(roi)
        mask = roi > threshold
        if not np.any(mask):
            return float(roi.shape[0] / 2), float(roi.shape[1] / 2)

        ii, jj = np.meshgrid(
            np.arange(roi.shape[0], dtype=np.float64),
            np.arange(roi.shape[1], dtype=np.float64),
            indexing='ij',
        )
        weights = roi[mask]
        total = np.sum(weights)
        if total < 1e-12:
            return float(roi.shape[0] / 2), float(roi.shape[1] / 2)

        row_c = float(np.sum(weights * ii[mask]) / total)
        col_c = float(np.sum(weights * jj[mask]) / total)
        return row_c, col_c

    def _extract_psf_from_bead(
        self, roi_hu: np.ndarray, centroid: tuple[float, float],
        pixel_spacing_mm: float, oversampling_factor: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute radial PSF from bead image. Apply Hanning window."""
        rows, cols = roi_hu.shape
        ii, jj = np.meshgrid(
            np.arange(rows, dtype=np.float64),
            np.arange(cols, dtype=np.float64),
            indexing='ij',
        )
        radial_dist = np.sqrt(
            (ii - centroid[0]) ** 2 + (jj - centroid[1]) ** 2
        ) * pixel_spacing_mm  # mm

        sub_pixel = pixel_spacing_mm / oversampling_factor
        max_radius = np.max(radial_dist)
        n_bins = int(max_radius / sub_pixel) + 1
        bins = np.arange(n_bins + 1) * sub_pixel

        psf = np.zeros(n_bins)
        counts = np.zeros(n_bins)

        for i in range(rows):
            for j in range(cols):
                r = radial_dist[i, j]
                bin_idx = int(r / sub_pixel)
                if bin_idx < n_bins:
                    psf[bin_idx] += roi_hu[i, j]
                    counts[bin_idx] += 1

        valid = counts > 0
        psf[valid] /= counts[valid]

        # Subtract background (last 20% of bins)
        bg_start = int(0.8 * n_bins)
        if bg_start < n_bins:
            bg = np.mean(psf[bg_start:])
            psf -= bg

        # Apply Hanning window
        window = np.hanning(n_bins)
        psf *= window

        radial_axis = (np.arange(n_bins) + 0.5) * sub_pixel
        return psf, radial_axis

    def _find_mtf_at_value(
        self, freq_axis: np.ndarray, mtf_values: np.ndarray, target_mtf: float,
    ) -> float:
        """Find frequency where MTF drops to target_mtf via interpolation.

        Returns float('nan') if MTF never reaches target.
        """
        if len(freq_axis) == 0 or len(mtf_values) == 0:
            return float('nan')

        # MTF should be monotonically decreasing; find where it crosses target
        if np.min(mtf_values) > target_mtf:
            logger.warning("MTF never reaches %.2f; min is %.4f", target_mtf, np.min(mtf_values))
            return float('nan')

        # Interpolate: find first crossing
        for i in range(1, len(mtf_values)):
            if mtf_values[i] <= target_mtf <= mtf_values[i - 1]:
                # Linear interpolation
                f1, f2 = freq_axis[i - 1], freq_axis[i]
                m1, m2 = mtf_values[i - 1], mtf_values[i]
                if abs(m1 - m2) < 1e-12:
                    return float(f1)
                freq = f1 + (target_mtf - m1) * (f2 - f1) / (m2 - m1)
                return float(freq)

        return float('nan')
