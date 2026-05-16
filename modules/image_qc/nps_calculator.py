# -*- coding: utf-8 -*-
"""
modules/image_qc/nps_calculator.py — Noise Power Spectrum Calculator.

Computes 2D and radially-averaged 1D NPS from volumetric HU arrays
per AAPM TG-233 methodology.

Physics:
    NPS(fx,fy) = (dx*dy)/(Nx*Ny) * |FFT2D[I_noise(x,y)]|^2
    Units: HU^2 * mm^2

Hardware failure mapping:
    NPS peak frequency shift → X-ray tube filament degradation
    NPS integral increase → tube output instability

References:
    - AAPM TG-233 (2019) Eq. 1
    - Boedeker et al. Med. Phys. 34(7) 2007
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

import numpy as np

from config import CONFIG, ImageQCConfig

if TYPE_CHECKING:
    from modules.image_qc.roi_stats import VolumetricQCResult

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = ["NPSCalculator", "NPSResult", "NPSPatch", "InsufficientPatchesError"]


class InsufficientPatchesError(ValueError):
    """Raised when too few patches can be extracted for a stable NPS estimate."""


@dataclass
class NPSPatch:
    """Single de-trended windowed noise patch."""
    patch_index: int
    row_origin: int
    col_origin: int
    raw_hu: np.ndarray
    detrended: np.ndarray
    windowed: np.ndarray
    power_spectrum_2d: np.ndarray


@dataclass
class NPSResult:
    """
    Complete NPS result for one CT series.

    Hardware failure mapping:
      nps_peak_frequency_lpmm — tracks X-ray tube filament wear.
        Downward drift of peak frequency indicates increasing low-frequency
        noise, consistent with non-uniform filament emission.
      nps_integral — tracks total noise power.
        Sustained increase indicates tube output instability.
    Reference: AAPM TG-233; clinical convention for tube wear monitoring.
    """
    series_description: str
    acquisition_date: str
    pixel_spacing_mm: tuple[float, float]
    patch_size_px: int
    n_patches_used: int
    n_slices_used: int
    nps_2d: np.ndarray
    nps_1d: np.ndarray
    freq_axis_lpmm: np.ndarray
    nps_peak_frequency_lpmm: float
    nps_peak_value: float
    nps_integral: float
    noise_std_from_nps: float

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    def passes_frequency_drift_check(
        self, reference_freq: float, tolerance_lpmm: float = 0.05
    ) -> bool:
        """Returns True if peak frequency is within tolerance of reference."""
        return abs(self.nps_peak_frequency_lpmm - reference_freq) <= tolerance_lpmm


class NPSCalculator:
    """Computes Noise Power Spectrum from volumetric CT data.

    Physics: NPS quantifies the spatial frequency distribution of noise.
    Reference: AAPM TG-233 (2019)
    """

    def __init__(self, config: ImageQCConfig) -> None:
        self._config = config

    def compute_from_volume(self, volumetric_result: "VolumetricQCResult") -> NPSResult:
        """Primary entry point. Consumes VolumetricQCResult.hu_arrays directly."""
        return self.compute_from_hu_arrays(
            hu_arrays=volumetric_result.hu_arrays,
            pixel_spacing_mm=volumetric_result.pixel_spacing_mm,
            series_description=volumetric_result.series_description,
            acquisition_date=volumetric_result.acquisition_date,
        )

    def compute_from_hu_arrays(
        self,
        hu_arrays: list[np.ndarray],
        pixel_spacing_mm: tuple[float, float],
        series_description: str = "",
        acquisition_date: str = "",
    ) -> NPSResult:
        """Compute NPS from HU arrays using the bulletproof TG-233 pipeline.

        Signal processing per patch:
          1. Mean subtraction (detrending)
          2. 2D Hanning window (spectral leakage suppression)
          3. FFT2 → |F|² → normalization

        Post-processing:
          4. Ensemble average of all patch power spectra
          5. Radial averaging via np.bincount
          6. Hard cutoff at 0.04 lp/mm (cupping artifact removal)
        """
        patch_size = self._config.nps_patch_size_px

        # Extract patches from all slices
        all_patches: list[NPSPatch] = []
        for hu_array in hu_arrays:
            patches = self._extract_patches(hu_array, patch_size)
            all_patches.extend(patches)

        if len(all_patches) < 1:
            raise InsufficientPatchesError(
                "No patches extracted from %d slices" % len(hu_arrays)
            )
        if len(all_patches) < self._config.nps_n_patches_min:
            logger.warning(
                "Only %d patches extracted (config min=%d). "
                "Using absolute center crop — proceeding with available patches.",
                len(all_patches), self._config.nps_n_patches_min,
            )

        dx = pixel_spacing_mm[1]  # col spacing mm
        dy = pixel_spacing_mm[0]  # row spacing mm
        pixel_area = dx * dy

        # Pre-compute 2D Hanning window (same size for all patches)
        wy = np.hanning(patch_size)
        wx = np.hanning(patch_size)
        window_2d = np.outer(wy, wx)

        # Process each patch and accumulate power spectra
        power_spectra = []
        for patch in all_patches:
            roi = patch.raw_hu.astype(np.float64)

            # 1. Detrend: subtract mean
            roi_detrend = roi - np.mean(roi)

            # 2. Apply 2D Hanning window
            roi_windowed = roi_detrend * window_2d

            # 3. FFT → power spectrum, normalized
            fft_result = np.fft.fftshift(np.fft.fft2(roi_windowed))
            power_2d = np.abs(fft_result) ** 2
            ny, nx = roi_windowed.shape
            power_2d = power_2d * pixel_area / (nx * ny)

            # Store for dataclass (last patch only, to save memory)
            patch.detrended = roi_detrend
            patch.windowed = roi_windowed
            patch.power_spectrum_2d = power_2d

            power_spectra.append(power_2d)

        # 4. Ensemble average
        nps_2d = np.mean(np.stack(power_spectra, axis=0), axis=0)

        # 5. Radial averaging using np.bincount (robust, no loop)
        ny, nx = nps_2d.shape
        center_y, center_x = ny // 2, nx // 2
        Y, X = np.indices((ny, nx))
        R = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2).astype(int)

        tbin = np.bincount(R.ravel(), nps_2d.ravel())
        nr = np.bincount(R.ravel())
        radial_profile = tbin / np.maximum(nr, 1)

        # Frequency axis (cycles/mm = lp/mm)
        freq_axis = np.arange(len(radial_profile)) / (max(ny, nx) * dx)

        # 6. Hard cutoff: zero out frequencies < 0.04 lp/mm
        #    These represent macroscopic cupping/beam hardening, NOT noise.
        cutoff_freq = 0.04  # lp/mm
        low_freq_mask = freq_axis < cutoff_freq
        radial_profile[low_freq_mask] = 0.0

        # Trim to Nyquist frequency
        nyquist = 1.0 / (2.0 * dx)
        nyquist_mask = freq_axis <= nyquist
        freq_axis = freq_axis[nyquist_mask]
        nps_1d = radial_profile[nyquist_mask]

        # Peak search (on filtered data)
        if len(nps_1d) > 1:
            peak_idx = int(np.argmax(nps_1d))
            nps_peak_frequency = float(freq_axis[peak_idx])
            nps_peak_value = float(nps_1d[peak_idx])
        else:
            nps_peak_frequency = 0.0
            nps_peak_value = 0.0

        # Integral (valid range only)
        valid = freq_axis >= cutoff_freq
        if np.any(valid):
            _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
            nps_integral = float(_trapz(nps_1d[valid], freq_axis[valid]))
        else:
            nps_integral = 0.0

        noise_std = float(np.sqrt(abs(nps_integral)))

        logger.info(
            "NPS computed: %d patches from %d slices, peak=%.3f lp/mm, "
            "integral=%.2f HU^2, SD=%.2f HU",
            len(all_patches), len(hu_arrays), nps_peak_frequency,
            nps_integral, noise_std,
        )

        return NPSResult(
            series_description=series_description,
            acquisition_date=acquisition_date,
            pixel_spacing_mm=pixel_spacing_mm,
            patch_size_px=patch_size,
            n_patches_used=len(all_patches),
            n_slices_used=len(hu_arrays),
            nps_2d=nps_2d,
            nps_1d=nps_1d,
            freq_axis_lpmm=freq_axis,
            nps_peak_frequency_lpmm=nps_peak_frequency,
            nps_peak_value=nps_peak_value,
            nps_integral=nps_integral,
            noise_std_from_nps=noise_std,
        )

    def _extract_patches(self, hu_array: np.ndarray, patch_size: int) -> list[NPSPatch]:
        """Extract a strict central patch from the TRUE phantom center.

        Uses circularity-filtered connected component analysis to find
        the phantom body (excluding patient couch), then extracts a
        single patch_size × patch_size ROI from its centroid.

        The couch is excluded by requiring aspect_ratio > 0.7 (circular)
        and fill_ratio > 0.5, which rejects elongated structures.

        Physics rationale:
          A uniform QA phantom (water section) should have noise SD ~4-8 HU.
          If the ROI captures the phantom edge (air = -1000 HU) or
          high-contrast inserts, noise SD can spike to >100 HU, producing
          a catastrophic 1e6 low-frequency artifact in the NPS FFT.

        Reference: AAPM TG-233 Section 4 — ROI placement for NPS.
        """
        from scipy import ndimage

        rows, cols = hu_array.shape[:2]

        # ── Step 1: Find true phantom center (couch-safe) ─────────────
        mask = hu_array > -300
        cy, cx = rows // 2, cols // 2  # default fallback

        if np.any(mask):
            filled = ndimage.binary_fill_holes(mask)
            labeled, n_labels = ndimage.label(filled)

            best_label = None
            best_area = 0
            for lbl in range(1, n_labels + 1):
                component = labeled == lbl
                area = int(component.sum())
                if area < 1000:
                    continue
                rr, cc_arr = np.where(component)
                bbox_h = rr.max() - rr.min() + 1
                bbox_w = cc_arr.max() - cc_arr.min() + 1
                aspect = min(bbox_h, bbox_w) / max(bbox_h, bbox_w)
                fill_ratio = area / (bbox_h * bbox_w)
                if aspect > 0.7 and fill_ratio > 0.5 and area > best_area:
                    best_area = area
                    best_label = lbl

            if best_label is None and n_labels > 0:
                sizes = ndimage.sum(filled, labeled, range(1, n_labels + 1))
                best_label = int(np.argmax(sizes)) + 1

            if best_label is not None:
                com = ndimage.center_of_mass(
                    (labeled == best_label).astype(float))
                cy, cx = int(com[0]), int(com[1])

            logger.debug(
                "NPS center detection (couch-safe): (cy=%d, cx=%d)",
                cy, cx,
            )
        else:
            logger.warning(
                "NPS mask empty (no pixels > -300 HU). "
                "Falling back to image center (%d, %d).", cy, cx,
            )

        # ── Step 2: Strict central crop (64×64 default) ───────────────
        half = patch_size // 2

        # Verify the crop fits within the image at the detected center
        if (cy - half >= 0 and cy + half <= rows and
                cx - half >= 0 and cx + half <= cols):
            r0, r1 = cy - half, cy + half
            c0, c1 = cx - half, cx + half
        else:
            # Fallback to absolute image center if mask center is too close
            # to the edge (should never happen with a centered phantom)
            logger.warning(
                "NPS phantom center (%d, %d) too close to image edge for "
                "%dx%d crop. Falling back to image center.",
                cy, cx, patch_size, patch_size,
            )
            cy, cx = rows // 2, cols // 2
            r0 = max(0, cy - half)
            r1 = min(rows, cy + half)
            c0 = max(0, cx - half)
            c1 = min(cols, cx + half)

        roi = hu_array[r0:r1, c0:c1].copy().astype(np.float64)

        # ── Step 3: Safety validation — reject contaminated ROI ───────
        roi_std = float(np.std(roi))
        if roi_std > 50.0:
            logger.warning(
                "NPS ROI std=%.1f HU (expected <10) at center=(%d,%d). "
                "Falling back to tighter %dx%d center crop.",
                roi_std, cy, cx, patch_size // 2, patch_size // 2,
            )
            # Tighter crop: half the patch size, still at detected center
            tight_half = patch_size // 4
            r0 = max(0, cy - tight_half)
            r1 = min(rows, cy + tight_half)
            c0 = max(0, cx - tight_half)
            c1 = min(cols, cx + tight_half)
            roi = hu_array[r0:r1, c0:c1].copy().astype(np.float64)

            roi_std = float(np.std(roi))
            logger.info(
                "NPS tight crop: ROI=[%d:%d, %d:%d], size=%s, std=%.2f HU",
                r0, r1, c0, c1, roi.shape, roi_std,
            )
        # ── Step 4: Size validation — reject undersized ROI ─────────
        if roi.shape[0] < patch_size or roi.shape[1] < patch_size:
            logger.warning(
                "NPS ROI too small: %s (need %dx%d). Image too small for NPS.",
                roi.shape, patch_size, patch_size,
            )
            return []  # Triggers InsufficientPatchesError upstream

        patches = [NPSPatch(
            patch_index=0, row_origin=r0, col_origin=c0,
            raw_hu=roi, detrended=np.empty(0),
            windowed=np.empty(0), power_spectrum_2d=np.empty(0),
        )]

        logger.debug(
            "NPS center crop: detected_center=(%d,%d), ROI=[%d:%d, %d:%d], "
            "size=%s, std=%.2f HU",
            cy, cx, r0, r1, c0, c1, roi.shape, float(np.std(roi)),
        )
        return patches

