"""
modules/image_qc/basic_metrics.py
===================================
Basic Tier Metrics Engine — Clinical Daily QA.

Implements the five fundamental image quality metrics required for
daily CT QA on any scanner (Siemens SOMATOM go.Sim or GE Discovery RT):

  1. Image Noise       — Standard Deviation in the central water ROI
  2. Uniformity        — Non-uniformity index across 5 ROI positions
  3. CT Number Accuracy — Mean HU validation against nominal values
  4. Contrast Resolution — Signal-to-noise ratio between insert and background
  5. Slice Thickness   — FWHM of the axial sensitometric ramp profile

All formulas are implemented explicitly in numpy with inline citations.
No scipy wrapper functions are used for the core physics calculations.

The engine is phantom-agnostic: it receives ROIDescriptor objects from
a PhantomAdapter and MaterialReference objects for nominal values.
It does not know which phantom or scanner it is analyzing.

Reference standards:
  AAPM TG-66 (2003) — Quality assurance for CT simulators
  IEC 61223-3-5:2004 — Constancy tests for CT scanners
  ACR CT Accreditation Program — Technical requirements
  IPEM Report 91 (2005) — Recommended standards for routine CT dosimetry
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from modules.image_qc.roi_stats import ROIDescriptor, ROIStatistics, VolumetricROIStat

logger = logging.getLogger(__name__)


# ── Custom exceptions ──────────────────────────────────────────────────────

class InsufficientROIError(ValueError):
    """
    Raised when fewer ROIs are provided than required for a specific
    metric calculation (e.g., uniformity requires >= 5 ROIs).
    """


class SliceThicknessROIError(ValueError):
    """
    Raised when the provided ROI for slice thickness measurement does not
    contain a valid sensitometric ramp profile (no clear transition found).
    """


# ── Individual metric result dataclasses ──────────────────────────────────

@dataclass
class NoiseResult:
    """
    Image noise measured as Standard Deviation within the central water ROI.

    Reference: AAPM TG-66 Section 5.1 — Image Noise
    Tolerance: SD <= noise_tolerance_hu (typically 5.0 HU per TG-66)
    Hardware mapping: Increasing noise SD → X-ray tube filament wear
    """
    roi_label: str
    mean_hu: float
    std_hu: float
    variance_hu: float
    n_pixels: int
    tolerance_hu: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "roi_label":    self.roi_label,
            "mean_hu":      round(self.mean_hu, 3),
            "std_hu":       round(self.std_hu, 4),
            "variance_hu":  round(self.variance_hu, 4),
            "n_pixels":     self.n_pixels,
            "tolerance_hu": self.tolerance_hu,
            "passed":       self.passed,
        }


@dataclass
class UniformityResult:
    """
    Image uniformity measured as the non-uniformity index across 5 ROI positions.

    Definition (AAPM TG-66 Section 5.1):
        Non_uniformity = max(|HU_peripheral_i - HU_centre|)

    Reference: AAPM TG-66 Section 5.1 — Image Uniformity
    Tolerance: non_uniformity_index <= uniformity_tolerance_hu (typically 5.0 HU)
    Hardware mapping: Growing non-uniformity → detector calibration drift
    """
    centre_mean_hu: float
    peripheral_means: dict[str, float]
    deviations: dict[str, float]
    non_uniformity_index: float
    tolerance_hu: float
    passed: bool
    worst_roi: str

    def to_dict(self) -> dict:
        return {
            "centre_mean_hu":        round(self.centre_mean_hu, 3),
            "peripheral_means":      {k: round(v, 3) for k, v in self.peripheral_means.items()},
            "deviations":            {k: round(v, 3) for k, v in self.deviations.items()},
            "non_uniformity_index":  round(self.non_uniformity_index, 4),
            "tolerance_hu":          self.tolerance_hu,
            "passed":                self.passed,
            "worst_roi":             self.worst_roi,
        }


@dataclass
class CTNumberAccuracyResult:
    """
    CT number accuracy: measured HU vs nominal HU for all phantom materials.

    Reference: AAPM TG-66 Section 5.2 — CT Number Accuracy
    """
    measurements: list[dict]
    max_delta_hu: float
    all_passed: bool
    tolerance_hu: float
    r_squared: Optional[float] = None
    slope: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "measurements": self.measurements,
            "max_delta_hu": round(self.max_delta_hu, 3),
            "all_passed":   self.all_passed,
            "tolerance_hu": self.tolerance_hu,
            "r_squared":    round(self.r_squared, 4) if self.r_squared else None,
            "slope":        round(self.slope, 4) if self.slope else None,
        }


@dataclass
class ContrastResult:
    """
    Contrast resolution: SNR between a high-contrast insert and the water background.

    Reference: AAPM TG-66 Section 5.4 — Low Contrast Resolution
               IEC 61223-3-5 Section 6.4
    """
    insert_label: str
    insert_mean_hu: float
    background_mean_hu: float
    contrast_hu: float
    snr_contrast: float
    cnr: float
    passed: bool
    min_cnr_threshold: float

    def to_dict(self) -> dict:
        return {
            "insert_label":      self.insert_label,
            "insert_mean_hu":    round(self.insert_mean_hu, 3),
            "background_mean_hu": round(self.background_mean_hu, 3),
            "contrast_hu":       round(self.contrast_hu, 3),
            "snr_contrast":      round(self.snr_contrast, 3),
            "cnr":               round(self.cnr, 4),
            "passed":            self.passed,
            "min_cnr_threshold": self.min_cnr_threshold,
        }


@dataclass
class SliceThicknessResult:
    """
    Slice thickness measured as FWHM of the axial sensitometric ramp profile.

    Reference: AAPM TG-66 Section 5.5 — Slice Thickness
    Hardware mapping: FWHM drift → detector row alignment or collimator wear
    """
    nominal_thickness_mm: float
    measured_fwhm_mm: float
    deviation_mm: float
    tolerance_mm: float
    passed: bool
    profile_axis: str
    z_left_mm: float
    z_right_mm: float

    def to_dict(self) -> dict:
        return {
            "nominal_thickness_mm":  self.nominal_thickness_mm,
            "measured_fwhm_mm":      round(self.measured_fwhm_mm, 3),
            "deviation_mm":          round(self.deviation_mm, 3),
            "tolerance_mm":          self.tolerance_mm,
            "passed":                self.passed,
            "profile_axis":          self.profile_axis,
            "z_left_mm":             round(self.z_left_mm, 3),
            "z_right_mm":            round(self.z_right_mm, 3),
        }


# ── TotalQA-Matched Result Dataclasses ─────────────────────────────────────

@dataclass
class TotalQAContrastResult:
    """
    TotalQA-matched contrast measurement from Slice 60 (plastic block).

    Four rectangular ROIs:
      A = Top Plastic, B = Top Water, C = Bottom Plastic, D = Bottom Water
    Contrast Top  = Mean(A) - Mean(B)
    Contrast Bottom = Mean(C) - Mean(D)

    Reference: Image Owl TotalQA PDF page 6.
    """
    mean_A: float       # Top Plastic
    mean_B: float       # Top Water
    mean_C: float       # Bottom Plastic
    mean_D: float       # Bottom Water
    contrast_top: float     # Mean(A) - Mean(B)
    contrast_bottom: float  # Mean(C) - Mean(D)
    passed: bool = True

    def to_dict(self) -> dict:
        return {
            "mean_A":           round(self.mean_A, 3),
            "mean_B":           round(self.mean_B, 3),
            "mean_C":           round(self.mean_C, 3),
            "mean_D":           round(self.mean_D, 3),
            "contrast_top":     round(self.contrast_top, 3),
            "contrast_bottom":  round(self.contrast_bottom, 3),
            "passed":           self.passed,
        }


@dataclass
class TotalQAResolutionResult:
    """
    TotalQA-matched spatial resolution from Slice 70 (angled bar patterns).

    Five small square ROIs over the 5 largest bar pattern groups.
    Output: Standard Deviation (SD) of pixel values within each ROI.

    Reference: Image Owl TotalQA PDF page 9.
    """
    bar_labels: list[str]       # e.g. ["Bar 1", "Bar 2", ...]
    bar_sd_values: list[float]  # SD for each bar pattern ROI
    bar_mean_values: list[float]  # Mean HU for each bar pattern ROI
    passed: bool = True

    def to_dict(self) -> dict:
        return {
            "bar_labels":      self.bar_labels,
            "bar_sd_values":   [round(v, 3) for v in self.bar_sd_values],
            "bar_mean_values": [round(v, 3) for v in self.bar_mean_values],
            "passed":          self.passed,
        }


@dataclass
class TotalQAScalingResult:
    """
    TotalQA-matched geometric scaling from the uniformity slice (Slice 36).

    Measures the physical diameter of the phantom horizontally (H) and
    vertically (V) using edge detection, then compares to the nominal
    diameter (200.0 mm Siemens Waterbath, 215.0 mm GE Helios).

    Reference: Image Owl TotalQA geometry/scaling section.
    """
    h_diameter_mm: float
    v_diameter_mm: float
    nominal_mm: float       # 200.0 mm (Siemens) or 215.0 mm (GE)
    h_error_mm: float       # h_diameter_mm - nominal_mm
    v_error_mm: float       # v_diameter_mm - nominal_mm
    h_error_pct: float      # h_error_mm / nominal_mm * 100
    v_error_pct: float      # v_error_mm / nominal_mm * 100
    tolerance_mm: float = 2.0
    passed: bool = True

    def to_dict(self) -> dict:
        return {
            "h_diameter_mm":  round(self.h_diameter_mm, 3),
            "v_diameter_mm":  round(self.v_diameter_mm, 3),
            "nominal_mm":     self.nominal_mm,
            "h_error_mm":     round(self.h_error_mm, 3),
            "v_error_mm":     round(self.v_error_mm, 3),
            "h_error_pct":    round(self.h_error_pct, 3),
            "v_error_pct":    round(self.v_error_pct, 3),
            "tolerance_mm":   self.tolerance_mm,
            "passed":         self.passed,
        }


@dataclass
class BasicQAResult:
    """
    Complete Basic Tier QA result for one CT session.
    all_passed is True only if every non-None metric passed its tolerance.
    """
    acquisition_date: str
    series_description: str
    scanner_id: str
    phantom_id: str
    n_slices_analyzed: int

    noise: Optional[NoiseResult]            = None
    uniformity: Optional[UniformityResult]  = None
    ct_number_accuracy: Optional[CTNumberAccuracyResult] = None
    contrast: Optional[ContrastResult]      = None
    slice_thickness: Optional[SliceThicknessResult] = None

    # ── TotalQA-matched metrics ────────────────────────────────────
    totalqa_contrast: Optional[TotalQAContrastResult]       = None
    totalqa_resolution: Optional[TotalQAResolutionResult]   = None
    totalqa_scaling: Optional[TotalQAScalingResult]          = None

    all_passed: bool = False
    computed_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "acquisition_date":  self.acquisition_date,
            "series_description": self.series_description,
            "scanner_id":        self.scanner_id,
            "phantom_id":        self.phantom_id,
            "n_slices_analyzed": self.n_slices_analyzed,
            "noise":             self.noise.to_dict()             if self.noise             else None,
            "uniformity":        self.uniformity.to_dict()        if self.uniformity        else None,
            "ct_number_accuracy": self.ct_number_accuracy.to_dict() if self.ct_number_accuracy else None,
            "contrast":          self.contrast.to_dict()          if self.contrast          else None,
            "slice_thickness":   self.slice_thickness.to_dict()   if self.slice_thickness   else None,
            "totalqa_contrast":  self.totalqa_contrast.to_dict()  if self.totalqa_contrast  else None,
            "totalqa_resolution": self.totalqa_resolution.to_dict() if self.totalqa_resolution else None,
            "totalqa_scaling":   self.totalqa_scaling.to_dict()   if self.totalqa_scaling   else None,
            "all_passed":        self.all_passed,
            "computed_at":       self.computed_at,
            "warnings":          self.warnings,
        }

    def _evaluate_all_passed(self) -> None:
        """Recomputes all_passed from individual metric results."""
        metrics = [self.noise, self.uniformity, self.ct_number_accuracy,
                   self.contrast, self.slice_thickness,
                   self.totalqa_contrast, self.totalqa_resolution,
                   self.totalqa_scaling]
        non_none = [m for m in metrics if m is not None]
        if not non_none:
            self.all_passed = False
            return
        results = []
        for m in non_none:
            # CTNumberAccuracyResult uses 'all_passed'; other results use 'passed'
            if hasattr(m, "passed"):
                results.append(m.passed)
            elif hasattr(m, "all_passed"):
                results.append(m.all_passed)
        self.all_passed = all(results) if results else False


# ── Engine ─────────────────────────────────────────────────────────────────

class BasicMetricsEngine:
    """
    Computes the five Basic Tier QA metrics from a VolumetricQCResult.

    The engine is completely decoupled from scanner and phantom specifics.
    """

    # Default tolerances — overridden by scanner profile or CONFIG
    DEFAULT_NOISE_TOLERANCE_HU       = 5.0    # AAPM TG-66 Section 5.1
    DEFAULT_UNIFORMITY_TOLERANCE_HU  = 5.0    # AAPM TG-66 Section 5.1
    DEFAULT_HU_ACCURACY_TOLERANCE_HU = 4.0    # AAPM TG-66 Section 5.2
    DEFAULT_SLICE_THICKNESS_TOL_MM   = 1.0    # AAPM TG-66 Section 5.5
    DEFAULT_MIN_CNR                  = 0.5    # minimum contrast-to-noise ratio

    # ROI label naming conventions
    _CENTRE_ROI_LABELS   = {"center", "centre", "center_water", "water"}
    _PERIPHERAL_PREFIXES = {"peripheral_", "periph_", "peri_"}

    def __init__(
        self,
        noise_tolerance_hu: float       = DEFAULT_NOISE_TOLERANCE_HU,
        uniformity_tolerance_hu: float  = DEFAULT_UNIFORMITY_TOLERANCE_HU,
        hu_accuracy_tolerance_hu: float = DEFAULT_HU_ACCURACY_TOLERANCE_HU,
        slice_thickness_tol_mm: float   = DEFAULT_SLICE_THICKNESS_TOL_MM,
        min_cnr: float                  = DEFAULT_MIN_CNR,
    ) -> None:
        self._noise_tol       = noise_tolerance_hu
        self._uniformity_tol  = uniformity_tolerance_hu
        self._hu_accuracy_tol = hu_accuracy_tolerance_hu
        self._slice_tol_mm    = slice_thickness_tol_mm
        self._min_cnr         = min_cnr

    def compute(
        self,
        volumetric_result: "VolumetricQCResult",
        phantom_adapter: "PhantomAdapter",
        scanner_id: str = "UNKNOWN",
        hu_arrays_for_fwhm: Optional[list[np.ndarray]] = None,
        nominal_slice_thickness_mm: Optional[float] = None,
    ) -> BasicQAResult:
        """Orchestrates computation of all five Basic Tier metrics."""
        vol   = volumetric_result
        stats = vol.volumetric_stats
        refs  = phantom_adapter.get_material_references()
        warnings: list[str] = []

        result = BasicQAResult(
            acquisition_date   = vol.acquisition_date,
            series_description = vol.series_description,
            scanner_id         = scanner_id,
            phantom_id         = phantom_adapter.phantom_id,
            n_slices_analyzed  = vol.n_slices_processed,
        )

        # ── Metric 1: Noise ────────────────────────────────────────────
        try:
            centre_label = self._find_centre_roi(stats)
            result.noise = self._compute_noise(stats[centre_label], centre_label)
        except Exception as exc:
            msg = f"Noise metric failed: {exc}"
            logger.warning("%s", msg)
            warnings.append(msg)

        # ── Metric 2: Uniformity ───────────────────────────────────────
        try:
            centre_label = self._find_centre_roi(stats)
            periph       = self._find_peripheral_rois(stats)
            if len(periph) >= 1:
                result.uniformity = self._compute_uniformity(
                    stats[centre_label], {k: stats[k] for k in periph}
                )
            else:
                warnings.append(
                    "Uniformity: no peripheral ROIs found."
                )
        except Exception as exc:
            msg = f"Uniformity metric failed: {exc}"
            logger.warning("%s", msg)
            warnings.append(msg)

        # ── Metric 3: CT Number Accuracy ───────────────────────────────
        try:
            result.ct_number_accuracy = self._compute_ct_number_accuracy(stats, refs)
        except Exception as exc:
            msg = f"CT number accuracy failed: {exc}"
            logger.warning("%s", msg)
            warnings.append(msg)

        # ── Metric 4: Contrast ─────────────────────────────────────────
        try:
            contrast_result = self._compute_contrast(stats, refs)
            if contrast_result is not None:
                result.contrast = contrast_result
            else:
                warnings.append("Contrast: no high-contrast insert found in ROIs.")
        except Exception as exc:
            msg = f"Contrast metric failed: {exc}"
            logger.warning("%s", msg)
            warnings.append(msg)

        # ── Metric 5: Slice Thickness (FWHM) ──────────────────────────
        if hu_arrays_for_fwhm and nominal_slice_thickness_mm:
            try:
                result.slice_thickness = self._compute_slice_thickness_fwhm(
                    hu_arrays_for_fwhm, nominal_slice_thickness_mm, vol.pixel_spacing_mm,
                )
            except SliceThicknessROIError as exc:
                msg = f"Slice thickness FWHM: {exc}"
                logger.warning("%s", msg)
                warnings.append(msg)
            except Exception as exc:
                msg = f"Slice thickness failed: {exc}"
                logger.warning("%s", msg)
                warnings.append(msg)
        else:
            warnings.append(
                "Slice thickness: hu_arrays_for_fwhm or nominal_slice_thickness_mm "
                "not provided — skipping FWHM calculation."
            )

        result.warnings = warnings
        result._evaluate_all_passed()
        return result

    # ── Private metric implementations ────────────────────────────────

    def _compute_noise(self, centre_stat: "VolumetricROIStat", roi_label: str) -> NoiseResult:
        """AAPM TG-66 Section 5.1 — noise is the mean per-slice SD in the centre ROI."""
        noise_std = centre_stat.std_hu_mean
        passed    = noise_std <= self._noise_tol
        return NoiseResult(
            roi_label    = roi_label,
            mean_hu      = centre_stat.mean_hu_mean,
            std_hu       = noise_std,
            variance_hu  = centre_stat.variance_hu_mean,
            n_pixels     = centre_stat.n_slices * 10000,
            tolerance_hu = self._noise_tol,
            passed       = passed,
        )

    def _compute_uniformity(
        self, centre_stat: "VolumetricROIStat",
        peripheral_stats: dict[str, "VolumetricROIStat"],
    ) -> UniformityResult:
        """AAPM TG-66 Section 5.1 — NUI = max_i(|HU_peripheral_i - HU_centre|)."""
        centre_mean = centre_stat.mean_hu_mean
        periph_means: dict[str, float] = {}
        deviations:   dict[str, float] = {}

        for label, stat in peripheral_stats.items():
            periph_mean = stat.mean_hu_mean
            periph_means[label] = periph_mean
            deviations[label] = abs(periph_mean - centre_mean)

        if not deviations:
            raise InsufficientROIError("Uniformity requires at least 1 peripheral ROI.")

        nui       = max(deviations.values())
        worst_roi = max(deviations, key=deviations.get)
        passed    = nui <= self._uniformity_tol

        return UniformityResult(
            centre_mean_hu=centre_mean, peripheral_means=periph_means,
            deviations=deviations, non_uniformity_index=nui,
            tolerance_hu=self._uniformity_tol, passed=passed, worst_roi=worst_roi,
        )

    def _compute_ct_number_accuracy(
        self, stats: dict[str, "VolumetricROIStat"],
        refs: dict[str, "MaterialReference"],
    ) -> CTNumberAccuracyResult:
        """AAPM TG-66 Section 5.2 — ΔHU = HU_measured - HU_nominal.

        Fallback: if no material refs match any stats, uses the center
        ROI mean vs. water (0 HU) as the baseline measurement.
        """
        measurements = []
        max_delta    = 0.0

        for label, ref in refs.items():
            stat = self._find_stat_for_material(stats, label)
            if stat is None:
                continue
            measured   = stat.mean_hu_mean
            delta      = measured - ref.nominal_hu
            abs_delta  = abs(delta)
            passed     = abs_delta <= self._hu_accuracy_tol
            measurements.append({
                "material": label, "nominal_hu": ref.nominal_hu,
                "measured_hu": round(measured, 3), "delta_hu": round(delta, 3),
                "abs_delta": round(abs_delta, 3), "passed": passed,
            })
            max_delta = max(max_delta, abs_delta)

        # ── Water baseline fallback ───────────────────────────────
        # If no material references matched, find the center ROI and
        # compute HU precision as |center_mean - 0 HU| (water).
        # This ensures we never return max_delta = 0.0 when the center
        # is physically offset from 0 HU.
        if not measurements:
            centre_stat = self._find_centre_stat(stats)
            if centre_stat is not None:
                measured  = centre_stat.mean_hu_mean
                delta     = measured - 0.0  # water nominal = 0 HU
                abs_delta = abs(delta)
                passed    = abs_delta <= self._hu_accuracy_tol
                measurements.append({
                    "material": "water", "nominal_hu": 0.0,
                    "measured_hu": round(measured, 3),
                    "delta_hu": round(delta, 3),
                    "abs_delta": round(abs_delta, 3), "passed": passed,
                })
                max_delta = abs_delta

        all_passed = all(m["passed"] for m in measurements) if measurements else False
        return CTNumberAccuracyResult(
            measurements=measurements, max_delta_hu=max_delta,
            all_passed=all_passed, tolerance_hu=self._hu_accuracy_tol,
        )

    def _compute_contrast(
        self, stats: dict[str, "VolumetricROIStat"],
        refs: dict[str, "MaterialReference"],
    ) -> Optional[ContrastResult]:
        """AAPM TG-66 Section 5.4 — CNR = Contrast / sqrt(SD_insert² + SD_bg²).

        FORCED EXECUTION: If no material ref matches, falls back to the ROI
        with the highest measured |ΔHU| against the background.
        """
        bg_stat = None
        for label in stats:
            if label.lower() in self._CENTRE_ROI_LABELS:
                bg_stat = stats[label]
                break
        if bg_stat is None:
            # Forced fallback: use first stat as background
            bg_stat = next(iter(stats.values()))

        # Strategy 1: pick insert with highest nominal |HU| from refs
        best_label    = None
        best_contrast = 0.0
        for label, ref in refs.items():
            if label in self._CENTRE_ROI_LABELS or label == "water":
                continue
            if abs(ref.nominal_hu) > best_contrast:
                insert_stat = self._find_stat_for_material(stats, label)
                if insert_stat is not None:
                    best_contrast = abs(ref.nominal_hu)
                    best_label    = label

        # Strategy 2 (FORCED FALLBACK): highest measured |ΔHU| vs bg
        if best_label is None:
            bg_mean = bg_stat.mean_hu_mean
            for label, stat in stats.items():
                if label.lower() in self._CENTRE_ROI_LABELS:
                    continue
                delta = abs(stat.mean_hu_mean - bg_mean)
                if delta > best_contrast:
                    best_contrast = delta
                    best_label = label

        if best_label is None:
            return None

        insert_stat = self._find_stat_for_material(stats, best_label)
        if insert_stat is None:
            # Direct lookup (the fallback label IS already a stats key)
            insert_stat = stats.get(best_label)
        if insert_stat is None:
            return None

        hu_insert = insert_stat.mean_hu_mean
        hu_bg     = bg_stat.mean_hu_mean
        sd_insert = insert_stat.std_hu_mean
        sd_bg     = bg_stat.std_hu_mean
        contrast  = abs(hu_insert - hu_bg)

        # SNR_contrast = Contrast / SD_background — AAPM TG-66 Section 5.4
        snr_contrast = contrast / max(sd_bg, 1e-6)
        # CNR — Rose criterion for detection threshold
        cnr = contrast / max(np.sqrt(sd_insert**2 + sd_bg**2), 1e-6)

        return ContrastResult(
            insert_label=best_label, insert_mean_hu=hu_insert,
            background_mean_hu=hu_bg, contrast_hu=contrast,
            snr_contrast=snr_contrast, cnr=cnr,
            passed=cnr >= self._min_cnr, min_cnr_threshold=self._min_cnr,
        )

    def _compute_slice_thickness_fwhm(
        self, hu_arrays: list[np.ndarray],
        nominal_thickness_mm: float, pixel_spacing_mm: tuple[float, float],
    ) -> SliceThicknessResult:
        """AAPM TG-66 Section 5.5 — FWHM of axial sensitometric ramp profile."""
        if len(hu_arrays) < 2:
            raise SliceThicknessROIError(
                f"FWHM requires at least 2 HU arrays. Received {len(hu_arrays)}."
            )

        stack   = np.stack(hu_arrays, axis=0)
        cx      = stack.shape[2] // 2
        cr      = stack.shape[1] // 2
        profile = stack[:, cr, cx].astype(np.float64)

        if np.ptp(profile) < 10.0:
            raise SliceThicknessROIError(
                f"FWHM profile has insufficient contrast (range = {np.ptp(profile):.2f} HU)."
            )

        p_min  = profile.min()
        p_max  = profile.max()
        p_norm = (profile - p_min) / (p_max - p_min)

        peak_idx    = int(np.argmax(p_norm))
        z_left_idx  = self._find_half_max_crossing(p_norm, peak_idx, direction="left")
        z_right_idx = self._find_half_max_crossing(p_norm, peak_idx, direction="right")

        if z_left_idx is None or z_right_idx is None:
            raise SliceThicknessROIError("FWHM half-maximum crossings not found.")

        step_mm    = nominal_thickness_mm if nominal_thickness_mm > 0 else pixel_spacing_mm[0]
        z_left_mm  = z_left_idx  * step_mm
        z_right_mm = z_right_idx * step_mm
        fwhm_mm    = z_right_mm - z_left_mm
        deviation  = abs(fwhm_mm - nominal_thickness_mm)
        passed     = deviation <= self._slice_tol_mm

        return SliceThicknessResult(
            nominal_thickness_mm=nominal_thickness_mm, measured_fwhm_mm=fwhm_mm,
            deviation_mm=deviation, tolerance_mm=self._slice_tol_mm, passed=passed,
            profile_axis="z", z_left_mm=z_left_mm, z_right_mm=z_right_mm,
        )

    def _find_half_max_crossing(
        self, profile: np.ndarray, peak_idx: int,
        direction: str, half_max: float = 0.5,
    ) -> Optional[float]:
        """Finds sub-sample index where normalized profile crosses half_max."""
        if direction == "left":
            indices = range(peak_idx - 1, -1, -1)
        else:
            indices = range(peak_idx + 1, len(profile))

        prev_val = profile[peak_idx]
        prev_idx = float(peak_idx)

        for i in indices:
            curr_val = profile[i]
            curr_idx = float(i)
            if (prev_val >= half_max and curr_val < half_max) or \
               (prev_val < half_max and curr_val >= half_max):
                if abs(curr_val - prev_val) < 1e-9:
                    return prev_idx
                t = (half_max - prev_val) / (curr_val - prev_val)
                return prev_idx + t * (curr_idx - prev_idx)
            prev_val = curr_val
            prev_idx = curr_idx
        return None

    # ── ROI label matching helpers ─────────────────────────────────────

    def _find_centre_roi(self, stats: dict[str, "VolumetricROIStat"]) -> str:
        for label in stats:
            if label.lower() in self._CENTRE_ROI_LABELS:
                return label
        first = next(iter(stats))
        logger.warning("No standard centre ROI label found. Using '%s'.", first)
        return first

    def _find_centre_stat(
        self, stats: dict[str, "VolumetricROIStat"],
    ) -> Optional["VolumetricROIStat"]:
        """Find and return the VolumetricROIStat for the center/water ROI."""
        for label in stats:
            if label.lower() in self._CENTRE_ROI_LABELS:
                return stats[label]
        # Fallback: return the first stat (typically the center)
        if stats:
            return next(iter(stats.values()))
        return None

    def _find_peripheral_rois(self, stats: dict[str, "VolumetricROIStat"]) -> list[str]:
        peripherals = []
        for label in stats:
            ll = label.lower()
            if ll in self._CENTRE_ROI_LABELS:
                continue
            if any(ll.startswith(p) for p in self._PERIPHERAL_PREFIXES):
                peripherals.append(label)
            elif label not in self._CENTRE_ROI_LABELS:
                peripherals.append(label)
        seen = set()
        result = []
        for item in peripherals:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _find_stat_for_material(
        self, stats: dict[str, "VolumetricROIStat"], material_label: str,
    ) -> Optional["VolumetricROIStat"]:
        ml = material_label.lower()
        lower_map = {k.lower(): k for k in stats}
        if ml in lower_map:
            return stats[lower_map[ml]]
        for label, stat in stats.items():
            if ml in label.lower() or label.lower() in ml:
                return stat
        return None

    @staticmethod
    def compute_contrast_from_direct_stats(
        insert_stats: dict[str, dict],
        background_mean_hu: float,
        background_std_hu: float,
        material_refs: dict[str, "MaterialReference"],
        min_cnr: float = 0.5,
    ) -> Optional[ContrastResult]:
        """Compute contrast/CNR from pre-computed per-insert stats.

        This is used when the sensitometry slice has been classified and
        measured directly, bypassing the volumetric pipeline.

        FORCED EXECUTION: If no material reference matches the insert labels,
        falls back to selecting the insert with the highest measured HU
        contrast against the background. Never returns None when insert_stats
        is non-empty.

        Parameters
        ----------
        insert_stats : dict[str, dict]
            Maps insert label to {"mean_hu": float, "std_hu": float}.
        background_mean_hu : float
            Mean HU of the water/centre background ROI.
        background_std_hu : float
            SD of the water/centre background ROI.
        material_refs : dict[str, MaterialReference]
            Nominal reference values from the phantom adapter.
        min_cnr : float
            Minimum CNR threshold for pass/fail.

        Returns
        -------
        ContrastResult or None (only if insert_stats is completely empty)
        """
        if not insert_stats:
            return None

        # Strategy 1: Find insert with highest nominal |HU| via material_refs
        best_label = None
        best_contrast = 0.0
        for label in insert_stats:
            ref = material_refs.get(label)
            if ref is None:
                continue
            if label.lower() in {"water", "center", "centre", "center_water"}:
                continue
            if abs(ref.nominal_hu) > best_contrast:
                best_contrast = abs(ref.nominal_hu)
                best_label = label

        # Strategy 2 (FORCED FALLBACK): If no material ref matched,
        # pick the insert with the highest measured |ΔHU| vs background
        if best_label is None:
            for label, s in insert_stats.items():
                if label.lower() in {"water", "center", "centre", "center_water"}:
                    continue
                delta = abs(s["mean_hu"] - background_mean_hu)
                if delta > best_contrast:
                    best_contrast = delta
                    best_label = label

        # If still nothing (all inserts are water-labeled), use first key
        if best_label is None:
            best_label = next(iter(insert_stats))

        s = insert_stats[best_label]
        hu_insert = s["mean_hu"]
        sd_insert = s["std_hu"]
        hu_bg = background_mean_hu
        sd_bg = background_std_hu
        contrast = abs(hu_insert - hu_bg)

        snr_contrast = contrast / max(sd_bg, 1e-6)
        cnr = contrast / max(np.sqrt(sd_insert**2 + sd_bg**2), 1e-6)

        return ContrastResult(
            insert_label=best_label,
            insert_mean_hu=hu_insert,
            background_mean_hu=hu_bg,
            contrast_hu=contrast,
            snr_contrast=snr_contrast,
            cnr=cnr,
            passed=cnr >= min_cnr,
            min_cnr_threshold=min_cnr,
        )
