# -*- coding: utf-8 -*-
"""
modules/image_qc/roi_stats.py — ROI Statistical Analyzer.

Core statistical engine for CT phantom image quality assessment.
Computes per-ROI statistics and volumetric multi-slice aggregation.

Standards:
    - AAPM TG-66 Section 5.1: Noise tolerance (std <= 5 HU)
    - AAPM TG-66 Section 5.2: HU linearity tolerance (±4 HU)
    - AAPM TG-233: CatPhan phantom QC protocol

Physics Notes:
    - Bessel's correction (ddof=1) is used for sample statistics
    - Skewness and kurtosis are computed from central moments
    - SNR = |mean| / std, returns NaN if std < 1e-9
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from pydicom.dataset import Dataset

from config import CONFIG, ImageQCConfig
from core.dicom_loader import DicomLoader, DicomMetadata

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = [
    "ROIDescriptor", "ROIStatistics", "SliceAnalysisResult",
    "VolumetricROIStat", "VolumetricQCResult",
    "PhantomROIAnalyzer", "ROIBoundsError", "compute_batch_statistics",
]


# ═══════════════════════════════════════════════════════════════════
# Custom Exceptions
# ═══════════════════════════════════════════════════════════════════

class ROIBoundsError(ValueError):
    """Raised when a requested ROI extends outside the image boundaries."""


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ROIDescriptor:
    """Defines a rectangular Region of Interest on a 2D image."""
    label: str
    row_start: int
    col_start: int
    height_px: int
    width_px: int

    @property
    def row_end(self) -> int:
        return self.row_start + self.height_px

    @property
    def col_end(self) -> int:
        return self.col_start + self.width_px

    @property
    def area_px(self) -> int:
        return self.height_px * self.width_px

    def validate(self) -> None:
        """Raises ValueError if any dimension is zero or negative."""
        if self.height_px <= 0:
            raise ValueError("ROI '%s' has invalid height: %d (must be > 0)" % (self.label, self.height_px))
        if self.width_px <= 0:
            raise ValueError("ROI '%s' has invalid width: %d (must be > 0)" % (self.label, self.width_px))


@dataclass
class ROIStatistics:
    """
    Pure statistical results for one ROI on one DICOM slice.
    All values in Hounsfield Units (HU) — computed AFTER rescale transform.
    ddof=1 (Bessel's correction) used for std and variance — sample statistics.

    Physics reference: AAPM TG-66 Section 5.1
    Noise tolerance: std_hu <= 5.0 HU for water phantom
    """
    roi_label: str
    slice_file: str
    acquisition_date: str
    mean_hu: float
    std_hu: float
    variance_hu: float      # std_hu² — direct input to NPS calculator
    min_hu: float
    max_hu: float
    snr: float              # |mean_hu| / std_hu, NaN if std < 1e-9
    skewness: float         # m3 / m2^(3/2) — detects ring artifact asymmetry
    kurtosis: float         # excess kurtosis m4/m2² - 3
    n_pixels: int
    roi_row_start: int
    roi_col_start: int
    roi_height_px: int
    roi_width_px: int

    def to_dict(self) -> dict:
        return asdict(self)

    def passes_tg66_noise_tolerance(self, tolerance_hu: float = None) -> bool:
        """Returns True if std_hu <= tolerance. Reference: AAPM TG-66 Section 5.1."""
        if tolerance_hu is None:
            tolerance_hu = CONFIG.image_qc.noise_tolerance_hu
        return self.std_hu <= tolerance_hu


@dataclass
class SliceAnalysisResult:
    """Container for all ROI results on one DICOM slice plus its metadata."""
    metadata: DicomMetadata
    roi_results: list[ROIStatistics]
    analysis_timestamp: str     # ISO 8601
    source_file: str

    def to_dict(self) -> dict:
        d = {
            "metadata": self.metadata.to_dict(),
            "roi_results": [r.to_dict() for r in self.roi_results],
            "analysis_timestamp": self.analysis_timestamp,
            "source_file": self.source_file,
        }
        return d

    def get_roi_stat(self, label: str) -> ROIStatistics:
        """Returns stat for named ROI. Raises KeyError if absent."""
        for stat in self.roi_results:
            if stat.roi_label == label:
                return stat
        raise KeyError("ROI label '%s' not found in slice results" % label)


@dataclass
class VolumetricROIStat:
    """
    Volumetric (multi-slice averaged) statistics for one named ROI.

    *_mean fields: arithmetic mean across the selected slice range.
    *_std fields: standard deviation across slices — quantifies slice-to-slice
    consistency, NOT within-slice pixel noise.

    std_hu_mean is the primary noise metric passed to the NPS calculator
    and the predictive maintenance archive.
    """
    roi_label: str
    n_slices: int
    mean_hu_mean: float
    mean_hu_std: float
    std_hu_mean: float          # PRIMARY NOISE METRIC
    std_hu_std: float
    variance_hu_mean: float     # NPS input
    variance_hu_std: float
    min_hu_overall: float
    max_hu_overall: float
    snr_mean: float
    passes_tg66: bool           # std_hu_mean <= CONFIG.image_qc.noise_tolerance_hu

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VolumetricQCResult:
    """
    Averaged QC metrics across a selected range of axial slices.

    This is the PRIMARY OUTPUT UNIT consumed by all downstream modules:
      NPS calculator:    consumes hu_arrays (list of float32 HU images)
      MTF calculator:    consumes hu_arrays[middle_slice]
      ED calibration:    consumes volumetric_stats per material ROI
      Dosimetry:         consumes pixel_spacing_mm and slice metadata
      Predictive:        consumes volumetric_stats["center_water"].std_hu_mean

    Physics rationale: averaging over N slices reduces the standard error
    of the noise estimate by sqrt(N), providing a more stable QC reference.
    """
    series_description: str
    acquisition_date: str
    start_slice: int                    # 1-based, as passed by user
    end_slice: int
    n_slices_selected: int
    n_slices_processed: int
    pixel_spacing_mm: tuple[float, float]
    slice_thickness_mm: float
    slice_results: list[SliceAnalysisResult]
    volumetric_stats: dict[str, VolumetricROIStat]
    hu_arrays: list[np.ndarray]         # NOT serialized to JSON

    def to_dict(self) -> dict:
        """Serializes to dict for JSON archival.
        hu_arrays is EXCLUDED — too large for JSON and consumed directly by NPSCalculator.
        """
        d = {
            "series_description": self.series_description,
            "acquisition_date": self.acquisition_date,
            "start_slice": self.start_slice,
            "end_slice": self.end_slice,
            "n_slices_selected": self.n_slices_selected,
            "n_slices_processed": self.n_slices_processed,
            "pixel_spacing_mm": list(self.pixel_spacing_mm),
            "slice_thickness_mm": self.slice_thickness_mm,
            "slice_results": [r.to_dict() for r in self.slice_results],
            "volumetric_stats": {k: v.to_dict() for k, v in self.volumetric_stats.items()},
            # hu_arrays excluded from serialization
        }
        return d

    def get_volumetric_stat(self, roi_label: str) -> VolumetricROIStat:
        """Returns VolumetricROIStat for named ROI. Raises KeyError if absent."""
        if roi_label not in self.volumetric_stats:
            raise KeyError("ROI label '%s' not found in volumetric stats" % roi_label)
        return self.volumetric_stats[roi_label]

    def passes_tg66_volumetric(self, roi_label: str = "center_water", tolerance_hu: float = None) -> bool:
        """
        Returns True if volumetric mean noise SD passes TG-66 tolerance.
        Uses std_hu_mean (averaged across slices), not any single slice.
        Reference: AAPM TG-66 Section 5.1 — noise tolerance <= 5.0 HU.
        """
        if tolerance_hu is None:
            tolerance_hu = CONFIG.image_qc.noise_tolerance_hu
        stat = self.get_volumetric_stat(roi_label)
        return stat.std_hu_mean <= tolerance_hu


# ═══════════════════════════════════════════════════════════════════
# Main Analyzer Class
# ═══════════════════════════════════════════════════════════════════

class PhantomROIAnalyzer:
    """Core statistical engine for CT phantom image quality analysis.

    Takes a DicomLoader instance as a dependency — not a directory path.
    This decouples loading from analysis and makes the class fully testable.
    """

    def __init__(self, dicom_loader: DicomLoader, config: ImageQCConfig) -> None:
        self._dicom_loader = dicom_loader
        self._config = config

    def analyze_dataset(self, ds: Dataset, rois: list[ROIDescriptor]) -> SliceAnalysisResult:
        """Analyzes a single already-loaded pydicom.Dataset."""
        hu_array = self._dicom_loader.to_hu_array(ds)
        metadata = self._dicom_loader.extract_metadata(ds)
        filename = str(getattr(ds, "filename", "in_memory"))
        if hasattr(Path(filename), "name"):
            filename = Path(filename).name

        roi_results = []
        for roi in rois:
            roi.validate()
            self._validate_roi_bounds(roi, hu_array.shape, filename)
            stats = self._compute_roi_statistics(hu_array, roi, metadata, filename)
            roi_results.append(stats)

        return SliceAnalysisResult(
            metadata=metadata,
            roi_results=roi_results,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            source_file=filename,
        )

    def analyze_directory(self, dir_path: Path, rois: list[ROIDescriptor]) -> list[SliceAnalysisResult]:
        """Loads all CT slices from directory. Wraps analyze_volume internally."""
        datasets = self._dicom_loader.load_directory(dir_path)
        if not datasets:
            logger.warning("No CT datasets found in: %s", dir_path)
            return []
        results = []
        for ds in datasets:
            result = self.analyze_dataset(ds, rois)
            results.append(result)
        results.sort(key=lambda r: r.metadata.instance_number)
        logger.info("Analyzed %d slices with %d ROIs each from: %s", len(results), len(rois), dir_path)
        return results

    def analyze_volume(
        self, dir_path: Path, rois: list[ROIDescriptor],
        start_slice: int, end_slice: int, sort_by: str = "InstanceNumber",
    ) -> VolumetricQCResult:
        """The primary method for downstream consumption.

        Steps:
        1. Load slice range
        2. Analyze each dataset (catch per-slice exceptions, log WARNING, continue)
        3. Collect hu_arrays
        4. Compute VolumetricROIStat for each ROI label
        5. Return VolumetricQCResult
        """
        datasets = self._dicom_loader.load_slice_range(dir_path, start_slice, end_slice, sort_by)

        slice_results: list[SliceAnalysisResult] = []
        hu_arrays: list[np.ndarray] = []
        successful_datasets: list[Dataset] = []

        for ds in datasets:
            try:
                result = self.analyze_dataset(ds, rois)
                slice_results.append(result)
                hu_array = self._dicom_loader.to_hu_array(ds)
                hu_arrays.append(hu_array)
                successful_datasets.append(ds)
            except Exception as exc:
                logger.warning("Skipping slice due to error: %s", exc)
                continue

        n_processed = len(slice_results)

        # Compute volumetric stats for each ROI label
        volumetric_stats: dict[str, VolumetricROIStat] = {}
        if slice_results:
            roi_labels = [roi.label for roi in rois]
            for label in roi_labels:
                means, stds, variances, mins, maxs, snrs = [], [], [], [], [], []
                for sr in slice_results:
                    try:
                        stat = sr.get_roi_stat(label)
                        means.append(stat.mean_hu)
                        stds.append(stat.std_hu)
                        variances.append(stat.variance_hu)
                        mins.append(stat.min_hu)
                        maxs.append(stat.max_hu)
                        snrs.append(stat.snr)
                    except KeyError:
                        continue

                if means:
                    means_arr = np.array(means)
                    stds_arr = np.array(stds)
                    vars_arr = np.array(variances)
                    snrs_arr = np.array([s for s in snrs if not np.isnan(s)])
                    std_hu_mean_val = float(np.mean(stds_arr))

                    volumetric_stats[label] = VolumetricROIStat(
                        roi_label=label,
                        n_slices=len(means),
                        mean_hu_mean=float(np.mean(means_arr)),
                        mean_hu_std=float(np.std(means_arr, ddof=1)) if len(means) > 1 else 0.0,  # ddof=1: Bessel's correction — AAPM TG-66
                        std_hu_mean=std_hu_mean_val,
                        std_hu_std=float(np.std(stds_arr, ddof=1)) if len(stds) > 1 else 0.0,  # ddof=1: Bessel's correction — AAPM TG-66
                        variance_hu_mean=float(np.mean(vars_arr)),
                        variance_hu_std=float(np.std(vars_arr, ddof=1)) if len(variances) > 1 else 0.0,  # ddof=1: Bessel's correction — AAPM TG-66
                        min_hu_overall=float(np.min(mins)),
                        max_hu_overall=float(np.max(maxs)),
                        snr_mean=float(np.mean(snrs_arr)) if len(snrs_arr) > 0 else float("nan"),
                        passes_tg66=std_hu_mean_val <= self._config.noise_tolerance_hu,  # AAPM TG-66 Section 5.1
                    )

        # Extract pixel_spacing and slice_thickness from first successful slice
        if slice_results:
            first_meta = slice_results[0].metadata
            pixel_spacing_mm = first_meta.pixel_spacing_mm
            slice_thickness_mm = first_meta.slice_thickness_mm
            series_description = first_meta.series_description
            acquisition_date = first_meta.acquisition_date
        else:
            pixel_spacing_mm = (1.0, 1.0)
            slice_thickness_mm = 0.0
            series_description = ""
            acquisition_date = ""

        # Log summary
        center_stat = volumetric_stats.get("center_water")
        if center_stat:
            logger.info(
                "Volumetric analysis complete: %d/%d slices processed, %d ROIs. "
                "Center water SD: %.3f ± %.3f HU",
                n_processed, end_slice - start_slice + 1, len(volumetric_stats),
                center_stat.std_hu_mean, center_stat.std_hu_std,
            )
        else:
            logger.info("Volumetric analysis complete: %d/%d slices processed, %d ROIs",
                        n_processed, end_slice - start_slice + 1, len(volumetric_stats))

        return VolumetricQCResult(
            series_description=series_description,
            acquisition_date=acquisition_date,
            start_slice=start_slice,
            end_slice=end_slice,
            n_slices_selected=end_slice - start_slice + 1,
            n_slices_processed=n_processed,
            pixel_spacing_mm=pixel_spacing_mm,
            slice_thickness_mm=slice_thickness_mm,
            slice_results=slice_results,
            volumetric_stats=volumetric_stats,
            hu_arrays=hu_arrays,
        )

    def _validate_roi_bounds(self, roi: ROIDescriptor, image_shape: tuple, filename: str) -> None:
        """Raise ROIBoundsError if ROI falls outside image bounds."""
        n_rows, n_cols = image_shape[:2]
        if roi.row_start < 0 or roi.col_start < 0 or roi.row_end > n_rows or roi.col_end > n_cols:
            raise ROIBoundsError(
                "ROI '%s' extends beyond image boundaries. "
                "ROI bounds: rows [%d, %d), cols [%d, %d). "
                "Image shape: (%d, %d). File: %s"
                % (roi.label, roi.row_start, roi.row_end, roi.col_start, roi.col_end, n_rows, n_cols, filename)
            )

    def _compute_roi_statistics(
        self, hu_array: np.ndarray, roi: ROIDescriptor,
        metadata: DicomMetadata, filename: str,
    ) -> ROIStatistics:
        """Extract ROI subarray and compute all statistical metrics."""
        roi_pixels = hu_array[roi.row_start:roi.row_end, roi.col_start:roi.col_end].copy().flatten().astype(np.float64)
        n_pixels = roi_pixels.size

        mean_hu = float(np.mean(roi_pixels))
        std_hu = float(np.std(roi_pixels, ddof=1))    # ddof=1: Bessel's correction for unbiased sample variance — AAPM TG-66
        variance_hu = float(np.var(roi_pixels, ddof=1))  # ddof=1: Bessel's correction — AAPM TG-66
        min_hu = float(np.min(roi_pixels))
        max_hu = float(np.max(roi_pixels))
        snr = abs(mean_hu) / std_hu if std_hu > 1e-9 else float("nan")

        # Central moments for skewness and kurtosis — computed manually, not scipy
        residuals = roi_pixels - mean_hu
        m2 = float(np.mean(residuals ** 2))  # 2nd central moment
        m3 = float(np.mean(residuals ** 3))  # 3rd central moment
        m4 = float(np.mean(residuals ** 4))  # 4th central moment

        # Fisher skewness: m3 / m2^(3/2) — detects ring artifact asymmetry
        skewness = m3 / (m2 ** 1.5) if m2 > 1e-12 else 0.0
        # Excess kurtosis: m4 / m2^2 - 3 (normal distribution = 0)
        kurtosis = (m4 / (m2 ** 2)) - 3.0 if m2 > 1e-12 else 0.0

        logger.debug("ROI '%s' stats — mean: %.2f HU, std: %.2f HU, skew: %.3f, kurt: %.3f, n: %d",
                      roi.label, mean_hu, std_hu, skewness, kurtosis, n_pixels)

        return ROIStatistics(
            roi_label=roi.label, slice_file=filename, acquisition_date=metadata.acquisition_date,
            mean_hu=mean_hu, std_hu=std_hu, variance_hu=variance_hu,
            min_hu=min_hu, max_hu=max_hu, snr=snr,
            skewness=skewness, kurtosis=kurtosis, n_pixels=n_pixels,
            roi_row_start=roi.row_start, roi_col_start=roi.col_start,
            roi_height_px=roi.height_px, roi_width_px=roi.width_px,
        )


# ═══════════════════════════════════════════════════════════════════
# Batch Statistics (Bridge Function)
# ═══════════════════════════════════════════════════════════════════

def compute_batch_statistics(results: list[SliceAnalysisResult], roi_label: str) -> dict:
    """Compute cross-slice statistics for a specific ROI.

    Bridge function for NPS (Phase 2) and predictive maintenance (Phase 4).
    """
    dates, mean_series, std_series, var_series = [], [], [], []

    for result in results:
        try:
            stat = result.get_roi_stat(roi_label)
            dates.append(result.metadata.acquisition_date)
            mean_series.append(stat.mean_hu)
            std_series.append(stat.std_hu)
            var_series.append(stat.variance_hu)
        except KeyError:
            continue

    if not mean_series:
        raise ValueError("ROI label '%s' not found in any of the %d results." % (roi_label, len(results)))

    mean_arr = np.array(mean_series)
    std_arr = np.array(std_series)
    var_arr = np.array(var_series)

    return {
        "roi_label": roi_label,
        "dates": dates,
        "mean_hu_series": mean_arr,
        "std_hu_series": std_arr,
        "variance_series": var_arr,
        "grand_mean": float(np.mean(mean_arr)),
        "grand_std": float(np.mean(std_arr)),
        "grand_variance": float(np.mean(var_arr)),
        "n_slices": len(mean_series),
    }
