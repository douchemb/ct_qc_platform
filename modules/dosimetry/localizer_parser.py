# -*- coding: utf-8 -*-
"""
modules/dosimetry/localizer_parser.py — CT Localizer Radiograph Parser.

Detects, validates, and calibrates CT localizer radiographs (topograms/scouts)
for water-equivalent diameter (D_w) estimation.

Implements the AAPM TG-220 Appendix A LPV calibration pipeline:
  1. Find air reference row (maximum transmission)
  2. Per-column LPV normalization (corrects heel effect / bowtie filter)
  3. Beer-Lambert conversion to water-equivalent path length
  4. Body mask detection via thresholding + morphological ops

Standards References:
    - AAPM TG-220 (2014): Use of Water Equivalent Diameter for SSDE
    - AAPM TG-220 Appendix A: Localizer-based D_w estimation
    - AAPM Report 204 (2011): Size-Specific Dose Estimates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
import scipy.ndimage

from config import DosimetryConfig
from core.dicom_loader import DicomLoader

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = [
    "LocalizerParser",
    "LocalizerData",
    "LocalizerMetadata",
    "LocalizerNotFoundError",
    "LocalizerCalibrationError",
    "LocalizerOrientationError",
]


# ═══════════════════════════════════════════════════════════════════
# Custom Exceptions
# ═══════════════════════════════════════════════════════════════════

class LocalizerNotFoundError(FileNotFoundError):
    """Raised when no CT localizer radiograph is found in a directory."""


class LocalizerCalibrationError(ValueError):
    """Raised when LPV calibration fails due to invalid air reference."""


class LocalizerOrientationError(ValueError):
    """Raised when localizer orientation cannot be determined."""


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LocalizerMetadata:
    """
    Metadata specific to a CT localizer radiograph (topogram).
    The localizer is a projection radiograph — pixel values represent
    integrated X-ray attenuation, NOT Hounsfield Units.
    """
    sop_instance_uid: str
    acquisition_date: str
    kvp: float                              # kVp tube voltage
    pixel_spacing_mm: tuple[float, float]   # (row_spacing_mm, col_spacing_mm)
    rows: int
    cols: int
    rescale_slope: float
    rescale_intercept: float
    image_orientation: str                  # "AP", "LAT", or "UNKNOWN"
    scan_length_mm: float                   # rows × pixel_spacing_mm[0]
    series_description: str
    ctdi_vol: Optional[float]               # from associated axial series, mGy

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return asdict(self)


@dataclass
class LocalizerData:
    """
    Fully calibrated CT localizer radiograph, ready for D_w computation.

    raw_pixel_array: rescaled projection values (NOT HU — localizer pixels
        represent integrated attenuation, not tomographic reconstruction)
    lpv_calibrated: values in range (0, 1] — fraction of open-beam transmission
    water_eq_path_length_cm: Beer-Lambert conversion, units cm
    body_mask: True where patient tissue is present

    Reference: AAPM TG-220 Appendix A
    """
    metadata: LocalizerMetadata
    raw_pixel_array: np.ndarray             # (rows, cols) float32
    air_reference_row: int
    lpv_calibrated: np.ndarray              # (rows, cols) float32, range (0, 1]
    water_eq_path_length_cm: np.ndarray     # (rows, cols) float32, >= 0
    body_mask: np.ndarray                   # (rows, cols) bool

    def to_dict(self) -> dict:
        """Serialize to dictionary — all arrays converted to lists."""
        return {
            "metadata": self.metadata.to_dict(),
            "air_reference_row": self.air_reference_row,
            "raw_pixel_array": self.raw_pixel_array.tolist(),
            "lpv_calibrated": self.lpv_calibrated.tolist(),
            "water_eq_path_length_cm": self.water_eq_path_length_cm.tolist(),
            "body_mask": self.body_mask.tolist(),
        }


# ═══════════════════════════════════════════════════════════════════
# Localizer Parser Class
# ═══════════════════════════════════════════════════════════════════

class LocalizerParser:
    """
    Detects, validates, and calibrates CT localizer radiographs.
    Implements AAPM TG-220 Appendix A LPV calibration pipeline.
    """

    # μ_water at 70 keV effective energy — AAPM TG-220 standard value
    # for 120 kVp CT beam (effective energy ≈ 70 keV)
    MU_WATER_CM = 0.1928  # cm⁻¹

    def __init__(self, dicom_loader: DicomLoader, config: DosimetryConfig) -> None:
        self._dicom_loader = dicom_loader
        self._config = config

    def find_localizer_in_directory(self, dir_path: Path) -> list[Path]:
        """Scan directory for CT localizer radiographs.

        Parameters
        ----------
        dir_path : Path
            Directory containing DICOM files.

        Returns
        -------
        list[Path]
            Paths to localizer files, sorted by AcquisitionTime.

        Raises
        ------
        LocalizerNotFoundError
            If no localizer radiographs are found.
        """
        dir_path = Path(dir_path).resolve()
        if not dir_path.is_dir():
            raise LocalizerNotFoundError(
                "Directory does not exist: %s" % dir_path
            )

        localizer_paths: list[Path] = []
        for fpath in sorted(dir_path.glob("*.dcm")):
            try:
                ds = pydicom.dcmread(str(fpath), force=False, stop_before_pixels=True)
            except Exception:
                continue

            if self._dicom_loader.is_localizer(ds):
                localizer_paths.append(fpath)

        if not localizer_paths:
            raise LocalizerNotFoundError(
                "No CT localizer radiographs found in directory: %s" % dir_path
            )

        # Sort by AcquisitionTime if available
        def _sort_key(p: Path) -> str:
            try:
                ds = pydicom.dcmread(str(p), force=False, stop_before_pixels=True)
                return str(getattr(ds, "AcquisitionTime", "000000"))
            except Exception:
                return "000000"

        localizer_paths.sort(key=_sort_key)
        logger.info("Found %d localizer(s) in %s", len(localizer_paths), dir_path)
        return localizer_paths

    def parse(self, dcm_path: Path) -> LocalizerData:
        """Full calibration pipeline for a CT localizer radiograph.

        Steps:
        1. Load DICOM (require_ct=False — localizers have different SOP)
        2. Extract metadata
        3. Apply rescale: raw_array = slope × pixel_array + intercept
        4. Find air reference row
        5. Calibrate LPV (per-column normalization)
        6. Compute water-equivalent path length (Beer-Lambert)
        7. Detect body mask (threshold + morphology)

        Parameters
        ----------
        dcm_path : Path
            Path to the localizer DICOM file.

        Returns
        -------
        LocalizerData
            Fully calibrated localizer data.
        """
        dcm_path = Path(dcm_path).resolve()

        # Step 1: Load DICOM — localizers are not standard CT images
        ds = self._dicom_loader.load_file(dcm_path, require_ct=False)

        # Step 2: Extract metadata
        metadata = self._extract_localizer_metadata(ds)

        # Step 3: Apply rescale transform
        # Localizer pixel values = slope × raw_stored_value + intercept
        raw_array = (
            metadata.rescale_slope
            * ds.pixel_array.astype(np.float32)
            + metadata.rescale_intercept
        )

        # Step 4: Find air reference row
        air_reference_row = self._find_air_reference_row(raw_array)

        # Step 5: Calibrate LPV — per-column normalization
        lpv_calibrated = self._calibrate_lpv(raw_array, air_reference_row)

        # Step 6: Compute water-equivalent path length (Beer-Lambert)
        water_eq_path_length_cm = self._compute_water_eq_path_length(lpv_calibrated)

        # Step 7: Detect body mask
        body_mask = self._detect_body_mask(water_eq_path_length_cm)

        # Log summary at INFO
        body_path_lengths = water_eq_path_length_cm[body_mask]
        mean_path_in_body = float(np.mean(body_path_lengths)) if body_path_lengths.size > 0 else 0.0
        body_coverage_pct = 100.0 * body_mask.sum() / body_mask.size

        logger.info(
            "Localizer parsed — dims: (%d, %d), orientation: %s, "
            "air_ref_row: %d, mean_path_in_body: %.2f cm, "
            "body_coverage: %.1f%%",
            metadata.rows, metadata.cols, metadata.image_orientation,
            air_reference_row, mean_path_in_body, body_coverage_pct,
        )

        return LocalizerData(
            metadata=metadata,
            raw_pixel_array=raw_array,
            air_reference_row=air_reference_row,
            lpv_calibrated=lpv_calibrated,
            water_eq_path_length_cm=water_eq_path_length_cm,
            body_mask=body_mask,
        )

    # ─────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────

    def _extract_localizer_metadata(
        self, ds: pydicom.Dataset
    ) -> LocalizerMetadata:
        """Extract all LocalizerMetadata fields from a DICOM dataset.

        Uses getattr with safe defaults for optional tags.
        """
        sop_instance_uid = str(getattr(ds, "SOPInstanceUID", ""))
        acquisition_date = str(getattr(ds, "AcquisitionDate", "19000101"))
        kvp = float(getattr(ds, "KVP", 120.0))
        series_description = str(getattr(ds, "SeriesDescription", ""))

        # Pixel spacing as tuple (row_spacing, col_spacing) in mm
        raw_spacing = getattr(ds, "PixelSpacing", None)
        if raw_spacing is not None:
            pixel_spacing_mm = (float(raw_spacing[0]), float(raw_spacing[1]))
        else:
            pixel_spacing_mm = (1.0, 1.0)

        rows = int(getattr(ds, "Rows", 0))
        cols = int(getattr(ds, "Columns", 0))

        # Rescale slope/intercept — exist on localizers, represent linear
        # transform of raw projection values
        rescale_slope = float(getattr(ds, "RescaleSlope", 1.0))
        rescale_intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        # Image orientation
        image_orientation = self._detect_image_orientation(ds)

        # Scan length = rows × row spacing
        scan_length_mm = float(rows) * pixel_spacing_mm[0]

        # CTDIvol — optional, from associated axial series
        ctdi_vol = getattr(ds, "CTDIvol", None)
        if ctdi_vol is not None:
            ctdi_vol = float(ctdi_vol)

        return LocalizerMetadata(
            sop_instance_uid=sop_instance_uid,
            acquisition_date=acquisition_date,
            kvp=kvp,
            pixel_spacing_mm=pixel_spacing_mm,
            rows=rows,
            cols=cols,
            rescale_slope=rescale_slope,
            rescale_intercept=rescale_intercept,
            image_orientation=image_orientation,
            scan_length_mm=scan_length_mm,
            series_description=series_description,
            ctdi_vol=ctdi_vol,
        )

    def _detect_image_orientation(
        self, ds: pydicom.Dataset
    ) -> str:
        """Determine localizer orientation from DICOM tags.

        Algorithm (DICOM tag ImageOrientationPatient [0020,0037]):
        - 6-element list: [F1,F2,F3,F4,F5,F6]
        - Row direction vector: [F1,F2,F3]
        - For AP: row direction primarily along left-right (abs(F1) dominant)
        - For LAT: row direction primarily along anterior-posterior (abs(F2) dominant)

        Falls back to SeriesDescription keyword matching if tag absent.
        """
        iop = getattr(ds, "ImageOrientationPatient", None)

        if iop is not None and len(iop) >= 3:
            row_dir = [float(iop[0]), float(iop[1]), float(iop[2])]
            if abs(row_dir[0]) > abs(row_dir[1]):
                return "AP"
            else:
                return "LAT"

        # Fallback: check SeriesDescription for orientation keywords
        series_desc = str(getattr(ds, "SeriesDescription", "")).upper()
        for keyword in ["AP", "PA"]:
            if keyword in series_desc:
                return "AP"
        for keyword in ["LAT", "LATERAL"]:
            if keyword in series_desc:
                return "LAT"

        # Default to AP with warning
        logger.warning(
            "Cannot determine localizer orientation from DICOM tags — "
            "defaulting to AP"
        )
        return "AP"

    def _find_air_reference_row(
        self, raw_array: np.ndarray
    ) -> int:
        """Find the air reference row (maximum mean pixel value).

        The air reference row has maximum X-ray transmission (no patient
        attenuation). Excludes first and last 5% of rows to avoid
        collimator edges and table artifacts.

        Parameters
        ----------
        raw_array : np.ndarray
            Rescaled pixel array (rows, cols), float32.

        Returns
        -------
        int
            Row index with maximum mean pixel value.
        """
        n_rows = raw_array.shape[0]
        margin = max(1, int(0.05 * n_rows))  # 5% margin

        # Compute mean pixel value per row
        row_means = np.mean(raw_array, axis=1)

        # Restrict to valid range (exclude collimator edges)
        valid_start = margin
        valid_end = n_rows - margin

        # Find row with maximum mean in valid range
        valid_means = row_means[valid_start:valid_end]
        air_ref_row = valid_start + int(np.argmax(valid_means))

        logger.debug("Air reference row identified at index %d (mean=%.1f)",
                      air_ref_row, row_means[air_ref_row])
        return air_ref_row

    def _calibrate_lpv(
        self,
        raw_array: np.ndarray,
        air_reference_row: int,
    ) -> np.ndarray:
        """Normalize pixel values to calibrated LPV (Localizer Pixel Value).

        LPV_calibrated(i,j) = raw_pixel_array(i,j) / air_ref_value(j)

        Per-column normalization corrects for heel effect and bowtie filter.

        Reference: AAPM TG-220 Appendix A

        Parameters
        ----------
        raw_array : np.ndarray
            Rescaled pixel array (rows, cols), float32.
        air_reference_row : int
            Row index of the air reference.

        Returns
        -------
        np.ndarray
            Calibrated LPV, clipped to [0.001, 1.0].

        Raises
        ------
        LocalizerCalibrationError
            If any air reference column value <= 0.
        """
        # Get per-column air reference values
        air_ref_values = raw_array[air_reference_row, :].astype(np.float64)

        # Check for invalid air reference values
        invalid_cols = np.where(air_ref_values <= 0)[0]
        if len(invalid_cols) > 0:
            raise LocalizerCalibrationError(
                "Air reference value <= 0 at column(s): %s. "
                "LPV calibration cannot proceed."
                % str(invalid_cols.tolist())
            )

        # Per-column normalization: LPV_cal(i,j) = raw(i,j) / air_ref(j)
        # AAPM TG-220 Appendix A — per-column normalization corrects for
        # heel effect and bowtie filter variation
        lpv_calibrated = raw_array.astype(np.float64) / air_ref_values[np.newaxis, :]
        lpv_calibrated = lpv_calibrated.astype(np.float32)

        # Count pixels before clipping for diagnostics
        n_above = int(np.sum(lpv_calibrated > 1.0))
        n_below = int(np.sum(lpv_calibrated < 0.001))
        total = lpv_calibrated.size
        frac_clipped = (n_above + n_below) / total

        # Clip to [0.001, 1.0]
        # Values > 1.0 are physically impossible (more transmission than air)
        # Values at 0 cause log(0) = -inf in Beer-Lambert step
        lpv_calibrated = np.clip(lpv_calibrated, 0.001, 1.0)

        logger.debug(
            "LPV calibration: %.2f%% of pixels clipped "
            "(%d above 1.0, %d below 0.001)",
            frac_clipped * 100.0, n_above, n_below,
        )

        return lpv_calibrated

    def _compute_water_eq_path_length(
        self,
        lpv_calibrated: np.ndarray,
    ) -> np.ndarray:
        """Apply Beer-Lambert law to convert LPV to water-equivalent path length.

        t_w(i,j) = -ln(LPV_calibrated(i,j)) / μ_water

        Reference: AAPM TG-220 Appendix A — Beer-Lambert attenuation law

        Parameters
        ----------
        lpv_calibrated : np.ndarray
            Calibrated LPV values in range (0, 1].

        Returns
        -------
        np.ndarray
            Water-equivalent path length in cm, clamped to [0.0, inf).
        """
        # AAPM TG-220 Appendix A — Beer-Lambert attenuation law:
        # t_w(i,j) = -ln(LPV_calibrated(i,j)) / μ_water
        # μ_water = 0.1928 cm⁻¹ at 70 keV effective energy
        water_eq_path = -np.log(lpv_calibrated.astype(np.float64)) / self.MU_WATER_CM
        water_eq_path = water_eq_path.astype(np.float32)

        # Clamp to [0.0, None] — negative values arise from pixels slightly
        # brighter than air reference due to noise; physically impossible
        water_eq_path = np.clip(water_eq_path, 0.0, None)

        mean_path = float(np.mean(water_eq_path))
        logger.debug("Mean water-equivalent path length: %.2f cm", mean_path)

        return water_eq_path

    def _detect_body_mask(
        self,
        water_eq_path_length: np.ndarray,
    ) -> np.ndarray:
        """Detect patient body region in the localizer image.

        Strategy:
        1. Threshold at 0.5 cm water-equivalent path length
        2. Morphological closing (fills small holes)
        3. Fill holes (closes body contour — lungs inside boundary)

        Parameters
        ----------
        water_eq_path_length : np.ndarray
            Water-equivalent path length in cm.

        Returns
        -------
        np.ndarray
            Boolean mask, True where patient tissue is present.
        """
        # Step 1: Threshold at 0.5 cm
        threshold_mask = water_eq_path_length > 0.5

        # Step 2: Morphological closing — fills small holes
        structure = np.ones((5, 5))
        closed = scipy.ndimage.binary_closing(threshold_mask, structure=structure)

        # Step 3: Fill holes — closes body contour so air-filled lungs
        # are inside the body boundary
        filled = scipy.ndimage.binary_fill_holes(closed)

        return filled.astype(bool)
