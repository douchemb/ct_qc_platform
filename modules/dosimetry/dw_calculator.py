# -*- coding: utf-8 -*-
"""
modules/dosimetry/dw_calculator.py — Water-Equivalent Diameter Calculator.

Computes the water-equivalent diameter D_w using two methods:
  1. Axial CT slices — primary method, most accurate (TG-220 Section 3)
  2. CT localizer radiograph — secondary method (TG-220 Appendix A)

D_w is the sole input to the SSDE f-factor lookup (AAPM Report 204).

Physics:
    D_w (cm) = 2 × sqrt(A_w / π)                    — AAPM TG-220 Eq. 2
    A_w (cm²) = Σ[(HU/1000+1) × pixel_area_cm²]     — AAPM TG-220 Eq. 1

Standards References:
    - AAPM TG-220 (2014): Water Equivalent Diameter
    - AAPM TG-220 Eq. 1: Water-equivalent area
    - AAPM TG-220 Eq. 2: Water-equivalent diameter
    - AAPM Report 204 (2011): SSDE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import scipy.ndimage

from config import DosimetryConfig
from core.dicom_loader import DicomMetadata

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = [
    "DwCalculator",
    "DwSliceResult",
    "DwSeriesResult",
    "DwLocalizerResult",
    "BodySegmentationError",
    "InsufficientSlicesError",
]


# ═══════════════════════════════════════════════════════════════════
# Custom Exceptions
# ═══════════════════════════════════════════════════════════════════

class BodySegmentationError(RuntimeError):
    """Raised when body segmentation fails to find a valid patient contour."""


class InsufficientSlicesError(ValueError):
    """Raised when too few axial slices are available for series-level D_w."""


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DwSliceResult:
    """
    D_w computed from one axial CT slice using AAPM TG-220 Equation 2.

    water_eq_area_cm2: A_w = Σ[(HU/1000+1) × pixel_area_cm²] over body mask
    dw_cm: D_w = 2×sqrt(A_w/π) — primary dosimetry input
    segmentation_method: "largest_contour" (preferred) or "threshold"
    """
    slice_position_mm: float        # z-position in mm
    instance_number: int
    acquisition_date: str
    water_eq_area_cm2: float        # A_w in cm²
    dw_cm: float                    # D_w in cm
    n_body_pixels: int
    n_total_pixels: int
    body_fraction: float            # n_body_pixels / n_total_pixels
    segmentation_method: str        # "largest_contour" or "threshold"

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return asdict(self)


@dataclass
class DwSeriesResult:
    """
    D_w results for a complete CT series (multiple axial slices).
    dw_at_isocenter_cm is the value used for single-point SSDE calculation.
    dw_mean_cm is the value used for series-averaged SSDE calculation.
    """
    series_description: str
    acquisition_date: str
    n_slices: int
    slice_results: list[DwSliceResult]      # sorted by slice_position_mm
    dw_mean_cm: float                       # mean D_w in cm
    dw_min_cm: float                        # min D_w in cm
    dw_max_cm: float                        # max D_w in cm
    dw_std_cm: float                        # std D_w in cm
    dw_at_isocenter_cm: float               # D_w at slice closest to z=0
    dw_from_localizer_cm: Optional[float]   # if localizer available, cm

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "series_description": self.series_description,
            "acquisition_date": self.acquisition_date,
            "n_slices": self.n_slices,
            "slice_results": [s.to_dict() for s in self.slice_results],
            "dw_mean_cm": self.dw_mean_cm,
            "dw_min_cm": self.dw_min_cm,
            "dw_max_cm": self.dw_max_cm,
            "dw_std_cm": self.dw_std_cm,
            "dw_at_isocenter_cm": self.dw_at_isocenter_cm,
            "dw_from_localizer_cm": self.dw_from_localizer_cm,
        }


@dataclass
class DwLocalizerResult:
    """
    D_w estimated per-row from the CT localizer radiograph.
    Less accurate than axial method but available before axial scan.
    Reference: AAPM TG-220 Appendix A.
    """
    acquisition_date: str
    image_orientation: str                      # "AP", "LAT", or "UNKNOWN"
    water_eq_area_per_row_cm2: np.ndarray       # A_w per localizer row, cm²
    dw_per_row_cm: np.ndarray                   # D_w per row, cm
    row_positions_mm: np.ndarray                # axial position of each row, mm
    dw_mean_cm: float                           # mean D_w in cm
    dw_at_center_row_cm: float                  # D_w at center row, cm

    def to_dict(self) -> dict:
        """Serialize to dictionary — all arrays converted to lists."""
        return {
            "acquisition_date": self.acquisition_date,
            "image_orientation": self.image_orientation,
            "water_eq_area_per_row_cm2": self.water_eq_area_per_row_cm2.tolist(),
            "dw_per_row_cm": self.dw_per_row_cm.tolist(),
            "row_positions_mm": self.row_positions_mm.tolist(),
            "dw_mean_cm": self.dw_mean_cm,
            "dw_at_center_row_cm": self.dw_at_center_row_cm,
        }


# ═══════════════════════════════════════════════════════════════════
# D_w Calculator Class
# ═══════════════════════════════════════════════════════════════════

class DwCalculator:
    """
    Computes water-equivalent diameter D_w using two methods:
    1. Axial CT slices — primary method, most accurate (TG-220 Section 3)
    2. CT localizer radiograph — secondary method (TG-220 Appendix A)
    """

    def __init__(self, config: DosimetryConfig) -> None:
        self._config = config

    def compute_from_axial_series(
        self,
        hu_arrays: list[np.ndarray],
        pixel_spacing_mm: tuple[float, float],
        slice_positions_mm: list[float],
        metadata_list: list[DicomMetadata],
    ) -> DwSeriesResult:
        """Compute D_w for all slices in an axial CT series.

        Parameters
        ----------
        hu_arrays : list[np.ndarray]
            HU arrays for each slice (float32).
        pixel_spacing_mm : tuple[float, float]
            (row_spacing_mm, col_spacing_mm).
        slice_positions_mm : list[float]
            Axial position of each slice in mm.
        metadata_list : list[DicomMetadata]
            Metadata for each slice.

        Returns
        -------
        DwSeriesResult

        Raises
        ------
        InsufficientSlicesError
            If fewer than 1 slice provided.
        """
        if len(hu_arrays) < 1:
            raise InsufficientSlicesError(
                "At least 1 axial slice is required, got %d" % len(hu_arrays)
            )

        slice_results: list[DwSliceResult] = []
        for i, (hu_array, meta) in enumerate(zip(hu_arrays, metadata_list)):
            pos = slice_positions_mm[i] if i < len(slice_positions_mm) else 0.0
            result = self.compute_from_single_slice(
                hu_array=hu_array,
                pixel_spacing_mm=pixel_spacing_mm,
                metadata=meta,
            )
            # Override slice_position_mm with the provided value
            result.slice_position_mm = pos
            slice_results.append(result)

        # Sort by slice position
        slice_results.sort(key=lambda r: r.slice_position_mm)

        # Compute series statistics
        dw_values = np.array([r.dw_cm for r in slice_results])
        dw_mean = float(np.mean(dw_values))
        dw_min = float(np.min(dw_values))
        dw_max = float(np.max(dw_values))
        dw_std = float(np.std(dw_values, ddof=1)) if len(dw_values) > 1 else 0.0

        # D_w at isocenter: slice where abs(slice_position_mm) is minimum
        positions = np.array([r.slice_position_mm for r in slice_results])
        isocenter_idx = int(np.argmin(np.abs(positions)))
        dw_at_isocenter = slice_results[isocenter_idx].dw_cm

        # Series metadata from first slice
        series_desc = metadata_list[0].series_description if metadata_list else ""
        acq_date = metadata_list[0].acquisition_date if metadata_list else ""

        logger.info(
            "D_w series: %d slices, mean=%.2f cm, range=[%.2f, %.2f] cm, "
            "isocenter=%.2f cm",
            len(slice_results), dw_mean, dw_min, dw_max, dw_at_isocenter,
        )

        return DwSeriesResult(
            series_description=series_desc,
            acquisition_date=acq_date,
            n_slices=len(slice_results),
            slice_results=slice_results,
            dw_mean_cm=dw_mean,
            dw_min_cm=dw_min,
            dw_max_cm=dw_max,
            dw_std_cm=dw_std,
            dw_at_isocenter_cm=dw_at_isocenter,
            dw_from_localizer_cm=None,
        )

    def compute_from_single_slice(
        self,
        hu_array: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
        metadata: DicomMetadata,
    ) -> DwSliceResult:
        """Compute D_w from a single axial CT slice.

        Parameters
        ----------
        hu_array : np.ndarray
            HU array (rows, cols), float32.
        pixel_spacing_mm : tuple[float, float]
            (row_spacing_mm, col_spacing_mm).
        metadata : DicomMetadata
            Slice metadata.

        Returns
        -------
        DwSliceResult
        """
        # Segment body from axial CT slice
        body_mask = self._segment_body_from_axial(hu_array)

        # Compute water-equivalent area
        # AAPM TG-220 Eq. 1
        water_eq_area = self._compute_water_eq_area_axial(
            hu_array, body_mask, pixel_spacing_mm
        )

        # Compute water-equivalent diameter
        # AAPM TG-220 Eq. 2
        dw = self._dw_from_area(water_eq_area)

        # Get slice position
        slice_pos = self._get_slice_position(metadata, metadata.instance_number)

        n_body = int(np.sum(body_mask))
        n_total = body_mask.size
        body_fraction = n_body / n_total if n_total > 0 else 0.0

        return DwSliceResult(
            slice_position_mm=slice_pos,
            instance_number=metadata.instance_number,
            acquisition_date=metadata.acquisition_date,
            water_eq_area_cm2=water_eq_area,
            dw_cm=dw,
            n_body_pixels=n_body,
            n_total_pixels=n_total,
            body_fraction=body_fraction,
            segmentation_method="largest_contour",
        )

    def compute_from_localizer(
        self,
        localizer_data: "LocalizerData",
    ) -> DwLocalizerResult:
        """Compute D_w per-row from a CT localizer radiograph.

        For each row i of the localizer:
        1. Apply body mask: zero out non-body columns
        2. Compute A_w(i) = sum(t_w_row) × Δx_cm
        3. Compute D_w(i) = 2×sqrt(A_w(i)/π)

        Reference: AAPM TG-220 Appendix A

        Parameters
        ----------
        localizer_data : LocalizerData
            Calibrated localizer data from LocalizerParser.

        Returns
        -------
        DwLocalizerResult
        """
        n_rows = localizer_data.water_eq_path_length_cm.shape[0]
        pixel_spacing_mm = localizer_data.metadata.pixel_spacing_mm

        # Column spacing in cm — AAPM TG-220 Appendix A
        dx_cm = pixel_spacing_mm[1] / 10.0

        water_eq_area_per_row = np.zeros(n_rows, dtype=np.float64)
        dw_per_row = np.zeros(n_rows, dtype=np.float64)

        for i in range(n_rows):
            # Copy row and zero out non-body pixels
            t_w_row = localizer_data.water_eq_path_length_cm[i, :].copy().astype(np.float64)
            t_w_row[~localizer_data.body_mask[i, :]] = 0.0

            # AAPM TG-220 Appendix A — water-equivalent area from localizer
            # A_w(i) = Σ_j [t_w(i,j) × Δx_cm]
            a_w = float(np.sum(t_w_row)) * dx_cm
            water_eq_area_per_row[i] = a_w

            # AAPM TG-220 Eq. 2 — D_w = 2 × sqrt(A_w / π)
            dw_per_row[i] = self._dw_from_area(a_w)

        # Row positions in mm
        row_positions_mm = np.arange(n_rows, dtype=np.float64) * pixel_spacing_mm[0]

        # Statistics
        dw_mean = float(np.mean(dw_per_row))
        center_row = n_rows // 2
        dw_at_center = float(dw_per_row[center_row])

        logger.info(
            "Localizer D_w: %d rows, mean=%.2f cm, center=%.2f cm",
            n_rows, dw_mean, dw_at_center,
        )

        return DwLocalizerResult(
            acquisition_date=localizer_data.metadata.acquisition_date,
            image_orientation=localizer_data.metadata.image_orientation,
            water_eq_area_per_row_cm2=water_eq_area_per_row,
            dw_per_row_cm=dw_per_row,
            row_positions_mm=row_positions_mm,
            dw_mean_cm=dw_mean,
            dw_at_center_row_cm=dw_at_center,
        )

    def compute_from_volume(
        self,
        volumetric_result: "VolumetricQCResult",
    ) -> DwSeriesResult:
        """Convenience method: extract data from VolumetricQCResult.

        This is the integration point with the Phase 1/2 data model.

        Parameters
        ----------
        volumetric_result : VolumetricQCResult
            Pre-computed volumetric QC result from Phase 1/2.

        Returns
        -------
        DwSeriesResult
        """
        hu_arrays = volumetric_result.hu_arrays
        pixel_spacing_mm = volumetric_result.pixel_spacing_mm

        # Build metadata list and slice positions from slice results
        metadata_list: list[DicomMetadata] = []
        slice_positions_mm: list[float] = []

        for sr in volumetric_result.slice_results:
            metadata_list.append(sr.metadata)
            pos = self._get_slice_position(sr.metadata, sr.metadata.instance_number)
            slice_positions_mm.append(pos)

        return self.compute_from_axial_series(
            hu_arrays=hu_arrays,
            pixel_spacing_mm=pixel_spacing_mm,
            slice_positions_mm=slice_positions_mm,
            metadata_list=metadata_list,
        )

    # ─────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────

    def _segment_body_from_axial(
        self,
        hu_array: np.ndarray,
    ) -> np.ndarray:
        """Segment patient body from an axial CT slice.

        Two-stage segmentation:
        Stage 1 — Threshold at HU = -300 (captures tissue, excludes air/table)
        Stage 2 — Largest connected component (removes table and external objects)
        Stage 3 — Fill holes (include lung air)

        Parameters
        ----------
        hu_array : np.ndarray
            HU array (rows, cols), float32.

        Returns
        -------
        np.ndarray
            Boolean body mask.

        Raises
        ------
        BodySegmentationError
            If largest component has fewer than 1000 pixels.
        """
        # Stage 1: Threshold at HU = -300
        # Captures all tissue (soft tissue, fat, bone) while excluding air
        threshold_mask = hu_array > -300.0

        # Stage 2: Largest connected component (removes table and external objects)
        labeled_array, n_components = scipy.ndimage.label(threshold_mask)
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0  # exclude background (label 0)
        largest_label = int(component_sizes.argmax())
        body_mask = labeled_array == largest_label

        largest_size = int(component_sizes[largest_label])

        logger.debug(
            "Body segmentation: %d components found, "
            "largest component: %d pixels",
            n_components, largest_size,
        )

        if largest_size < 1000:
            raise BodySegmentationError(
                "Largest connected component has only %d pixels "
                "(minimum 1000 required). Image may not contain a patient."
                % largest_size
            )

        # Stage 3: Fill holes (include lung air within body contour)
        body_mask = scipy.ndimage.binary_fill_holes(body_mask)

        return body_mask.astype(bool)

    def _compute_water_eq_area_axial(
        self,
        hu_array: np.ndarray,
        body_mask: np.ndarray,
        pixel_spacing_mm: tuple[float, float],
    ) -> float:
        """Compute water-equivalent area from an axial CT slice.

        AAPM TG-220 Eq. 1 — water-equivalent area:
        A_w = Σ[(HU/1000 + 1) × pixel_area_cm²] over body mask

        The factor (HU/1000 + 1) is the linear attenuation coefficient
        relative to water:
        - For water:  0/1000 + 1 = 1.0
        - For air:   -1000/1000 + 1 = 0.0
        - For bone:   1000/1000 + 1 = 2.0

        Parameters
        ----------
        hu_array : np.ndarray
            HU array (rows, cols).
        body_mask : np.ndarray
            Boolean body mask.
        pixel_spacing_mm : tuple[float, float]
            (row_spacing_mm, col_spacing_mm).

        Returns
        -------
        float
            Water-equivalent area A_w in cm².
        """
        # AAPM TG-220 Eq. 1 — water-equivalent area
        # density_factor = HU/1000 + 1 = attenuation relative to water
        # For water: 0/1000 + 1 = 1.0
        # For air:  -1000/1000 + 1 = 0.0
        # For bone:  1000/1000 + 1 = 2.0
        pixel_area_cm2 = (pixel_spacing_mm[0] / 10.0) * (pixel_spacing_mm[1] / 10.0)
        density_factor = hu_array.astype(np.float64) / 1000.0 + 1.0

        # Clamp to [0, None] — pixels with HU < -1000 give negative density,
        # physically impossible
        # CRITICAL: missing this clamp produces negative A_w and undefined D_w
        density_factor = np.clip(density_factor, 0.0, None)

        # Sum only over body mask pixels
        a_w = float(np.sum(density_factor[body_mask])) * pixel_area_cm2

        return a_w

    def _dw_from_area(self, water_eq_area_cm2: float) -> float:
        """Convert water-equivalent area to water-equivalent diameter.

        AAPM TG-220 Eq. 2 — water-equivalent diameter:
        D_w (cm) = 2 × sqrt(A_w / π)

        Parameters
        ----------
        water_eq_area_cm2 : float
            Water-equivalent area in cm².

        Returns
        -------
        float
            Water-equivalent diameter D_w in cm.
        """
        # AAPM TG-220 Eq. 2 — water-equivalent diameter
        # D_w (cm) = 2 × sqrt(A_w / π)
        if water_eq_area_cm2 <= 0.0:
            return 0.0
        dw = 2.0 * np.sqrt(water_eq_area_cm2 / np.pi)

        # Validate against config bounds
        if dw < self._config.dw_min_cm or dw > self._config.dw_max_cm:
            logger.warning(
                "D_w=%.2f cm is outside TG-220 validated table range "
                "[%.1f, %.1f] cm — SSDE f-factor will be extrapolated",
                dw, self._config.dw_min_cm, self._config.dw_max_cm,
            )

        return float(dw)

    def _get_slice_position(
        self,
        metadata: DicomMetadata,
        index: int,
    ) -> float:
        """Extract slice z-position from metadata.

        Tries ImagePositionPatient[2] first. Falls back to
        index × slice_thickness_mm.

        Parameters
        ----------
        metadata : DicomMetadata
            Slice metadata.
        index : int
            Fallback index for position calculation.

        Returns
        -------
        float
            Slice z-position in mm.
        """
        if (
            metadata.image_position_patient is not None
            and len(metadata.image_position_patient) >= 3
        ):
            return float(metadata.image_position_patient[2])

        # Fallback: use slice location if available
        if metadata.slice_location is not None:
            return float(metadata.slice_location)

        # Final fallback: index × slice_thickness
        return float(index) * metadata.slice_thickness_mm
