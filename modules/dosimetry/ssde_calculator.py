# -*- coding: utf-8 -*-
"""
modules/dosimetry/ssde_calculator.py — Size-Specific Dose Estimate Calculator.

Computes SSDE per AAPM Report 204 (2011):
    SSDE (mGy) = f(D_w) × CTDIvol (mGy)

The f-factor is a dimensionless conversion factor that depends on the
water-equivalent diameter D_w of the patient. It corrects the scanner-reported
CTDIvol (measured in a 32 cm PMMA phantom) for the actual patient size.

Also computes:
    - Dose-Length Product (DLP) per AAPM Report 96
    - Effective dose estimate per ICRP Publication 103 / EUR 16262

Standards References:
    - AAPM Report 204 (2011): Size-Specific Dose Estimates, Eq. 1
    - AAPM Report 96: Measurement, Reporting, and Management of CT Dose
    - ICRP Publication 103: Effective dose k-factors
    - European Commission EUR 16262: Diagnostic Reference Levels
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from config import DosimetryConfig
from core.dicom_loader import DicomMetadata
from modules.dosimetry.dw_calculator import DwSeriesResult, DwSliceResult

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = [
    "SSDECalculator",
    "SSDESliceResult",
    "SSDESeriesResult",
    "MissingCTDIvolError",
    "SSDEComputationError",
]


# ═══════════════════════════════════════════════════════════════════
# Custom Exceptions
# ═══════════════════════════════════════════════════════════════════

class MissingCTDIvolError(ValueError):
    """
    Raised when CTDIvol is not available in DICOM metadata.
    CTDIvol is mandatory for SSDE calculation.
    Without it, SSDE = f × CTDIvol cannot be evaluated.
    """


class SSDEComputationError(RuntimeError):
    """Raised when SSDE computation fails for reasons other than missing data."""


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SSDESliceResult:
    """
    SSDE for one axial slice position.
    ssde_mgy = f_factor × ctdi_vol_mgy  (AAPM Report 204 Eq. 1)
    """
    slice_position_mm: float            # z-position in mm
    instance_number: int
    dw_cm: float                        # water-equivalent diameter, cm
    ctdi_vol_mgy: float                 # scanner-reported CTDIvol, mGy
    f_factor: float                     # dimensionless conversion factor
    ssde_mgy: float                     # size-specific dose estimate, mGy
    f_interpolation_method: str         # "table_lookup", "interpolated", "extrapolated"
    within_table_range: bool            # True if D_w within [10, 40] cm

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return asdict(self)


@dataclass
class SSDESeriesResult:
    """
    SSDE results for a complete CT series.

    ssde_mean_mgy is the primary clinical dosimetry output.
    dlp_mgy_cm and effective_dose_msv are derived estimates.

    Reference: AAPM Report 204 (2011); ICRP Publication 103.
    """
    series_description: str
    acquisition_date: str
    patient_id: str
    ctdi_vol_mgy: float                 # scanner-reported CTDIvol, mGy
    n_slices: int
    slice_results: list[SSDESliceResult]
    ssde_mean_mgy: float                # mean SSDE across slices, mGy
    ssde_min_mgy: float                 # min SSDE, mGy
    ssde_max_mgy: float                 # max SSDE, mGy
    ssde_at_isocenter_mgy: float        # SSDE at slice closest to z=0, mGy
    dlp_mgy_cm: float                   # dose-length product, mGy·cm
    effective_dose_msv: float           # effective dose estimate, mSv
    scan_length_cm: float               # scan length, cm

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "series_description": self.series_description,
            "acquisition_date": self.acquisition_date,
            "patient_id": self.patient_id,
            "ctdi_vol_mgy": self.ctdi_vol_mgy,
            "n_slices": self.n_slices,
            "slice_results": [s.to_dict() for s in self.slice_results],
            "ssde_mean_mgy": self.ssde_mean_mgy,
            "ssde_min_mgy": self.ssde_min_mgy,
            "ssde_max_mgy": self.ssde_max_mgy,
            "ssde_at_isocenter_mgy": self.ssde_at_isocenter_mgy,
            "dlp_mgy_cm": self.dlp_mgy_cm,
            "effective_dose_msv": self.effective_dose_msv,
            "scan_length_cm": self.scan_length_cm,
        }

    def passes_diagnostic_reference_level(
        self,
        drl_ctdi_vol_mgy: float = 25.0,
    ) -> bool:
        """
        Returns True if CTDIvol is below the Diagnostic Reference Level.
        Default DRL: 25 mGy (European Commission EUR 16262, abdomen CT 120 kVp).
        Reference: EUR 16262 EN (1999, updated 2014).
        """
        return self.ctdi_vol_mgy < drl_ctdi_vol_mgy


# ═══════════════════════════════════════════════════════════════════
# SSDE Calculator Class
# ═══════════════════════════════════════════════════════════════════

class SSDECalculator:
    """
    Computes Size-Specific Dose Estimate per AAPM Report 204 (2011).
    SSDE = f(D_w) × CTDIvol
    """

    # k-factors from ICRP Publication 103 / EUR 16262
    # Units: mSv / (mGy·cm)
    K_FACTORS: dict[str, float] = {
        "head":                   0.0023,
        "neck":                   0.0054,
        "chest":                  0.014,
        "abdomen":                0.015,
        "pelvis":                 0.015,
        "chest_abdomen_pelvis":   0.015,
        "spine":                  0.014,
    }

    def __init__(self, config: DosimetryConfig) -> None:
        self._config = config

        # Pre-extract table arrays for efficiency
        self._dw_table = np.array(
            [row[0] for row in config.ssde_conversion_table]
        )
        self._f_table = np.array(
            [row[1] for row in config.ssde_conversion_table]
        )

    def compute_series_ssde(
        self,
        dw_series: DwSeriesResult,
        ctdi_vol_mgy: float,
        metadata: DicomMetadata,
        body_region: str = "abdomen",
    ) -> SSDESeriesResult:
        """Compute SSDE for all slices in a D_w series.

        Parameters
        ----------
        dw_series : DwSeriesResult
            D_w results from DwCalculator.
        ctdi_vol_mgy : float
            Scanner-reported CTDIvol in mGy.
        metadata : DicomMetadata
            Metadata from the series (for patient_id, etc.).
        body_region : str
            Body region for effective dose k-factor lookup.

        Returns
        -------
        SSDESeriesResult
        """
        # Compute SSDE for each slice
        slice_results: list[SSDESliceResult] = []
        for dw_slice in dw_series.slice_results:
            ssde_slice = self.compute_slice_ssde(dw_slice, ctdi_vol_mgy)
            slice_results.append(ssde_slice)

        # Series statistics
        ssde_values = np.array([s.ssde_mgy for s in slice_results])
        ssde_mean = float(np.mean(ssde_values))
        ssde_min = float(np.min(ssde_values))
        ssde_max = float(np.max(ssde_values))

        # SSDE at isocenter: slice closest to z=0
        positions = np.array([s.slice_position_mm for s in slice_results])
        isocenter_idx = int(np.argmin(np.abs(positions)))
        ssde_at_isocenter = slice_results[isocenter_idx].ssde_mgy

        # Scan length in cm
        if len(slice_results) > 1:
            scan_length_cm = (float(np.max(positions)) - float(np.min(positions))) / 10.0
        else:
            scan_length_cm = metadata.slice_thickness_mm / 10.0

        # Ensure scan_length is positive
        scan_length_cm = max(scan_length_cm, metadata.slice_thickness_mm / 10.0)

        # DLP and effective dose
        dlp = self._compute_dlp(ssde_mean, scan_length_cm)
        effective_dose = self._estimate_effective_dose(dlp, body_region)

        logger.info(
            "SSDE series: %d slices, mean=%.2f mGy, CTDIvol=%.2f mGy, "
            "DLP=%.1f mGy·cm, E=%.2f mSv",
            len(slice_results), ssde_mean, ctdi_vol_mgy, dlp, effective_dose,
        )

        return SSDESeriesResult(
            series_description=metadata.series_description,
            acquisition_date=metadata.acquisition_date,
            patient_id=metadata.patient_id,
            ctdi_vol_mgy=ctdi_vol_mgy,
            n_slices=len(slice_results),
            slice_results=slice_results,
            ssde_mean_mgy=ssde_mean,
            ssde_min_mgy=ssde_min,
            ssde_max_mgy=ssde_max,
            ssde_at_isocenter_mgy=ssde_at_isocenter,
            dlp_mgy_cm=dlp,
            effective_dose_msv=effective_dose,
            scan_length_cm=scan_length_cm,
        )

    def compute_slice_ssde(
        self,
        dw_slice: DwSliceResult,
        ctdi_vol_mgy: float,
    ) -> SSDESliceResult:
        """Compute SSDE for a single slice.

        AAPM Report 204, Equation 1:
        SSDE (mGy) = f(D_w) × CTDIvol (mGy)

        Parameters
        ----------
        dw_slice : DwSliceResult
            D_w result for one slice.
        ctdi_vol_mgy : float
            Scanner-reported CTDIvol in mGy.

        Returns
        -------
        SSDESliceResult
        """
        # AAPM Report 204, Equation 1
        # SSDE (mGy) = f(D_w) × CTDIvol (mGy)
        f_factor, method, in_range = self._lookup_f_factor(dw_slice.dw_cm)
        ssde = f_factor * ctdi_vol_mgy

        logger.debug(
            "Slice z=%.1f mm: D_w=%.2f cm, f=%.4f, CTDIvol=%.2f mGy → SSDE=%.2f mGy",
            dw_slice.slice_position_mm, dw_slice.dw_cm, f_factor,
            ctdi_vol_mgy, ssde,
        )

        return SSDESliceResult(
            slice_position_mm=dw_slice.slice_position_mm,
            instance_number=dw_slice.instance_number,
            dw_cm=dw_slice.dw_cm,
            ctdi_vol_mgy=ctdi_vol_mgy,
            f_factor=f_factor,
            ssde_mgy=ssde,
            f_interpolation_method=method,
            within_table_range=in_range,
        )

    def _lookup_f_factor(
        self,
        dw_cm: float,
    ) -> tuple[float, str, bool]:
        """Look up the f(D_w) conversion factor from AAPM Report 204 Table A.

        Logic:
        - Exact match (within 1e-6): table_lookup
        - Within [10, 40] cm: numpy.interp (interpolated)
        - Below 10 cm: linear extrapolation from first two points
        - Above 40 cm: linear extrapolation from last two points

        Parameters
        ----------
        dw_cm : float
            Water-equivalent diameter in cm.

        Returns
        -------
        tuple[float, str, bool]
            (f_factor, method, within_table_range)
        """
        dw_table = self._dw_table
        f_table = self._f_table

        # Check for exact table match (within 1e-6)
        for i in range(len(dw_table)):
            if abs(dw_cm - dw_table[i]) < 1e-6:
                return (float(f_table[i]), "table_lookup", True)

        # Within table range: interpolate
        if dw_table[0] <= dw_cm <= dw_table[-1]:
            f = float(np.interp(dw_cm, dw_table, f_table))
            return (f, "interpolated", True)

        # Below table range: extrapolate from first two points
        if dw_cm < dw_table[0]:
            # Linear extrapolation: f = f[0] + (dw - dw[0]) * slope
            slope = (f_table[1] - f_table[0]) / (dw_table[1] - dw_table[0])
            f = float(f_table[0] + (dw_cm - dw_table[0]) * slope)
            logger.warning(
                "D_w=%.2f cm is outside TG-220 table range [%.1f, %.1f] cm "
                "— f-factor extrapolated, increased uncertainty",
                dw_cm, dw_table[0], dw_table[-1],
            )
            return (f, "extrapolated", False)

        # Above table range: extrapolate from last two points
        slope = (f_table[-1] - f_table[-2]) / (dw_table[-1] - dw_table[-2])
        f = float(f_table[-1] + (dw_cm - dw_table[-1]) * slope)
        logger.warning(
            "D_w=%.2f cm is outside TG-220 table range [%.1f, %.1f] cm "
            "— f-factor extrapolated, increased uncertainty",
            dw_cm, dw_table[0], dw_table[-1],
        )
        return (f, "extrapolated", False)

    def _compute_dlp(
        self,
        ssde_mean_mgy: float,
        scan_length_cm: float,
    ) -> float:
        """Compute Dose-Length Product.

        DLP (mGy·cm) = SSDE_mean × scan_length
        Reference: AAPM Report 96, Section 3

        Parameters
        ----------
        ssde_mean_mgy : float
            Mean SSDE in mGy.
        scan_length_cm : float
            Scan length in cm.

        Returns
        -------
        float
            DLP in mGy·cm.
        """
        # DLP (mGy·cm) = SSDE_mean × scan_length
        # Reference: AAPM Report 96, Section 3
        dlp = ssde_mean_mgy * scan_length_cm
        return dlp

    def _estimate_effective_dose(
        self,
        dlp_mgy_cm: float,
        body_region: str,
    ) -> float:
        """Estimate effective dose from DLP.

        E (mSv) = k × DLP (mGy·cm)
        Reference: ICRP Publication 103; EUR 16262 EN

        Parameters
        ----------
        dlp_mgy_cm : float
            Dose-length product in mGy·cm.
        body_region : str
            Body region for k-factor selection.

        Returns
        -------
        float
            Effective dose in mSv.
        """
        # E (mSv) = k × DLP (mGy·cm)
        # Reference: ICRP Publication 103; EUR 16262 EN
        k = self.K_FACTORS.get(body_region, self.K_FACTORS["abdomen"])
        if body_region not in self.K_FACTORS:
            logger.warning(
                "Unknown body region '%s', defaulting to abdomen k=0.015",
                body_region,
            )
        effective_dose = k * dlp_mgy_cm
        return effective_dose
