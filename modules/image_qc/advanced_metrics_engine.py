"""
modules/image_qc/advanced_metrics_engine.py
=============================================
Advanced Tier Metrics Engine — Scientific QA Analysis.

Orchestrates the full scientific QA pipeline by routing to the correct
existing calculators based on the phantom adapter capabilities and scanner
profile. This engine does NOT implement any physics — it calls existing
modules (NPSCalculator, MTFCalculator, EDCalibrationAnalyzer, SSDECalculator)
via dependency injection.

Reference: AAPM TG-66; AAPM TG-233; AAPM TG-220; IAEA TRS-430.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from modules.image_qc.mtf_calculator import MTFResult

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class AdvancedQAResult:
    """Complete Advanced Tier QA result for one CT session."""
    acquisition_date: str
    series_description: str
    scanner_id: str
    phantom_id: str

    nps: Optional["NPSResult"]                          = None
    mtf: Optional["MTFResult"]                          = None
    hu_linearity: Optional["HULinearityResult"]         = None
    ed_calibration: Optional["EDCalibrationResult"]     = None
    dw_series: Optional["DwSeriesResult"]               = None
    ssde_series: Optional["SSDESeriesResult"]           = None

    all_passed: bool = False
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    errors: list[str]   = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[str]  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "acquisition_date":  self.acquisition_date,
            "series_description": self.series_description,
            "scanner_id":        self.scanner_id,
            "phantom_id":        self.phantom_id,
            "nps":               self.nps.to_dict()           if self.nps           else None,
            "mtf":               self.mtf.to_dict()           if self.mtf           else None,
            "hu_linearity":      self.hu_linearity.to_dict()  if self.hu_linearity  else None,
            "ed_calibration":    self.ed_calibration.to_dict() if self.ed_calibration else None,
            "dw_series":         self.dw_series.to_dict()     if self.dw_series     else None,
            "ssde_series":       self.ssde_series.to_dict()   if self.ssde_series   else None,
            "all_passed":        self.all_passed,
            "computed_at":       self.computed_at,
            "errors":            self.errors,
            "warnings":          self.warnings,
            "skipped":           self.skipped,
        }

    def _evaluate_all_passed(self) -> None:
        checks = []
        if self.nps:
            checks.append(self.nps.noise_std_from_nps <= 5.0)  # TG-66 noise tolerance
        if self.mtf:
            checks.append(self.mtf.passes_resolution_check())
        if self.hu_linearity:
            checks.append(self.hu_linearity.all_passed)
        if self.ed_calibration:
            checks.append(self.ed_calibration.all_passed)
        if self.ssde_series:
            checks.append(self.ssde_series.passes_diagnostic_reference_level())
        self.all_passed = all(checks) if checks else False


# ── Engine ─────────────────────────────────────────────────────────────────

class AdvancedMetricsEngine:
    """
    Routes scientific QA computation to the correct existing calculators
    based on PhantomAdapter capabilities and ScannerProfile configuration.

    All injected calculators are optional — if not provided, the
    corresponding metric is skipped with a note in result.skipped.
    """

    def __init__(
        self,
        nps_calculator=None,
        mtf_calculator=None,
        hu_analyzer=None,
        ed_analyzer=None,
        dw_calculator=None,
        ssde_calculator=None,
    ) -> None:
        self._nps  = nps_calculator
        self._mtf  = mtf_calculator
        self._hu   = hu_analyzer
        self._ed   = ed_analyzer
        self._dw   = dw_calculator
        self._ssde = ssde_calculator

    def compute(
        self,
        volumetric_result: "VolumetricQCResult",
        phantom_adapter: "PhantomAdapter",
        scanner_id: str = "UNKNOWN",
        dose_metadata: Optional["DoseMetadata"] = None,
        skip_modules: Optional[list[str]] = None,
        edge_roi: Optional["ROIDescriptor"] = None,
        mtf_override: Optional["MTFResult"] = None,
    ) -> AdvancedQAResult:
        """Orchestrates all advanced metric computations.

        Parameters
        ----------
        mtf_override : MTFResult, optional
            If provided, this pre-computed MTF result (from the classified
            MTF slice) is used directly instead of recomputing from the
            volumetric mid-slice.  This ensures the correct wire/edge
            slice data feeds the MTF calculator.
        """
        skip = set(skip_modules or [])
        vol  = volumetric_result
        result = AdvancedQAResult(
            acquisition_date   = vol.acquisition_date,
            series_description = vol.series_description,
            scanner_id         = scanner_id,
            phantom_id         = phantom_adapter.phantom_id,
        )

        # ── NPS ────────────────────────────────────────────────────────
        if "nps" in skip:
            result.skipped.append("nps")
        elif self._nps is None:
            result.skipped.append("nps (calculator not injected)")
        else:
            try:
                from modules.image_qc.nps_calculator import InsufficientPatchesError
                result.nps = self._nps.compute_from_volume(vol)
            except InsufficientPatchesError:
                result.warnings.append("NPS: insufficient patches — skipped.")
                result.skipped.append("nps (insufficient patches)")
            except Exception as exc:
                result.errors.append(f"NPS computation failed: {exc}")

        # ── MTF ────────────────────────────────────────────────────────
        if mtf_override is not None:
            # Pre-computed MTF from the correctly classified MTF slice
            result.mtf = mtf_override
            logger.info("MTF: using pre-computed override (MTF50=%.3f lp/mm)",
                        mtf_override.mtf_50_lpmm)
        elif "mtf" in skip:
            result.skipped.append("mtf")
        elif not phantom_adapter.has_edge_insert() and edge_roi is None:
            result.skipped.append("mtf (no edge insert in phantom)")
        elif self._mtf is None:
            result.skipped.append("mtf (calculator not injected)")
        else:
            try:
                mid_idx = len(vol.hu_arrays) // 2
                hu_mid  = vol.hu_arrays[mid_idx]
                spacing = vol.pixel_spacing_mm[0]
                meta    = vol.slice_results[mid_idx].metadata

                if edge_roi is None:
                    rows, cols = hu_mid.shape
                    edge_roi = _build_edge_roi(rows, cols)

                result.mtf = self._mtf.compute_from_edge(hu_mid, spacing, meta, edge_roi)
            except Exception as exc:
                result.errors.append(f"MTF computation failed: {exc}")

        # ── HU Linearity ───────────────────────────────────────────────
        if "hu_linearity" in skip:
            result.skipped.append("hu_linearity")
        elif self._hu is None:
            result.skipped.append("hu_linearity (analyzer not injected)")
        else:
            try:
                material_rois = phantom_adapter.get_roi_descriptors(
                    vol.hu_arrays[len(vol.hu_arrays)//2], vol.pixel_spacing_mm,
                )
                refs     = phantom_adapter.get_material_references()
                mat_rois = {k: v for k, v in material_rois.items() if k in refs}
                if len(mat_rois) >= 2:
                    result.hu_linearity = self._hu.analyze(vol.slice_results, mat_rois)
                else:
                    result.skipped.append(f"hu_linearity (< 2 material ROIs; found {len(mat_rois)})")
            except Exception as exc:
                result.errors.append(f"HU linearity failed: {exc}")

        # ── ED Calibration ─────────────────────────────────────────────
        if "ed" in skip:
            result.skipped.append("ed_calibration")
        elif not phantom_adapter.has_density_inserts():
            result.skipped.append(
                f"ed_calibration (phantom '{phantom_adapter.phantom_id}' has no density inserts)"
            )
        elif self._ed is None:
            result.skipped.append("ed_calibration (analyzer not injected)")
        else:
            try:
                material_rois = {
                    k: v for k, v in phantom_adapter.get_roi_descriptors(
                        vol.hu_arrays[len(vol.hu_arrays)//2], vol.pixel_spacing_mm,
                    ).items()
                    if k != "water"
                }
                if material_rois:
                    result.ed_calibration = self._ed.analyze(vol, material_rois, scanner_id=scanner_id)
                else:
                    result.skipped.append("ed_calibration (no material ROIs)")
            except Exception as exc:
                result.errors.append(f"ED calibration failed: {exc}")

        # ── D_w and SSDE ───────────────────────────────────────────────
        if "ssde" in skip:
            result.skipped.append("ssde")
        elif dose_metadata is None or not dose_metadata.has_ctdi_vol:
            result.skipped.append("ssde (CTDIvol missing)")
        elif self._dw is None or self._ssde is None:
            result.skipped.append("ssde (calculators not injected)")
        else:
            try:
                result.dw_series = self._dw.compute_from_volume(vol)
                first_meta       = vol.slice_results[0].metadata
                result.ssde_series = self._ssde.compute_series_ssde(
                    result.dw_series, dose_metadata.ctdi_vol_mgy, first_meta,
                )
            except Exception as exc:
                result.errors.append(f"SSDE computation failed: {exc}")

        result._evaluate_all_passed()
        return result


def _build_edge_roi(rows: int, cols: int) -> "ROIDescriptor":
    """Builds a default edge ROI in the upper-right quadrant."""
    from modules.image_qc.roi_stats import ROIDescriptor
    size = min(rows, cols) // 4
    return ROIDescriptor(
        label="auto_edge", row_start=rows // 4,
        col_start=cols // 2, height_px=size, width_px=size,
    )
