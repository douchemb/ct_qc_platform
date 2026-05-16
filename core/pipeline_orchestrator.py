# -*- coding: utf-8 -*-
"""
core/pipeline_orchestrator.py — Pipeline Orchestrator.

Top-level coordinator for the CT QC platform.
Dispatches DICOM data to all analysis modules in the correct order.
Implements graceful degradation: individual module failures do not
abort the entire pipeline.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from config import AppConfig, CONFIG
from core.dicom_loader import DicomLoader, DicomMetadata
from modules.image_qc.roi_stats import (
    PhantomROIAnalyzer,
    ROIDescriptor,
    SliceAnalysisResult,
    VolumetricQCResult,
    compute_batch_statistics,
)
from modules.image_qc.nps_calculator import NPSCalculator, NPSResult
from modules.image_qc.mtf_calculator import MTFCalculator, MTFResult
from modules.image_qc.hu_linearity import HULinearityAnalyzer, HULinearityResult
from modules.image_qc.ed_calibration import EDCalibrationAnalyzer, EDCalibrationResult
from modules.dosimetry.localizer_parser import LocalizerParser, LocalizerData, LocalizerNotFoundError
from modules.dosimetry.dw_calculator import DwCalculator, DwSeriesResult, InsufficientSlicesError
from modules.dosimetry.ssde_calculator import SSDECalculator, SSDESeriesResult, MissingCTDIvolError
from modules.dosimetry.dosimetry_report import DosimetryReport
from modules.predictive.metrics_archive import MetricsArchive, QCSessionRecord
from modules.predictive.trend_model import QCTrendModel, TrendModelResult
from modules.predictive.failure_predictor import (
    FailurePredictor,
    FailurePrediction,
    MaintenanceAlert,
    HardwareStatusReport,
)

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = [
    "PipelineOrchestrator",
    "PipelineResult",
    "ImageQCBundle",
    "DosimetryBundle",
    "PredictiveBundle",
]


# ═══════════════════════════════════════════════════════════════════
# Result Bundle Dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ImageQCBundle:
    """All Image QC results from one pipeline run."""
    volumetric_result: Optional[VolumetricQCResult] = None
    nps_result: Optional[NPSResult] = None
    mtf_result: Optional[MTFResult] = None
    hu_linearity_result: Optional[HULinearityResult] = None
    ed_calibration_result: Optional[EDCalibrationResult] = None
    batch_stats: Optional[dict] = None
    all_passed: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class DosimetryBundle:
    """All dosimetry results from one pipeline run."""
    localizer_data: Optional[LocalizerData] = None
    dw_result: Optional[DwSeriesResult] = None
    ssde_result: Optional[SSDESeriesResult] = None
    dosimetry_report: Optional[DosimetryReport] = None
    all_computed: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class PredictiveBundle:
    """All predictive maintenance results from one pipeline run."""
    trend_results: dict[str, TrendModelResult] = field(default_factory=dict)
    predictions: list[FailurePrediction] = field(default_factory=list)
    maintenance_alert: Optional[MaintenanceAlert] = None
    has_sufficient_data: bool = False


@dataclass
class PipelineResult:
    """
    Complete output of one full pipeline run.
    Top-level object returned by run_full_pipeline().
    """
    session_record: Optional[QCSessionRecord] = None
    image_qc: Optional[ImageQCBundle] = None
    dosimetry: Optional[DosimetryBundle] = None
    predictive: Optional[PredictiveBundle] = None
    run_duration_seconds: float = 0.0
    output_files: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Pipeline Orchestrator Class
# ═══════════════════════════════════════════════════════════════════

class PipelineOrchestrator:
    """
    Top-level coordinator for the CT QC platform.
    Dispatches DICOM data to all analysis modules in the correct order.
    Implements graceful degradation: individual module failures do not
    abort the entire pipeline.
    """

    def __init__(
        self,
        config: AppConfig,
        dicom_loader: DicomLoader,
        archive: MetricsArchive,
    ) -> None:
        self._config = config
        self._dicom_loader = dicom_loader
        self._archive = archive

    def run_full_pipeline(
        self,
        dicom_dir: Path,
        rois: list[ROIDescriptor],
        start_slice: int,
        end_slice: int,
        scanner_id: str,
        operator_id: str = "",
        notes: str = "",
        skip_modules: list[str] = None,
        dry_run: bool = False,
    ) -> PipelineResult:
        """Execute the complete pipeline: Image QC → Dosimetry → Predictive.

        Parameters
        ----------
        dicom_dir : Path
            Directory containing DICOM files.
        rois : list[ROIDescriptor]
            ROI layout for phantom analysis.
        start_slice, end_slice : int
            Slice range for volumetric analysis.
        scanner_id : str
            Scanner identifier.
        operator_id : str
            Operator identifier.
        notes : str
            Session notes.
        skip_modules : list[str]
            Modules to skip: "nps", "mtf", "ed", "dosimetry", "predictive".
        dry_run : bool
            If True, do not write any outputs.

        Returns
        -------
        PipelineResult
        """
        start_time = time.monotonic()
        skip = set(skip_modules or [])

        result = PipelineResult()
        image_qc_bundle = None
        dosimetry_bundle = None
        predictive_bundle = None

        # Phase 1: Image QC
        try:
            image_qc_bundle = self.run_image_qc(
                dicom_dir, rois, start_slice, end_slice, skip=skip,
            )
            result.image_qc = image_qc_bundle
            result.errors.extend(image_qc_bundle.errors)
        except Exception as exc:
            logger.error("Image QC phase failed: %s", exc)
            result.errors.append("image_qc: %s" % exc)

        # Phase 2: Dosimetry
        if "dosimetry" not in skip:
            try:
                vol_result = image_qc_bundle.volumetric_result if image_qc_bundle else None
                dosimetry_bundle = self.run_dosimetry(dicom_dir, vol_result)
                result.dosimetry = dosimetry_bundle
                result.errors.extend(dosimetry_bundle.errors)
            except Exception as exc:
                logger.error("Dosimetry phase failed: %s", exc)
                result.errors.append("dosimetry: %s" % exc)

        # Phase 3: Predictive
        if "predictive" not in skip:
            try:
                predictive_bundle = self.run_predictive(scanner_id)
                result.predictive = predictive_bundle
            except Exception as exc:
                logger.error("Predictive phase failed: %s", exc)
                result.errors.append("predictive: %s" % exc)

        # Assemble session record
        result.session_record = self._assemble_session_record(
            image_qc_bundle, dosimetry_bundle, scanner_id, operator_id, notes,
        )

        result.run_duration_seconds = time.monotonic() - start_time

        logger.info(
            "Pipeline complete: %.1f seconds, %d errors",
            result.run_duration_seconds, len(result.errors),
        )

        return result

    def run_image_qc(
        self,
        dicom_dir: Path,
        rois: list[ROIDescriptor],
        start_slice: int,
        end_slice: int,
        skip: set[str] = None,
    ) -> ImageQCBundle:
        """Run the Image QC analysis phase.

        Parameters
        ----------
        dicom_dir : Path
        rois : list[ROIDescriptor]
        start_slice, end_slice : int
        skip : set[str]
            Modules to skip.

        Returns
        -------
        ImageQCBundle
        """
        skip = skip or set()
        bundle = ImageQCBundle()
        volumetric_result = None

        # 1. Volumetric ROI analysis
        try:
            analyzer = PhantomROIAnalyzer(self._dicom_loader, self._config.image_qc)
            volumetric_result = analyzer.analyze_volume(
                dicom_dir, rois, start_slice, end_slice,
            )
            bundle.volumetric_result = volumetric_result
            try:
                bundle.batch_stats = compute_batch_statistics(
                    volumetric_result.slice_results, "center_water",
                )
            except (ValueError, KeyError) as exc:
                logger.warning("Batch statistics failed: %s", exc)
        except Exception as exc:
            logger.error("Image QC volumetric analysis failed: %s", exc)
            bundle.errors.append("volumetric: %s" % exc)

        # 2. NPS
        if "nps" not in skip and volumetric_result is not None:
            try:
                nps_calc = NPSCalculator(self._config.image_qc)
                bundle.nps_result = nps_calc.compute_from_volume(volumetric_result)
            except Exception as exc:
                logger.warning("NPS computation failed: %s", exc)
                bundle.errors.append("nps: %s" % exc)

        # 3. MTF
        if "mtf" not in skip and volumetric_result is not None:
            try:
                mtf_calc = MTFCalculator(self._config.image_qc)
                mid_idx = len(volumetric_result.hu_arrays) // 2
                center_slice = volumetric_result.hu_arrays[mid_idx]
                bundle.mtf_result = mtf_calc.compute_from_edge(
                    center_slice, volumetric_result.pixel_spacing_mm,
                )
            except Exception as exc:
                logger.warning("MTF computation failed: %s", exc)
                bundle.errors.append("mtf: %s" % exc)

        # 4. HU Linearity
        if volumetric_result is not None:
            try:
                hu_analyzer = HULinearityAnalyzer(self._config.image_qc)
                bundle.hu_linearity_result = hu_analyzer.analyze(volumetric_result)
            except Exception as exc:
                logger.warning("HU linearity analysis failed: %s", exc)
                bundle.errors.append("hu_linearity: %s" % exc)

        # 5. ED Calibration
        if "ed" not in skip and volumetric_result is not None:
            try:
                ed_analyzer = EDCalibrationAnalyzer(
                    self._config.image_qc, self._config.ed_calibration,
                )
                bundle.ed_calibration_result = ed_analyzer.calibrate(volumetric_result)
            except Exception as exc:
                logger.warning("ED calibration failed: %s", exc)
                bundle.errors.append("ed_calibration: %s" % exc)

        # Determine pass/fail
        bundle.all_passed = (
            volumetric_result is not None
            and (
                "center_water" not in volumetric_result.volumetric_stats
                or volumetric_result.passes_tg66_volumetric("center_water")
            )
        )

        logger.info(
            "Image QC complete: volumetric=%s, nps=%s, mtf=%s, hu_lin=%s, ed=%s, passed=%s",
            volumetric_result is not None,
            bundle.nps_result is not None,
            bundle.mtf_result is not None,
            bundle.hu_linearity_result is not None,
            bundle.ed_calibration_result is not None,
            bundle.all_passed,
        )

        return bundle

    def run_dosimetry(
        self,
        dicom_dir: Path,
        volumetric_result: Optional[VolumetricQCResult] = None,
    ) -> DosimetryBundle:
        """Run the Dosimetry analysis phase.

        Parameters
        ----------
        dicom_dir : Path
        volumetric_result : VolumetricQCResult, optional

        Returns
        -------
        DosimetryBundle
        """
        bundle = DosimetryBundle()

        # 1. Find and parse localizer
        try:
            parser = LocalizerParser(self._dicom_loader, self._config.dosimetry)
            localizer_paths = parser.find_localizer_in_directory(dicom_dir)
            bundle.localizer_data = parser.parse(localizer_paths[0])
        except LocalizerNotFoundError:
            logger.warning("No localizer found — proceeding without localizer D_w")
        except Exception as exc:
            logger.error("Localizer parsing failed: %s", exc)
            bundle.errors.append("localizer: %s" % exc)

        # 2. Compute D_w from axial slices (primary method)
        try:
            dw_calc = DwCalculator(self._config.dosimetry)
            if volumetric_result is not None:
                bundle.dw_result = dw_calc.compute_from_volume(volumetric_result)
            else:
                raise InsufficientSlicesError("No volumetric result available for D_w")
        except Exception as exc:
            logger.error("D_w computation failed: %s", exc)
            bundle.errors.append("dw: %s" % exc)

        # 3. Compute SSDE
        if bundle.dw_result is not None:
            try:
                ctdi_vol = self._extract_ctdi_vol_from_dir(dicom_dir)
                ssde_calc = SSDECalculator(self._config.dosimetry)
                datasets = self._dicom_loader.load_directory(dicom_dir)
                first_meta = self._dicom_loader.extract_metadata(datasets[0])
                bundle.ssde_result = ssde_calc.compute_series_ssde(
                    bundle.dw_result, ctdi_vol, first_meta,
                )
                bundle.dosimetry_report = DosimetryReport(
                    bundle.ssde_result, bundle.dw_result,
                    localizer_result=None,
                )
            except MissingCTDIvolError:
                logger.warning("CTDIvol tag absent — SSDE cannot be computed")
            except Exception as exc:
                logger.error("SSDE computation failed: %s", exc)
                bundle.errors.append("ssde: %s" % exc)

        bundle.all_computed = bundle.ssde_result is not None

        logger.info(
            "Dosimetry complete: localizer=%s, dw=%s, ssde=%s",
            bundle.localizer_data is not None,
            bundle.dw_result is not None,
            bundle.ssde_result is not None,
        )

        return bundle

    def run_predictive(
        self,
        scanner_id: str,
    ) -> PredictiveBundle:
        """Run the Predictive Maintenance analysis phase.

        Parameters
        ----------
        scanner_id : str

        Returns
        -------
        PredictiveBundle
        """
        today_str = str(date.today())

        trend_model = QCTrendModel(self._config.predictive)
        predictor = FailurePredictor(self._config.predictive)

        trend_results = trend_model.fit_all_metrics(self._archive)

        if not trend_results:
            logger.info(
                "Insufficient historical data for predictive analysis — need >= 2 sessions"
            )
            # Build a minimal empty alert
            empty_report = HardwareStatusReport(
                scanner_id=scanner_id,
                report_date=today_str,
                tube_filament_predictions=[],
                focal_spot_predictions=[],
                kvp_generator_predictions=[],
                tube_filament_urgency="stable",
                focal_spot_urgency="stable",
                kvp_generator_urgency="stable",
                overall_urgency="stable",
                recommended_action="Collect more QC sessions before predictive analysis.",
                recommended_maintenance_date=None,
            )
            empty_alert = MaintenanceAlert(
                scanner_id=scanner_id,
                alert_date=today_str,
                overall_urgency="stable",
                predictions=[],
                hardware_report=empty_report,
                recommended_action="Collect more QC sessions before predictive analysis.",
                recommended_maintenance_date=None,
            )
            return PredictiveBundle(
                trend_results={},
                predictions=[],
                maintenance_alert=empty_alert,
                has_sufficient_data=False,
            )

        alert = predictor.generate_maintenance_alert(
            trend_results, self._archive, scanner_id,
        )

        return PredictiveBundle(
            trend_results=trend_results,
            predictions=alert.predictions,
            maintenance_alert=alert,
            has_sufficient_data=True,
        )

    # ─────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────

    def _assemble_session_record(
        self,
        image_qc: Optional[ImageQCBundle],
        dosimetry: Optional[DosimetryBundle],
        scanner_id: str,
        operator_id: str,
        notes: str,
    ) -> QCSessionRecord:
        """Assemble a QCSessionRecord from analysis bundles.

        Returns
        -------
        QCSessionRecord
        """
        now = datetime.now(timezone.utc)
        session_id = str(uuid.uuid4())

        record = QCSessionRecord(
            session_id=session_id,
            session_date=now.strftime("%Y-%m-%d"),
            session_timestamp=now.isoformat(),
            scanner_id=scanner_id,
            operator_id=operator_id,
            notes=notes,
            schema_version="1.0",
        )

        # Extract Image QC metrics
        if image_qc is not None and image_qc.volumetric_result is not None:
            vol = image_qc.volumetric_result
            if "center_water" in vol.volumetric_stats:
                cw = vol.volumetric_stats["center_water"]
                record.center_water_mean_hu = cw.mean_hu_mean
                record.center_water_std_hu = cw.std_hu_mean
                record.center_water_variance_hu = cw.variance_hu_mean

            record.all_image_qc_passed = image_qc.all_passed

            # NPS metrics
            if image_qc.nps_result is not None:
                record.nps_peak_frequency_lpmm = image_qc.nps_result.peak_frequency_lpmm
                record.nps_peak_value_hu2mm2 = image_qc.nps_result.peak_value_hu2mm2
                record.nps_integral_hu2mm2 = image_qc.nps_result.integral_hu2mm2

            # MTF metrics
            if image_qc.mtf_result is not None:
                record.mtf_50_lpmm = image_qc.mtf_result.mtf_50_lpmm
                record.mtf_10_lpmm = image_qc.mtf_result.mtf_10_lpmm

            # HU Linearity
            if image_qc.hu_linearity_result is not None:
                record.hu_linearity_max_deviation_hu = image_qc.hu_linearity_result.max_deviation_hu
                record.hu_linearity_r_squared = image_qc.hu_linearity_result.r_squared

            # ED Calibration
            if image_qc.ed_calibration_result is not None:
                record.ed_calibration_passed = image_qc.ed_calibration_result.all_passed
                # Extract slopes if available
                if hasattr(image_qc.ed_calibration_result, 'soft_tissue_slope'):
                    record.ed_soft_tissue_slope = image_qc.ed_calibration_result.soft_tissue_slope
                if hasattr(image_qc.ed_calibration_result, 'bone_slope'):
                    record.ed_bone_slope = image_qc.ed_calibration_result.bone_slope

        # Extract Dosimetry metrics
        if dosimetry is not None and dosimetry.ssde_result is not None:
            ssde = dosimetry.ssde_result
            record.ctdi_vol_mgy = ssde.ctdi_vol_mgy
            record.ssde_mean_mgy = ssde.ssde_mean_mgy
            record.effective_dose_msv = ssde.effective_dose_msv
            record.all_dosimetry_computed = True

            if dosimetry.dw_result is not None:
                record.dw_mean_cm = dosimetry.dw_result.dw_mean_cm

        return record

    def _extract_ctdi_vol_from_dir(self, dicom_dir: Path) -> float:
        """Extract CTDIvol from the first CT slice in a directory.

        Returns
        -------
        float
            CTDIvol in mGy.

        Raises
        ------
        MissingCTDIvolError
            If CTDIvol tag is not present.
        """
        datasets = self._dicom_loader.load_directory(dicom_dir)
        if not datasets:
            raise MissingCTDIvolError("No DICOM files found in %s" % dicom_dir)

        meta = self._dicom_loader.extract_metadata(datasets[0])
        if meta.ctdi_vol is None:
            raise MissingCTDIvolError(
                "CTDIvol tag not present in %s" % dicom_dir
            )
        return meta.ctdi_vol
