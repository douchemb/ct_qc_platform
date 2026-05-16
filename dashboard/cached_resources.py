"""
dashboard/cached_resources.py — Cached Resource Initializers.
All heavy instantiations wrapped with @st.cache_resource.
"""
from __future__ import annotations

import streamlit as st

from config import CONFIG
from core.dicom_loader import DicomLoader
from core.scanner_profiles import ScannerProfileRegistry
from modules.image_qc.phantom_adapters import PhantomAdapterFactory
from modules.image_qc.basic_metrics import BasicMetricsEngine
from modules.image_qc.advanced_metrics_engine import AdvancedMetricsEngine
from modules.image_qc.nps_calculator import NPSCalculator
from modules.image_qc.mtf_calculator import MTFCalculator
from modules.image_qc.hu_linearity import HULinearityAnalyzer
from modules.image_qc.ed_calibration import EDCalibrationAnalyzer
from modules.dosimetry.dw_calculator import DwCalculator
from modules.dosimetry.ssde_calculator import SSDECalculator
from modules.predictive.metrics_archive import MetricsArchive


@st.cache_resource
def get_loader() -> DicomLoader:
    return DicomLoader(CONFIG.dicom)


@st.cache_resource
def get_registry() -> ScannerProfileRegistry:
    return ScannerProfileRegistry()


@st.cache_resource
def get_adapter_factory() -> PhantomAdapterFactory:
    return PhantomAdapterFactory()


@st.cache_resource
def get_archive() -> MetricsArchive:
    return MetricsArchive(CONFIG.paths.archive_file)


@st.cache_resource
def get_basic_engine() -> BasicMetricsEngine:
    return BasicMetricsEngine(
        noise_tolerance_hu=CONFIG.image_qc.noise_tolerance_hu,
        uniformity_tolerance_hu=CONFIG.image_qc.noise_tolerance_hu,
        hu_accuracy_tolerance_hu=CONFIG.image_qc.hu_linearity_tolerance,
        slice_thickness_tol_mm=1.0,
        min_cnr=0.5,
    )


@st.cache_resource
def get_advanced_engine() -> AdvancedMetricsEngine:
    return AdvancedMetricsEngine(
        nps_calculator=NPSCalculator(CONFIG.image_qc),
        mtf_calculator=MTFCalculator(CONFIG.image_qc),
        hu_analyzer=HULinearityAnalyzer(CONFIG.image_qc),
        ed_analyzer=EDCalibrationAnalyzer(CONFIG),
        dw_calculator=DwCalculator(CONFIG.dosimetry),
        ssde_calculator=SSDECalculator(CONFIG.dosimetry),
    )
