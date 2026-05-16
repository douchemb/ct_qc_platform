# -*- coding: utf-8 -*-
"""tests/test_hu_linearity.py — HU linearity analyzer tests."""

import numpy as np
import pytest

from config import CONFIG
from core.dicom_loader import DicomMetadata
from modules.image_qc.hu_linearity import HULinearityAnalyzer, HULinearityResult
from modules.image_qc.roi_stats import (
    SliceAnalysisResult, ROIStatistics, ROIDescriptor,
)


def _make_slice_result(roi_stats_list, acq_date="20240115", series="QA_HEAD"):
    """Helper: create a synthetic SliceAnalysisResult with given ROI stats."""
    metadata = DicomMetadata(
        sop_instance_uid="1.2.3.4.5",
        series_description=series,
        acquisition_date=acq_date,
        patient_id="QA",
        kvp=120.0,
        mas=100.0,
        slice_thickness_mm=3.0,
        pixel_spacing_mm=(0.977, 0.977),
        rescale_slope=1.0,
        rescale_intercept=-1024.0,
        instance_number=1,
        slice_location=0.0,
        image_position_patient=(0.0, 0.0, 0.0),
        ctdi_vol=12.5,
        reconstruction_kernel="STANDARD",
        x_ray_tube_current=200.0,
    )
    return SliceAnalysisResult(
        metadata=metadata, roi_results=roi_stats_list,
        analysis_timestamp="2024-01-15T00:00:00Z", source_file="test.dcm",
    )


def _make_roi_stat(label, mean_hu, std_hu=3.0):
    return ROIStatistics(
        roi_label=label, slice_file="test.dcm", acquisition_date="20240115",
        mean_hu=mean_hu, std_hu=std_hu, variance_hu=std_hu ** 2,
        min_hu=mean_hu - 10, max_hu=mean_hu + 10,
        snr=abs(mean_hu) / std_hu if std_hu > 0 else 0,
        skewness=0.0, kurtosis=0.0, n_pixels=3600,
        roi_row_start=0, roi_col_start=0, roi_height_px=60, roi_width_px=60,
    )


@pytest.fixture
def hu_analyzer():
    return HULinearityAnalyzer(config=CONFIG.image_qc)


@pytest.fixture
def perfect_slice_results():
    """Slice results where measured HU exactly matches nominal."""
    stats = [
        _make_roi_stat("water", CONFIG.image_qc.hu_water_nominal),
        _make_roi_stat("air", CONFIG.image_qc.hu_air_nominal),
        _make_roi_stat("acrylic", CONFIG.image_qc.hu_acrylic_nominal),
    ]
    return [_make_slice_result(stats)]


@pytest.fixture
def material_rois():
    return {
        "water": ROIDescriptor("water", 226, 226, 60, 60),
        "air": ROIDescriptor("air", 80, 80, 40, 40),
        "acrylic": ROIDescriptor("acrylic", 236, 390, 40, 40),
    }


class TestHULinearityAnalyzer:
    def test_analyze_runs(self, hu_analyzer, perfect_slice_results, material_rois):
        result = hu_analyzer.analyze(perfect_slice_results, material_rois)
        assert isinstance(result, HULinearityResult)

    def test_all_passed_within_tolerance(self, hu_analyzer, perfect_slice_results, material_rois):
        result = hu_analyzer.analyze(perfect_slice_results, material_rois)
        assert result.all_passed is True

    def test_fails_with_deviation(self, hu_analyzer, material_rois):
        """One material with +10 HU deviation should fail."""
        stats = [
            _make_roi_stat("water", CONFIG.image_qc.hu_water_nominal + 10.0),
            _make_roi_stat("air", CONFIG.image_qc.hu_air_nominal),
            _make_roi_stat("acrylic", CONFIG.image_qc.hu_acrylic_nominal),
        ]
        results = [_make_slice_result(stats)]
        result = hu_analyzer.analyze(results, material_rois)
        assert result.all_passed is False

    def test_r_squared_range(self, hu_analyzer, perfect_slice_results, material_rois):
        result = hu_analyzer.analyze(perfect_slice_results, material_rois)
        assert 0.0 <= result.r_squared <= 1.0

    def test_perfect_linearity_slope_intercept(self, hu_analyzer, perfect_slice_results, material_rois):
        result = hu_analyzer.analyze(perfect_slice_results, material_rois)
        assert result.slope == pytest.approx(1.0, abs=0.01)
        assert result.intercept == pytest.approx(0.0, abs=1.0)

    def test_to_dict_serializable(self, hu_analyzer, perfect_slice_results, material_rois):
        result = hu_analyzer.analyze(perfect_slice_results, material_rois)
        d = result.to_dict()
        assert isinstance(d, dict)
        # No numpy arrays
        _assert_no_ndarray(d)

    def test_passes_tg66(self, hu_analyzer, perfect_slice_results, material_rois):
        result = hu_analyzer.analyze(perfect_slice_results, material_rois)
        assert isinstance(result.passes_tg66(), bool)


def _assert_no_ndarray(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_no_ndarray(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _assert_no_ndarray(item)
    else:
        assert not isinstance(obj, np.ndarray)
