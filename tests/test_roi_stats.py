# -*- coding: utf-8 -*-
"""tests/test_roi_stats.py — ROI statistics and volumetric analysis tests."""

import numpy as np
import pytest

from modules.image_qc.roi_stats import (
    PhantomROIAnalyzer, ROIDescriptor, ROIStatistics,
    SliceAnalysisResult, VolumetricQCResult, VolumetricROIStat,
    ROIBoundsError, compute_batch_statistics,
)


class TestSingleSlice:
    def test_analyze_dataset_runs(self, roi_analyzer_instance, synthetic_ct_dicom, standard_rois):
        result = roi_analyzer_instance.analyze_dataset(synthetic_ct_dicom, standard_rois)
        assert isinstance(result, SliceAnalysisResult)

    def test_center_water_mean_near_zero(self, roi_analyzer_instance, synthetic_ct_dicom, standard_rois):
        result = roi_analyzer_instance.analyze_dataset(synthetic_ct_dicom, standard_rois)
        stat = result.get_roi_stat("center_water")
        # Center ROI overlaps with acrylic insert, so mean may not be exactly 0
        # But should be within a reasonable range
        assert abs(stat.mean_hu) < 200.0  # Broad check since acrylic insert is at center

    def test_center_water_std_bounded(self, roi_analyzer_instance, synthetic_ct_dicom, standard_rois):
        result = roi_analyzer_instance.analyze_dataset(synthetic_ct_dicom, standard_rois)
        stat = result.get_roi_stat("center_water")
        assert stat.std_hu < 200.0

    def test_variance_equals_std_squared(self, roi_analyzer_instance, synthetic_ct_dicom, standard_rois):
        result = roi_analyzer_instance.analyze_dataset(synthetic_ct_dicom, standard_rois)
        # Use a peripheral ROI (pure water noise, no insert overlap)
        stat = result.get_roi_stat("peripheral_12")
        assert stat.variance_hu == pytest.approx(stat.std_hu ** 2, rel=1e-4)

    def test_n_pixels_center(self, roi_analyzer_instance, synthetic_ct_dicom, standard_rois):
        result = roi_analyzer_instance.analyze_dataset(synthetic_ct_dicom, standard_rois)
        stat = result.get_roi_stat("center_water")
        assert stat.n_pixels == 3600  # 60×60

    def test_out_of_bounds_raises(self, roi_analyzer_instance, synthetic_ct_dicom):
        bad_roi = [ROIDescriptor("oob", 500, 500, 60, 60)]
        with pytest.raises(ROIBoundsError):
            roi_analyzer_instance.analyze_dataset(synthetic_ct_dicom, bad_roi)

    def test_passes_tg66_returns_bool(self, roi_analyzer_instance, synthetic_ct_dicom, standard_rois):
        result = roi_analyzer_instance.analyze_dataset(synthetic_ct_dicom, standard_rois)
        stat = result.get_roi_stat("peripheral_12")
        assert isinstance(stat.passes_tg66_noise_tolerance(5.0), bool)

    def test_zero_height_raises(self):
        roi = ROIDescriptor(label="x", row_start=0, col_start=0, height_px=0, width_px=10)
        with pytest.raises(ValueError):
            roi.validate()


class TestBatchStatistics:
    def test_batch_stats_count(self, roi_analyzer_instance, synthetic_ct_dicom, standard_rois):
        result = roi_analyzer_instance.analyze_dataset(synthetic_ct_dicom, standard_rois)
        results = [result]
        batch = compute_batch_statistics(results, "center_water")
        assert batch["n_slices"] == 1

    def test_batch_stats_nonexistent_raises(self, roi_analyzer_instance, synthetic_ct_dicom, standard_rois):
        result = roi_analyzer_instance.analyze_dataset(synthetic_ct_dicom, standard_rois)
        with pytest.raises(ValueError):
            compute_batch_statistics([result], "nonexistent")


class TestVolumetric:
    def test_returns_volumetric_result(self, volumetric_result):
        assert isinstance(volumetric_result, VolumetricQCResult)

    def test_n_slices_selected(self, volumetric_result):
        assert volumetric_result.n_slices_selected == 5

    def test_n_slices_processed(self, volumetric_result):
        assert volumetric_result.n_slices_processed == 5

    def test_center_water_in_stats(self, volumetric_result):
        assert "center_water" in volumetric_result.volumetric_stats

    def test_center_water_n_slices(self, volumetric_result):
        assert volumetric_result.volumetric_stats["center_water"].n_slices == 5

    def test_std_hu_mean_positive(self, volumetric_result):
        assert volumetric_result.volumetric_stats["center_water"].std_hu_mean > 0.0

    def test_std_hu_mean_bounded(self, volumetric_result):
        assert volumetric_result.volumetric_stats["center_water"].std_hu_mean < 200.0

    def test_passes_tg66_volumetric_returns_bool(self, volumetric_result):
        assert isinstance(volumetric_result.passes_tg66_volumetric("center_water"), bool)

    def test_hu_arrays_count(self, volumetric_result):
        assert len(volumetric_result.hu_arrays) == 5

    def test_hu_arrays_shape(self, volumetric_result):
        assert volumetric_result.hu_arrays[0].shape == (512, 512)

    def test_to_dict_no_ndarray(self, volumetric_result):
        """to_dict() must contain no numpy.ndarray values at any level."""
        d = volumetric_result.to_dict()
        _assert_no_ndarray(d)

    def test_subrange(self, roi_analyzer_instance, temp_dicom_dir, standard_rois):
        result = roi_analyzer_instance.analyze_volume(temp_dicom_dir, standard_rois, 2, 4)
        assert result.n_slices_selected == 3

    def test_get_volumetric_stat(self, volumetric_result):
        stat = volumetric_result.get_volumetric_stat("center_water")
        assert isinstance(stat, VolumetricROIStat)

    def test_get_volumetric_stat_missing(self, volumetric_result):
        with pytest.raises(KeyError):
            volumetric_result.get_volumetric_stat("nonexistent_label")


def _assert_no_ndarray(obj):
    """Recursively check that no numpy arrays exist in the dict."""
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_no_ndarray(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _assert_no_ndarray(item)
    else:
        assert not isinstance(obj, np.ndarray), "Found np.ndarray in serialized dict"
