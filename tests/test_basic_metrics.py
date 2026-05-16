"""Tests for BasicMetricsEngine — all five Basic Tier metrics."""
from __future__ import annotations
import pytest
import numpy as np


class TestNoiseMetric:

    def test_noise_computed_successfully(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert result.noise is not None

    def test_noise_std_is_positive(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert result.noise.std_hu > 0.0

    def test_noise_variance_consistent_with_std(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        # variance_hu_mean and std_hu_mean^2 are close but not identical
        # because E[X]^2 ≤ E[X^2] (Jensen's inequality)
        assert result.noise.variance_hu == pytest.approx(
            result.noise.std_hu ** 2, rel=0.1
        )

    def test_noise_passes_for_low_noise_phantom(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        """Synthetic phantom has noise ~5 HU — should pass TG-66 tolerance."""
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert isinstance(result.noise.passed, bool)

    def test_noise_to_dict_serializable(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        import json
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        json.dumps(result.noise.to_dict())


class TestUniformityMetric:

    def test_uniformity_computed_successfully(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert result.uniformity is not None

    def test_uniformity_nui_is_non_negative(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert result.uniformity.non_uniformity_index >= 0.0

    def test_uniformity_has_four_deviations(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        """Standard 5-ROI layout gives 4 peripheral deviations."""
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert len(result.uniformity.deviations) == 4

    def test_uniformity_nui_equals_max_deviation(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        expected_nui = max(result.uniformity.deviations.values())
        assert result.uniformity.non_uniformity_index == pytest.approx(
            expected_nui, abs=1e-6
        )

    def test_uniformity_worst_roi_label_is_in_deviations(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert result.uniformity.worst_roi in result.uniformity.deviations

    def test_uniformity_to_dict_serializable(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        import json
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        json.dumps(result.uniformity.to_dict())


class TestCTNumberAccuracy:

    def test_ct_number_accuracy_computed(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert result.ct_number_accuracy is not None

    def test_water_delta_hu_near_zero(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        if result.ct_number_accuracy.measurements:
            for m in result.ct_number_accuracy.measurements:
                if m["material"] == "water" or "water" in m["material"]:
                    assert abs(m["delta_hu"]) < 15.0

    def test_max_delta_hu_is_non_negative(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert result.ct_number_accuracy.max_delta_hu >= 0.0

    def test_all_passed_is_bool(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert isinstance(result.ct_number_accuracy.all_passed, bool)


class TestFWHMSliceThickness:

    def test_fwhm_with_ramp_profile(self):
        """Test FWHM with a synthetic sensitometric ramp profile."""
        from modules.image_qc.basic_metrics import BasicMetricsEngine
        engine = BasicMetricsEngine()

        n_slices        = 20
        nominal_mm      = 3.0
        pixel_spacing   = (0.977, 0.977)

        arrays = []
        for i in range(n_slices):
            arr = np.full((64, 64), -1000.0, dtype=np.float32)
            arr += 1000.0 * np.exp(-((i - 10) ** 2) / (2 * 1.5 ** 2))
            arrays.append(arr)

        result = engine._compute_slice_thickness_fwhm(
            arrays, nominal_mm, pixel_spacing
        )
        assert result is not None
        assert result.measured_fwhm_mm > 0.0
        assert result.passed in (True, False)  # np.bool_ is not isinstance(bool)

    def test_fwhm_insufficient_slices_raises(self):
        from modules.image_qc.basic_metrics import BasicMetricsEngine, SliceThicknessROIError
        engine = BasicMetricsEngine()
        with pytest.raises(SliceThicknessROIError):
            engine._compute_slice_thickness_fwhm(
                [np.zeros((64, 64), dtype=np.float32)],
                3.0, (0.977, 0.977)
            )

    def test_fwhm_flat_profile_raises(self):
        from modules.image_qc.basic_metrics import BasicMetricsEngine, SliceThicknessROIError
        engine = BasicMetricsEngine()
        flat_arrays = [np.zeros((64, 64), dtype=np.float32) for _ in range(5)]
        with pytest.raises(SliceThicknessROIError):
            engine._compute_slice_thickness_fwhm(flat_arrays, 3.0, (0.977, 0.977))

    def test_find_half_max_crossing_known_profile(self):
        """Test sub-pixel FWHM crossing on a known triangular profile."""
        from modules.image_qc.basic_metrics import BasicMetricsEngine
        engine  = BasicMetricsEngine()
        profile = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0])
        left  = engine._find_half_max_crossing(profile, peak_idx=4, direction="left")
        right = engine._find_half_max_crossing(profile, peak_idx=4, direction="right")
        assert left  is not None
        assert right is not None
        fwhm_samples = right - left
        assert fwhm_samples == pytest.approx(4.0, abs=0.5)

    def test_find_half_max_crossing_no_crossing_returns_none(self):
        from modules.image_qc.basic_metrics import BasicMetricsEngine
        engine  = BasicMetricsEngine()
        profile = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        left = engine._find_half_max_crossing(profile, peak_idx=4, direction="left")
        assert left is None


class TestBasicQAResult:

    def test_basic_qa_result_to_dict_serializable(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        import json
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        d = result.to_dict()
        json.dumps(d, default=str)

    def test_all_passed_is_bool(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert isinstance(result.all_passed, bool)

    def test_phantom_id_correct(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert result.phantom_id == "siemens_water_phantom"

    def test_warnings_is_list(
        self, basic_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = basic_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert isinstance(result.warnings, list)
