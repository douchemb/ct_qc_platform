# -*- coding: utf-8 -*-
"""
tests/test_dw_calculator.py — Tests for Water-Equivalent Diameter Calculator.

Validates AAPM TG-220 Eq. 1 and 2 implementations for both
axial CT slices and CT localizer radiograph methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.dosimetry.dw_calculator import (
    DwCalculator,
    DwSliceResult,
    DwSeriesResult,
    DwLocalizerResult,
    BodySegmentationError,
    InsufficientSlicesError,
)
from core.dicom_loader import DicomMetadata


class TestDwCalculatorInstantiation:
    """Test that DwCalculator instantiates correctly."""

    def test_instantiates_without_error(self, dw_calculator_instance):
        """DwCalculator should instantiate without error."""
        assert dw_calculator_instance is not None
        assert isinstance(dw_calculator_instance, DwCalculator)


class TestDwAxialMethod:
    """Test D_w computation from axial CT slices."""

    def test_compute_from_single_slice_runs(
        self, dw_calculator_instance, volumetric_result
    ):
        """compute_from_single_slice should run without error on synthetic CT."""
        hu_array = volumetric_result.hu_arrays[0]
        metadata = volumetric_result.slice_results[0].metadata
        result = dw_calculator_instance.compute_from_single_slice(
            hu_array=hu_array,
            pixel_spacing_mm=(0.977, 0.977),
            metadata=metadata,
        )
        assert isinstance(result, DwSliceResult)
        assert result.dw_cm > 0.0

    def test_water_eq_area_circular_mask(self, dw_calculator_instance):
        """A_w of a circular water region should match π × r² in cm²."""
        # Create a 512×512 array of HU = -1000 (air) with a circle of HU=0 (water)
        hu_array = np.full((512, 512), -1000.0, dtype=np.float32)
        yy, xx = np.ogrid[:512, :512]
        radius_px = 120
        circle_mask = ((yy - 256) ** 2 + (xx - 256) ** 2) <= radius_px ** 2
        hu_array[circle_mask] = 0.0  # water

        # Body mask is the circle itself
        body_mask = circle_mask.astype(bool)

        # Spacing: 0.977 mm
        spacing = (0.977, 0.977)

        a_w = dw_calculator_instance._compute_water_eq_area_axial(
            hu_array, body_mask, spacing
        )

        # Expected: π × (120 × 0.977/10)² = π × 11.724² ≈ 431.8 cm²
        r_cm = radius_px * spacing[0] / 10.0
        expected_a_w = np.pi * r_cm ** 2
        assert a_w == pytest.approx(expected_a_w, rel=0.05), (
            "A_w=%.2f cm², expected≈%.2f cm²" % (a_w, expected_a_w)
        )

    def test_water_eq_area_single_water_pixel(self, dw_calculator_instance):
        """Single pixel of HU=0 with area 1 cm² → A_w = 1.0 cm²."""
        hu_array = np.array([[0.0]], dtype=np.float32)
        body_mask = np.array([[True]])
        # Spacing that gives 1 cm² per pixel: 10 mm × 10 mm
        spacing = (10.0, 10.0)

        a_w = dw_calculator_instance._compute_water_eq_area_axial(
            hu_array, body_mask, spacing
        )
        assert a_w == pytest.approx(1.0, abs=1e-6)

    def test_water_eq_area_single_air_pixel(self, dw_calculator_instance):
        """Single pixel of HU=-1000 with area 1 cm² → A_w = 0.0 cm²."""
        hu_array = np.array([[-1000.0]], dtype=np.float32)
        body_mask = np.array([[True]])
        spacing = (10.0, 10.0)

        a_w = dw_calculator_instance._compute_water_eq_area_axial(
            hu_array, body_mask, spacing
        )
        assert a_w == pytest.approx(0.0, abs=1e-6)

    def test_water_eq_area_single_bone_pixel(self, dw_calculator_instance):
        """Single pixel of HU=1000 with area 1 cm² → A_w = 2.0 cm²."""
        hu_array = np.array([[1000.0]], dtype=np.float32)
        body_mask = np.array([[True]])
        spacing = (10.0, 10.0)

        a_w = dw_calculator_instance._compute_water_eq_area_axial(
            hu_array, body_mask, spacing
        )
        assert a_w == pytest.approx(2.0, abs=1e-6)

    def test_dw_from_area_pi(self, dw_calculator_instance):
        """D_w from area π should be 2.0 (circle of area π has diameter 2)."""
        dw = dw_calculator_instance._dw_from_area(np.pi)
        assert dw == pytest.approx(2.0, abs=1e-6)

    def test_dw_from_area_zero(self, dw_calculator_instance):
        """D_w from area 0 should be 0.0."""
        dw = dw_calculator_instance._dw_from_area(0.0)
        assert dw == 0.0


class TestDwFromVolume:
    """Test compute_from_volume integration with VolumetricQCResult."""

    def test_compute_from_volume_returns_series_result(
        self, dw_calculator_instance, volumetric_result
    ):
        """compute_from_volume should return a DwSeriesResult."""
        result = dw_calculator_instance.compute_from_volume(volumetric_result)
        assert isinstance(result, DwSeriesResult)

    def test_dw_series_n_slices(self, dw_series_result):
        """Series result should have 5 slices."""
        assert dw_series_result.n_slices == 5

    def test_dw_series_mean_positive(self, dw_series_result):
        """Mean D_w should be positive."""
        assert dw_series_result.dw_mean_cm > 0.0

    def test_dw_series_mean_upper_bound(self, dw_series_result):
        """Mean D_w should be less than 60 cm (sanity upper bound for synthetic phantom)."""
        assert dw_series_result.dw_mean_cm < 60.0

    def test_slice_result_to_dict(self, dw_series_result):
        """DwSliceResult.to_dict() should return a serializable dict."""
        d = dw_series_result.slice_results[0].to_dict()
        assert isinstance(d, dict)
        assert "dw_cm" in d
        assert "water_eq_area_cm2" in d


class TestDwLocalizerMethod:
    """Test D_w computation from localizer radiograph."""

    def test_compute_from_localizer_runs(
        self, dw_calculator_instance, parsed_localizer_data
    ):
        """compute_from_localizer should return DwLocalizerResult."""
        result = dw_calculator_instance.compute_from_localizer(parsed_localizer_data)
        assert isinstance(result, DwLocalizerResult)

    def test_localizer_dw_per_row_shape(
        self, dw_calculator_instance, parsed_localizer_data
    ):
        """D_w per row should have 512 elements (one per localizer row)."""
        result = dw_calculator_instance.compute_from_localizer(parsed_localizer_data)
        assert result.dw_per_row_cm.shape[0] == 512

    def test_localizer_dw_non_negative(
        self, dw_calculator_instance, parsed_localizer_data
    ):
        """All D_w values should be non-negative."""
        result = dw_calculator_instance.compute_from_localizer(parsed_localizer_data)
        assert np.all(result.dw_per_row_cm >= 0.0)

    def test_localizer_dw_mean_physical_range(
        self, dw_calculator_instance, parsed_localizer_data
    ):
        """Mean D_w from localizer should be between 5.0 and 35.0 cm."""
        result = dw_calculator_instance.compute_from_localizer(parsed_localizer_data)
        assert 5.0 <= result.dw_mean_cm <= 35.0, (
            "Mean D_w=%.2f cm outside expected range [5, 35]" % result.dw_mean_cm
        )

    def test_localizer_arrays_equal_length(
        self, dw_calculator_instance, parsed_localizer_data
    ):
        """water_eq_area and dw_per_row should have equal length."""
        result = dw_calculator_instance.compute_from_localizer(parsed_localizer_data)
        assert len(result.water_eq_area_per_row_cm2) == len(result.dw_per_row_cm)

    def test_localizer_to_dict_no_ndarray(
        self, dw_calculator_instance, parsed_localizer_data
    ):
        """DwLocalizerResult.to_dict() should contain no numpy.ndarray."""
        result = dw_calculator_instance.compute_from_localizer(parsed_localizer_data)
        d = result.to_dict()

        def _check_no_ndarray(obj, path=""):
            if isinstance(obj, np.ndarray):
                pytest.fail("Found numpy.ndarray at path: %s" % path)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _check_no_ndarray(v, path="%s.%s" % (path, k))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _check_no_ndarray(v, path="%s[%d]" % (path, i))

        _check_no_ndarray(d, "root")
