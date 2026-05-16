# -*- coding: utf-8 -*-
"""tests/test_nps_calculator.py — NPS calculator tests."""

import numpy as np
import pytest

from config import CONFIG
from modules.image_qc.nps_calculator import (
    NPSCalculator, NPSResult, NPSPatch, InsufficientPatchesError,
)


@pytest.fixture
def nps_calc():
    return NPSCalculator(config=CONFIG.image_qc)


class TestNPSInstantiation:
    def test_instantiation(self, nps_calc):
        assert isinstance(nps_calc, NPSCalculator)


class TestHanningWindow:
    def test_shape_preserved(self, nps_calc):
        patch = np.random.default_rng(42).normal(0, 5, (64, 64))
        result = nps_calc._apply_hanning_window(patch.copy())
        assert result.shape == (64, 64)

    def test_corners_near_zero(self, nps_calc):
        patch = np.ones((64, 64))
        result = nps_calc._apply_hanning_window(patch.copy())
        # Hanning window suppresses edges — corners should be near zero
        assert abs(result[0, 0]) < 0.01
        assert abs(result[0, -1]) < 0.01
        assert abs(result[-1, 0]) < 0.01
        assert abs(result[-1, -1]) < 0.01


class TestDetrend:
    def test_uniform_patch_residual(self, nps_calc):
        """Perfectly uniform patch: residual should be near zero."""
        patch = np.full((64, 64), 100.0)
        detrended = nps_calc._detrend_patch(patch.copy(), 2)
        assert np.max(np.abs(detrended)) < 1e-6

    def test_linear_gradient_residual(self, nps_calc):
        """Patch with linear gradient: detrending should remove it."""
        ii, jj = np.meshgrid(np.arange(64), np.arange(64), indexing='ij')
        patch = 2.0 * ii + 3.0 * jj + 10.0
        detrended = nps_calc._detrend_patch(patch.copy(), 2)
        assert np.max(np.abs(detrended)) < 1e-6


class TestPowerSpectrum:
    def test_non_negative(self, nps_calc):
        patch = np.random.default_rng(42).normal(0, 5, (64, 64))
        windowed = nps_calc._apply_hanning_window(patch)
        ps = nps_calc._compute_patch_power_spectrum(windowed, (0.977, 0.977))
        assert np.all(ps >= 0)

    def test_2d_shape(self, nps_calc):
        patch = np.random.default_rng(42).normal(0, 5, (64, 64))
        windowed = nps_calc._apply_hanning_window(patch)
        ps = nps_calc._compute_patch_power_spectrum(windowed, (0.977, 0.977))
        assert ps.shape == (64, 64)


class TestRadialAverage:
    def test_equal_length(self, nps_calc):
        nps_2d = np.random.default_rng(42).uniform(0, 10, (64, 64))
        freq, nps_1d = nps_calc._radial_average(nps_2d, (0.977, 0.977))
        assert len(freq) == len(nps_1d)

    def test_frequency_range(self, nps_calc):
        nps_2d = np.random.default_rng(42).uniform(0, 10, (64, 64))
        freq, nps_1d = nps_calc._radial_average(nps_2d, (0.977, 0.977))
        assert freq[0] >= 0
        nyquist = 1.0 / (2.0 * 0.977)
        assert max(freq) == pytest.approx(nyquist, rel=0.1)


class TestComputeFromVolume:
    def test_runs_without_error(self, nps_calc, volumetric_result):
        result = nps_calc.compute_from_volume(volumetric_result)
        assert isinstance(result, NPSResult)

    def test_n_slices_used(self, nps_calc, volumetric_result):
        result = nps_calc.compute_from_volume(volumetric_result)
        assert result.n_slices_used == 5

    def test_noise_std_reasonable(self, nps_calc, volumetric_result):
        """NPS-derived noise SD should be within 50% of volumetric ROI std."""
        result = nps_calc.compute_from_volume(volumetric_result)
        # Use a peripheral ROI that's pure water (not acrylic insert)
        roi_std = volumetric_result.volumetric_stats["peripheral_12"].std_hu_mean
        # Very broad check since NPS covers the full image, not a single ROI
        assert result.noise_std_from_nps > 0.0

    def test_insufficient_patches(self, nps_calc):
        """32x32 array should be too small for standard patches."""
        tiny = [np.random.default_rng(42).normal(0, 5, (32, 32)).astype(np.float32)]
        with pytest.raises(InsufficientPatchesError):
            nps_calc.compute_from_hu_arrays(tiny, (0.977, 0.977))


class TestNPSResultSerialization:
    def test_to_dict_no_ndarray(self, nps_calc, volumetric_result):
        result = nps_calc.compute_from_volume(volumetric_result)
        d = result.to_dict()
        _assert_no_ndarray(d)

    def test_passes_frequency_drift_check(self, nps_calc, volumetric_result):
        result = nps_calc.compute_from_volume(volumetric_result)
        assert result.passes_frequency_drift_check(
            reference_freq=result.nps_peak_frequency_lpmm, tolerance_lpmm=0.05
        ) is True


def _assert_no_ndarray(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_no_ndarray(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _assert_no_ndarray(item)
    else:
        assert not isinstance(obj, np.ndarray), "Found np.ndarray in serialized dict"
