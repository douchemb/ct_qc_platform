# -*- coding: utf-8 -*-
"""tests/test_mtf_calculator.py — MTF calculator tests."""

import numpy as np
import pytest

from config import CONFIG
from modules.image_qc.mtf_calculator import MTFCalculator, MTFResult


@pytest.fixture
def mtf_calc():
    return MTFCalculator(config=CONFIG.image_qc)


class TestMTFInstantiation:
    def test_instantiation(self, mtf_calc):
        assert isinstance(mtf_calc, MTFCalculator)


class TestLSFtoMTF:
    def test_dirac_delta_flat_mtf(self, mtf_calc):
        """Dirac delta LSF => flat MTF (all values near 1.0)."""
        lsf = np.zeros(128)
        lsf[64] = 1.0
        axis = np.arange(128) * 0.25  # 0.25 mm spacing

        freq, mtf = mtf_calc._lsf_to_mtf(lsf, axis)
        assert len(freq) > 0
        # All MTF values should be close to 1.0 (within 5%)
        assert np.all(np.abs(mtf - 1.0) < 0.05)

    def test_boxcar_sinc_mtf(self, mtf_calc):
        """Boxcar LSF of width W => first zero of sinc at f=1/(W*dx)."""
        dx = 0.25  # mm
        W = 8      # pixels
        lsf = np.zeros(128)
        lsf[60:60 + W] = 1.0
        axis = np.arange(128) * dx

        freq, mtf = mtf_calc._lsf_to_mtf(lsf, axis)

        # First zero of sinc at f = 1/(W*dx) = 1/(8*0.25) = 0.5 lp/mm
        expected_zero = 1.0 / (W * dx)

        # Find first zero crossing (MTF < 0.05)
        first_zero_idx = np.where(mtf < 0.05)[0]
        if len(first_zero_idx) > 0:
            measured_zero = freq[first_zero_idx[0]]
            assert measured_zero == pytest.approx(expected_zero, rel=0.15)


class TestFindMTFAtValue:
    def test_linear_mtf(self, mtf_calc):
        """Linear MTF: MTF(f) = 1 - f/f_max -> MTF50 at f=0.5*f_max."""
        f_max = 1.0
        freq = np.linspace(0, f_max, 100)
        mtf = 1.0 - freq / f_max

        result = mtf_calc._find_mtf_at_value(freq, mtf, 0.5)
        assert result == pytest.approx(0.5, abs=0.02)

    def test_never_reaches_target(self, mtf_calc):
        """If MTF min > target, should return NaN."""
        freq = np.linspace(0, 1.0, 100)
        mtf = np.full(100, 0.6)  # Never drops below 0.6

        result = mtf_calc._find_mtf_at_value(freq, mtf, 0.5)
        assert np.isnan(result)


class TestDifferentiateESF:
    def test_heaviside_gives_peak(self, mtf_calc):
        """Heaviside step ESF => LSF with clear peak at the step location."""
        n = 128
        esf = np.zeros(n)
        esf[n // 2:] = 100.0  # Step at midpoint
        esf_axis = np.arange(n) * 0.25

        lsf, lsf_axis = mtf_calc._differentiate_esf_to_lsf(esf, esf_axis)

        # Peak should be near the midpoint
        peak_idx = np.argmax(np.abs(lsf))
        # Peak position should be near index n//2
        assert abs(peak_idx - n // 2) < 5


class TestLocateBeadCentroid:
    def test_gaussian_bead(self, mtf_calc):
        """Gaussian bead at known center: centroid should be within 0.5 px."""
        center_r, center_c = 32.3, 31.7
        roi = np.zeros((64, 64))
        ii, jj = np.meshgrid(np.arange(64), np.arange(64), indexing='ij')
        roi = 1000.0 * np.exp(-((ii - center_r) ** 2 + (jj - center_c) ** 2) / (2 * 3 ** 2))

        row_c, col_c = mtf_calc._locate_bead_centroid(roi, 0.5)
        assert abs(row_c - center_r) < 0.5
        assert abs(col_c - center_c) < 0.5


class TestMTFResultMethods:
    def test_passes_resolution_check_true(self):
        result = MTFResult(
            series_description="test", acquisition_date="20240101",
            pixel_spacing_mm=0.5, method="edge",
            freq_axis_lpmm=np.array([0, 0.5, 1.0]),
            mtf_values=np.array([1.0, 0.5, 0.1]),
            mtf_50_lpmm=0.5, mtf_10_lpmm=1.0,
            mtf_at_nyquist=0.1,
            lsf=np.array([0, 1, 0]),
            lsf_axis_mm=np.array([0, 0.5, 1.0]),
        )
        assert result.passes_resolution_check(0.4) is True

    def test_passes_resolution_check_false(self):
        result = MTFResult(
            series_description="test", acquisition_date="20240101",
            pixel_spacing_mm=0.5, method="edge",
            freq_axis_lpmm=np.array([0, 0.5, 1.0]),
            mtf_values=np.array([1.0, 0.5, 0.1]),
            mtf_50_lpmm=0.3, mtf_10_lpmm=0.8,
            mtf_at_nyquist=0.1,
            lsf=np.array([0, 1, 0]),
            lsf_axis_mm=np.array([0, 0.5, 1.0]),
        )
        assert result.passes_resolution_check(0.4) is False

    def test_to_dict_no_ndarray(self):
        result = MTFResult(
            series_description="test", acquisition_date="20240101",
            pixel_spacing_mm=0.5, method="edge",
            freq_axis_lpmm=np.array([0.0, 0.5]),
            mtf_values=np.array([1.0, 0.5]),
            mtf_50_lpmm=0.5, mtf_10_lpmm=1.0, mtf_at_nyquist=0.1,
            lsf=np.array([0.0, 1.0]),
            lsf_axis_mm=np.array([0.0, 0.5]),
        )
        d = result.to_dict()
        _assert_no_ndarray(d)


def _assert_no_ndarray(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_no_ndarray(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _assert_no_ndarray(item)
    else:
        assert not isinstance(obj, np.ndarray)
