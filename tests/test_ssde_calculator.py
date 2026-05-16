# -*- coding: utf-8 -*-
"""
tests/test_ssde_calculator.py — Tests for SSDE Calculator.

Validates AAPM Report 204 f-factor lookup, SSDE computation,
DLP calculation, and effective dose estimation.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import CONFIG
from modules.dosimetry.ssde_calculator import (
    SSDECalculator,
    SSDESliceResult,
    SSDESeriesResult,
    MissingCTDIvolError,
)
from modules.dosimetry.dw_calculator import DwSliceResult, DwSeriesResult


# ═══════════════════════════════════════════════════════════════════
# Helper fixtures for hand-calculated test cases
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def dw_slice_20cm():
    """DwSliceResult with D_w = 20.0 cm (exact table value)."""
    return DwSliceResult(
        slice_position_mm=0.0,
        instance_number=1,
        acquisition_date="20240115",
        water_eq_area_cm2=np.pi * 10.0 ** 2,  # area for D_w=20 cm
        dw_cm=20.0,
        n_body_pixels=10000,
        n_total_pixels=262144,
        body_fraction=0.038,
        segmentation_method="largest_contour",
    )


@pytest.fixture
def dw_slice_32cm():
    """DwSliceResult with D_w = 32.0 cm (exact table value, f=0.900)."""
    return DwSliceResult(
        slice_position_mm=0.0,
        instance_number=1,
        acquisition_date="20240115",
        water_eq_area_cm2=np.pi * 16.0 ** 2,
        dw_cm=32.0,
        n_body_pixels=20000,
        n_total_pixels=262144,
        body_fraction=0.076,
        segmentation_method="largest_contour",
    )


@pytest.fixture
def dw_slice_10cm():
    """DwSliceResult with D_w = 10.0 cm (smallest table value, f=1.528)."""
    return DwSliceResult(
        slice_position_mm=0.0,
        instance_number=1,
        acquisition_date="20240115",
        water_eq_area_cm2=np.pi * 5.0 ** 2,
        dw_cm=10.0,
        n_body_pixels=5000,
        n_total_pixels=262144,
        body_fraction=0.019,
        segmentation_method="largest_contour",
    )


class TestFFactorLookup:
    """Test f-factor lookup from AAPM Report 204 Table A."""

    def test_f_factor_10cm(self, ssde_calculator_instance):
        """f(10.0) = 1.528 (exact table lookup)."""
        f, method, in_range = ssde_calculator_instance._lookup_f_factor(10.0)
        assert f == pytest.approx(1.528, abs=0.001)
        assert method == "table_lookup"
        assert in_range is True

    def test_f_factor_20cm(self, ssde_calculator_instance):
        """f(20.0) = 1.202 (exact table lookup)."""
        f, method, in_range = ssde_calculator_instance._lookup_f_factor(20.0)
        assert f == pytest.approx(1.202, abs=0.001)
        assert method == "table_lookup"
        assert in_range is True

    def test_f_factor_40cm(self, ssde_calculator_instance):
        """f(40.0) = 0.742 (exact table lookup)."""
        f, method, in_range = ssde_calculator_instance._lookup_f_factor(40.0)
        assert f == pytest.approx(0.742, abs=0.001)
        assert method == "table_lookup"
        assert in_range is True

    def test_f_factor_21cm_interpolated(self, ssde_calculator_instance):
        """f(21.0) should be interpolated between f(20)=1.202 and f(22)=1.145."""
        f, method, in_range = ssde_calculator_instance._lookup_f_factor(21.0)
        assert method == "interpolated"
        assert in_range is True
        # Midpoint between 1.202 and 1.145
        assert f == pytest.approx(1.1735, abs=0.005)

    def test_f_factor_5cm_extrapolated(self, ssde_calculator_instance):
        """f(5.0) should be extrapolated below table range."""
        f, method, in_range = ssde_calculator_instance._lookup_f_factor(5.0)
        assert method == "extrapolated"
        assert in_range is False

    def test_f_factor_45cm_extrapolated(self, ssde_calculator_instance):
        """f(45.0) should be extrapolated above table range."""
        f, method, in_range = ssde_calculator_instance._lookup_f_factor(45.0)
        assert method == "extrapolated"
        assert in_range is False


class TestSSDEComputation:
    """Test SSDE computation against hand-calculated values."""

    def test_ssde_dw20_ctdi10(self, ssde_calculator_instance, dw_slice_20cm):
        """SSDE(D_w=20, CTDIvol=10) = 1.202 × 10.0 = 12.02 mGy."""
        result = ssde_calculator_instance.compute_slice_ssde(dw_slice_20cm, 10.0)
        assert result.ssde_mgy == pytest.approx(12.02, abs=0.01)

    def test_ssde_dw32_ctdi15(self, ssde_calculator_instance, dw_slice_32cm):
        """SSDE(D_w=32, CTDIvol=15) = 0.900 × 15.0 = 13.50 mGy."""
        result = ssde_calculator_instance.compute_slice_ssde(dw_slice_32cm, 15.0)
        assert result.ssde_mgy == pytest.approx(13.50, abs=0.01)

    def test_ssde_dw10_ctdi5(self, ssde_calculator_instance, dw_slice_10cm):
        """SSDE(D_w=10, CTDIvol=5) = 1.528 × 5.0 = 7.64 mGy."""
        result = ssde_calculator_instance.compute_slice_ssde(dw_slice_10cm, 5.0)
        assert result.ssde_mgy == pytest.approx(7.64, abs=0.01)


class TestSSDESeriesComputation:
    """Test series-level SSDE computation."""

    def test_compute_series_ssde_runs(
        self, ssde_calculator_instance, dw_series_result, volumetric_result
    ):
        """compute_series_ssde should return SSDESeriesResult."""
        metadata = volumetric_result.slice_results[0].metadata
        result = ssde_calculator_instance.compute_series_ssde(
            dw_series_result, ctdi_vol_mgy=12.5, metadata=metadata,
        )
        assert isinstance(result, SSDESeriesResult)

    def test_series_ctdi_vol(
        self, ssde_calculator_instance, dw_series_result, volumetric_result
    ):
        """Series CTDIvol should match input."""
        metadata = volumetric_result.slice_results[0].metadata
        result = ssde_calculator_instance.compute_series_ssde(
            dw_series_result, ctdi_vol_mgy=12.5, metadata=metadata,
        )
        assert result.ctdi_vol_mgy == 12.5

    def test_series_ssde_mean_positive(
        self, ssde_calculator_instance, dw_series_result, volumetric_result
    ):
        """Series mean SSDE should be positive."""
        metadata = volumetric_result.slice_results[0].metadata
        result = ssde_calculator_instance.compute_series_ssde(
            dw_series_result, ctdi_vol_mgy=12.5, metadata=metadata,
        )
        assert result.ssde_mean_mgy > 0.0

    def test_series_n_slices(
        self, ssde_calculator_instance, dw_series_result, volumetric_result
    ):
        """Series should have 5 slices."""
        metadata = volumetric_result.slice_results[0].metadata
        result = ssde_calculator_instance.compute_series_ssde(
            dw_series_result, ctdi_vol_mgy=12.5, metadata=metadata,
        )
        assert result.n_slices == 5


class TestDLPAndEffectiveDose:
    """Test DLP and effective dose calculations."""

    def test_compute_dlp(self, ssde_calculator_instance):
        """DLP = SSDE_mean × scan_length = 10.0 × 30.0 = 300.0."""
        dlp = ssde_calculator_instance._compute_dlp(
            ssde_mean_mgy=10.0, scan_length_cm=30.0
        )
        assert dlp == 300.0

    def test_effective_dose_abdomen(self, ssde_calculator_instance):
        """E(abdomen) = 0.015 × 300.0 = 4.5 mSv."""
        eff = ssde_calculator_instance._estimate_effective_dose(
            dlp_mgy_cm=300.0, body_region="abdomen"
        )
        assert eff == pytest.approx(4.5, abs=0.1)

    def test_effective_dose_head(self, ssde_calculator_instance):
        """E(head) = 0.0023 × 300.0 = 0.69 mSv."""
        eff = ssde_calculator_instance._estimate_effective_dose(
            dlp_mgy_cm=300.0, body_region="head"
        )
        assert eff == pytest.approx(0.69, abs=0.01)

    def test_effective_dose_unknown_region_no_raise(self, ssde_calculator_instance):
        """Unknown body region should not raise — defaults to abdomen k=0.015."""
        eff = ssde_calculator_instance._estimate_effective_dose(
            dlp_mgy_cm=100.0, body_region="unknown_region"
        )
        # k defaults to 0.015 (abdomen): 0.015 × 100 = 1.5 mSv
        assert eff == pytest.approx(1.5, abs=0.1)


class TestCompliance:
    """Test DRL compliance checking."""

    def test_passes_drl_below(
        self, ssde_calculator_instance, dw_series_result, volumetric_result
    ):
        """CTDIvol=12.5 mGy should pass DRL of 25 mGy."""
        metadata = volumetric_result.slice_results[0].metadata
        result = ssde_calculator_instance.compute_series_ssde(
            dw_series_result, ctdi_vol_mgy=12.5, metadata=metadata,
        )
        assert result.passes_diagnostic_reference_level(25.0) is True

    def test_fails_drl_above(
        self, ssde_calculator_instance, dw_series_result, volumetric_result
    ):
        """CTDIvol=30.0 mGy should fail DRL of 25 mGy."""
        metadata = volumetric_result.slice_results[0].metadata
        result = ssde_calculator_instance.compute_series_ssde(
            dw_series_result, ctdi_vol_mgy=30.0, metadata=metadata,
        )
        assert result.passes_diagnostic_reference_level(25.0) is False


class TestSSDESerialization:
    """Test serialization of SSDE results."""

    def test_to_dict_returns_dict(
        self, ssde_calculator_instance, dw_series_result, volumetric_result
    ):
        """SSDESeriesResult.to_dict() should return a JSON-serializable dict."""
        metadata = volumetric_result.slice_results[0].metadata
        result = ssde_calculator_instance.compute_series_ssde(
            dw_series_result, ctdi_vol_mgy=12.5, metadata=metadata,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ssde_mean_mgy" in d
        assert "slice_results" in d
        assert "dlp_mgy_cm" in d
