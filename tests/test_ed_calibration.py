# -*- coding: utf-8 -*-
"""tests/test_ed_calibration.py — ED calibration analyzer tests."""

import csv
import numpy as np
import pytest

from config import CONFIG
from modules.image_qc.ed_calibration import (
    EDCalibrationAnalyzer, EDCalibrationResult, EDMaterialMeasurement,
)


@pytest.fixture
def ed_analyzer():
    return EDCalibrationAnalyzer(config=CONFIG)


class TestEDConfig:
    def test_phantom_materials_count(self):
        assert len(CONFIG.ed_calibration.phantom_materials) == 13

    def test_each_entry_3_tuple(self):
        for entry in CONFIG.ed_calibration.phantom_materials:
            assert len(entry) == 3
            assert isinstance(entry[0], str)
            assert isinstance(entry[1], (int, float))
            assert isinstance(entry[2], float)

    def test_red_values_bounded(self):
        for _, _, red in CONFIG.ed_calibration.phantom_materials:
            assert 0.0 <= red <= 2.0

    def test_nominal_hu_monotonically_increasing(self):
        hus = [e[1] for e in CONFIG.ed_calibration.phantom_materials]
        for i in range(1, len(hus)):
            assert hus[i] > hus[i - 1]


class TestFitCalibrationCurve:
    def test_fit_from_reference_data(self, ed_analyzer):
        """Fitting the reference table data should give R^2 > 0.99 for each segment."""
        hu_vals = [e[1] for e in CONFIG.ed_calibration.phantom_materials]
        red_vals = [e[2] for e in CONFIG.ed_calibration.phantom_materials]

        soft_params, bone_params = ed_analyzer._fit_calibration_curve(
            hu_vals, red_vals, 100.0
        )
        assert soft_params["r_squared"] > 0.98
        assert bone_params["r_squared"] > 0.98

    def test_compute_red_water(self, ed_analyzer):
        """RED at HU=0 should be approximately 1.0 (water)."""
        hu_vals = [e[1] for e in CONFIG.ed_calibration.phantom_materials]
        red_vals = [e[2] for e in CONFIG.ed_calibration.phantom_materials]
        soft, bone = ed_analyzer._fit_calibration_curve(hu_vals, red_vals, 100.0)

        red_water = ed_analyzer._compute_red_from_curve(0.0, soft, bone, 100.0)
        assert red_water == pytest.approx(1.0, abs=0.05)

    def test_compute_red_air(self, ed_analyzer):
        """RED at HU=-1000 should be approximately 0.001 (air)."""
        hu_vals = [e[1] for e in CONFIG.ed_calibration.phantom_materials]
        red_vals = [e[2] for e in CONFIG.ed_calibration.phantom_materials]
        soft, bone = ed_analyzer._fit_calibration_curve(hu_vals, red_vals, 100.0)

        red_air = ed_analyzer._compute_red_from_curve(-1000.0, soft, bone, 100.0)
        assert red_air == pytest.approx(0.001, abs=0.1)

    def test_generate_curve_samples_length(self, ed_analyzer):
        hu_vals = [e[1] for e in CONFIG.ed_calibration.phantom_materials]
        red_vals = [e[2] for e in CONFIG.ed_calibration.phantom_materials]
        soft, bone = ed_analyzer._fit_calibration_curve(hu_vals, red_vals, 100.0)

        hu_arr, red_arr = ed_analyzer._generate_curve_samples(soft, bone, 100.0, n_points=250)
        assert len(hu_arr) == 250
        assert len(red_arr) == 250


class TestEDCalibrationResult:
    @pytest.fixture
    def sample_result(self, ed_analyzer):
        """Build a calibration result from the reference table directly."""
        hu_vals = [float(e[1]) for e in CONFIG.ed_calibration.phantom_materials]
        red_vals = [float(e[2]) for e in CONFIG.ed_calibration.phantom_materials]

        soft, bone = ed_analyzer._fit_calibration_curve(hu_vals, red_vals, 100.0)
        hu_curve, red_curve = ed_analyzer._generate_curve_samples(soft, bone, 100.0)

        measurements = []
        for name, nom_hu, ref_red in CONFIG.ed_calibration.phantom_materials:
            computed_red = ed_analyzer._compute_red_from_curve(float(nom_hu), soft, bone, 100.0)
            deviation = abs(computed_red - ref_red)
            measurements.append(EDMaterialMeasurement(
                material_name=name, nominal_hu=float(nom_hu),
                measured_mean_hu=float(nom_hu), measured_std_hu=2.0,
                reference_red=ref_red, computed_red=computed_red,
                red_deviation=deviation, passed=deviation <= 0.02,
            ))

        return EDCalibrationResult(
            acquisition_date="20240115", series_description="QA_HEAD",
            scanner_id="SCANNER_001", measurements=measurements,
            soft_tissue_slope=soft["slope"], soft_tissue_intercept=soft["intercept"],
            soft_tissue_r_squared=soft["r_squared"], soft_tissue_hu_range=soft["hu_range"],
            bone_slope=bone["slope"], bone_intercept=bone["intercept"],
            bone_r_squared=bone["r_squared"], bone_hu_range=bone["hu_range"],
            segment_join_hu=100.0, hu_curve=hu_curve, red_curve=red_curve,
            max_red_deviation=max(m.red_deviation for m in measurements),
            mean_red_deviation=float(np.mean([m.red_deviation for m in measurements])),
            all_passed=all(m.passed for m in measurements),
        )

    def test_get_red_for_hu_water(self, sample_result):
        assert sample_result.get_red_for_hu(0.0) == pytest.approx(1.0, abs=0.05)

    def test_get_red_for_hu_air(self, sample_result):
        assert sample_result.get_red_for_hu(-1000.0) == pytest.approx(0.001, abs=0.1)

    def test_passes_clinical_acceptance(self, sample_result):
        assert isinstance(sample_result.passes_clinical_acceptance(), bool)

    def test_export_csv(self, sample_result, tmp_path):
        out = tmp_path / "curve.csv"
        result_path = sample_result.export_for_tps(out, format="generic_csv")
        assert result_path.exists()

        # Verify CSV structure
        with open(result_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert "HU" in headers
            assert "RED" in headers

    def test_to_dict_no_ndarray(self, sample_result):
        d = sample_result.to_dict()
        _assert_no_ndarray(d)


class TestCompareToReference:
    def test_identical_no_warning(self, ed_analyzer):
        """Comparing identical results should have zero drift and no warning."""
        hu_vals = [float(e[1]) for e in CONFIG.ed_calibration.phantom_materials]
        red_vals = [float(e[2]) for e in CONFIG.ed_calibration.phantom_materials]
        soft, bone = ed_analyzer._fit_calibration_curve(hu_vals, red_vals, 100.0)
        hu_curve, red_curve = ed_analyzer._generate_curve_samples(soft, bone, 100.0)

        measurements = []
        for name, nom_hu, ref_red in CONFIG.ed_calibration.phantom_materials:
            computed_red = ed_analyzer._compute_red_from_curve(float(nom_hu), soft, bone, 100.0)
            measurements.append(EDMaterialMeasurement(
                material_name=name, nominal_hu=float(nom_hu),
                measured_mean_hu=float(nom_hu), measured_std_hu=2.0,
                reference_red=ref_red, computed_red=computed_red,
                red_deviation=abs(computed_red - ref_red), passed=True,
            ))

        result = EDCalibrationResult(
            acquisition_date="20240115", series_description="QA", scanner_id="S1",
            measurements=measurements,
            soft_tissue_slope=soft["slope"], soft_tissue_intercept=soft["intercept"],
            soft_tissue_r_squared=soft["r_squared"], soft_tissue_hu_range=soft["hu_range"],
            bone_slope=bone["slope"], bone_intercept=bone["intercept"],
            bone_r_squared=bone["r_squared"], bone_hu_range=bone["hu_range"],
            segment_join_hu=100.0, hu_curve=hu_curve, red_curve=red_curve,
            max_red_deviation=0.01, mean_red_deviation=0.005, all_passed=True,
        )

        comparison = ed_analyzer.compare_to_reference(result, result)
        assert comparison["hardware_warning"] is False
        assert comparison["slope_soft_tissue_drift"] == pytest.approx(0.0, abs=1e-10)
        assert comparison["slope_bone_drift"] == pytest.approx(0.0, abs=1e-10)


def _assert_no_ndarray(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_no_ndarray(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _assert_no_ndarray(item)
    else:
        assert not isinstance(obj, np.ndarray)
