# -*- coding: utf-8 -*-
"""tests/test_config.py — Configuration validation tests."""

import dataclasses
from pathlib import Path

import pytest

from config import CONFIG, AppConfig


class TestConfigImport:
    def test_config_is_app_config(self):
        assert isinstance(CONFIG, AppConfig)

    def test_noise_tolerance(self):
        assert CONFIG.image_qc.noise_tolerance_hu == 5.0

    def test_hu_water_nominal(self):
        assert CONFIG.image_qc.hu_water_nominal == 0.0

    def test_default_start_slice(self):
        assert CONFIG.image_qc.default_start_slice == 1

    def test_default_end_slice(self):
        assert CONFIG.image_qc.default_end_slice == 999

    def test_min_slices_for_volumetric(self):
        assert CONFIG.image_qc.min_slices_for_volumetric == 3

    def test_data_dir_name(self):
        assert CONFIG.paths.data_dir.name == "data"

    def test_all_paths_are_path_instances(self):
        for f in dataclasses.fields(CONFIG.paths):
            assert isinstance(getattr(CONFIG.paths, f.name), Path)


class TestSSDEConversionTable:
    def test_non_empty_tuple(self):
        assert isinstance(CONFIG.dosimetry.ssde_conversion_table, tuple)
        assert len(CONFIG.dosimetry.ssde_conversion_table) > 0

    def test_each_entry_has_two_elements(self):
        for entry in CONFIG.dosimetry.ssde_conversion_table:
            assert len(entry) == 2

    def test_dw_monotonically_increasing(self):
        dws = [e[0] for e in CONFIG.dosimetry.ssde_conversion_table]
        for i in range(1, len(dws)):
            assert dws[i] > dws[i - 1]

    def test_f_factors_positive_and_bounded(self):
        for _, f in CONFIG.dosimetry.ssde_conversion_table:
            assert f > 0.0
            assert f < 3.0


class TestEDCalibration:
    def test_phantom_materials_count(self):
        assert len(CONFIG.ed_calibration.phantom_materials) == 13

    def test_each_entry_is_3_tuple(self):
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

    def test_max_red_deviation(self):
        assert CONFIG.ed_calibration.max_red_deviation == 0.02


class TestFrozen:
    def test_frozen_raises(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            CONFIG.image_qc = None
