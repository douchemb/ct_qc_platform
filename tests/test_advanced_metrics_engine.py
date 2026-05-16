"""Tests for AdvancedMetricsEngine — routing logic and calculator integration."""
from __future__ import annotations
import json
import pytest
import numpy as np


class TestAdvancedMetricsEngineRouting:

    def test_engine_instantiates_no_calculators(self, advanced_metrics_engine):
        assert advanced_metrics_engine is not None

    def test_all_metrics_skipped_when_no_calculators(
        self, advanced_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = advanced_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert result.nps is None
        assert result.mtf is None
        assert len(result.skipped) > 0

    def test_nps_skipped_when_explicitly_requested(
        self, advanced_metrics_engine_full,
        volumetric_result_siemens, siemens_water_adapter
    ):
        result = advanced_metrics_engine_full.compute(
            volumetric_result_siemens, siemens_water_adapter,
            skip_modules=["nps", "mtf", "ed", "ssde"]
        )
        assert result.nps is None
        assert "nps" in result.skipped

    def test_mtf_skipped_for_phantom_without_edge_insert(
        self, advanced_metrics_engine_full,
        volumetric_result_siemens, siemens_water_adapter
    ):
        """Siemens water phantom has no edge insert — MTF must be skipped."""
        assert siemens_water_adapter.has_edge_insert() is False
        result = advanced_metrics_engine_full.compute(
            volumetric_result_siemens, siemens_water_adapter,
            skip_modules=["nps", "ed", "ssde"]
        )
        assert result.mtf is None
        assert any("edge insert" in s or "mtf" in s.lower() for s in result.skipped)

    def test_ed_skipped_for_phantom_without_density_inserts(
        self, advanced_metrics_engine_full,
        volumetric_result_siemens, siemens_water_adapter
    ):
        """Siemens water phantom has no density inserts — ED must be skipped."""
        assert siemens_water_adapter.has_density_inserts() is False
        result = advanced_metrics_engine_full.compute(
            volumetric_result_siemens, siemens_water_adapter,
            skip_modules=["nps", "mtf", "ssde"]
        )
        assert result.ed_calibration is None
        assert any("density" in s.lower() or "ed" in s.lower() for s in result.skipped)

    def test_ssde_skipped_when_ctdivol_missing(
        self, advanced_metrics_engine_full,
        volumetric_result_siemens, siemens_water_adapter
    ):
        result = advanced_metrics_engine_full.compute(
            volumetric_result_siemens, siemens_water_adapter,
            dose_metadata=None,
            skip_modules=["nps", "mtf", "ed"]
        )
        assert result.ssde_series is None
        assert any("ssde" in s.lower() or "ctdivol" in s.lower() for s in result.skipped)

    def test_result_to_dict_serializable(
        self, advanced_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = advanced_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        json.dumps(result.to_dict(), default=str)

    def test_all_passed_is_bool(
        self, advanced_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = advanced_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert isinstance(result.all_passed, bool)

    def test_errors_is_list(
        self, advanced_metrics_engine, volumetric_result_siemens, siemens_water_adapter
    ):
        result = advanced_metrics_engine.compute(
            volumetric_result_siemens, siemens_water_adapter
        )
        assert isinstance(result.errors, list)

    def test_nps_computed_for_full_engine(
        self, advanced_metrics_engine_full,
        volumetric_result_siemens, siemens_water_adapter
    ):
        result = advanced_metrics_engine_full.compute(
            volumetric_result_siemens, siemens_water_adapter,
            skip_modules=["mtf", "hu_linearity", "ed", "ssde"]
        )
        assert isinstance(result.errors, list)
