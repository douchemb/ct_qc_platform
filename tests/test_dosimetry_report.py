# -*- coding: utf-8 -*-
"""
tests/test_dosimetry_report.py — Tests for Dosimetry Report Generator.

Validates report assembly, JSON serialization, compliance flags,
and summary logging.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from modules.dosimetry.dosimetry_report import DosimetryReport
from modules.dosimetry.ssde_calculator import SSDESeriesResult
from modules.dosimetry.dw_calculator import DwSeriesResult, DwLocalizerResult


@pytest.fixture
def ssde_series_result(ssde_calculator_instance, dw_series_result, volumetric_result):
    """Pre-computed SSDESeriesResult for report testing."""
    metadata = volumetric_result.slice_results[0].metadata
    return ssde_calculator_instance.compute_series_ssde(
        dw_series_result, ctdi_vol_mgy=12.5, metadata=metadata,
    )


@pytest.fixture
def dosimetry_report(ssde_series_result, dw_series_result):
    """DosimetryReport without localizer."""
    return DosimetryReport(
        ssde_result=ssde_series_result,
        dw_result=dw_series_result,
    )


@pytest.fixture
def localizer_dw_result(dw_calculator_instance, parsed_localizer_data):
    """DwLocalizerResult from parsed localizer."""
    return dw_calculator_instance.compute_from_localizer(parsed_localizer_data)


class TestDosimetryReportInstantiation:
    """Test report construction."""

    def test_instantiates_without_error(self, dosimetry_report):
        """DosimetryReport should instantiate without error."""
        assert dosimetry_report is not None
        assert isinstance(dosimetry_report, DosimetryReport)


class TestDosimetryReportToDict:
    """Test to_dict serialization."""

    def test_to_dict_returns_dict(self, dosimetry_report):
        """to_dict() should return a dict."""
        d = dosimetry_report.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_required_keys(self, dosimetry_report):
        """to_dict() should contain required keys."""
        d = dosimetry_report.to_dict()
        assert "ssde_result" in d
        assert "dw_result" in d
        assert "compliance_flags" in d


class TestComplianceFlags:
    """Test compliance flag generation."""

    def test_get_compliance_flags_returns_dict(self, dosimetry_report):
        """get_compliance_flags() should return a dict."""
        flags = dosimetry_report.get_compliance_flags()
        assert isinstance(flags, dict)

    def test_get_compliance_flags_has_5_keys(self, dosimetry_report):
        """get_compliance_flags() should have exactly 5 keys."""
        flags = dosimetry_report.get_compliance_flags()
        expected_keys = {
            "ctdi_below_drl",
            "ssde_computed",
            "dw_within_table_range",
            "effective_dose_estimated",
            "localizer_available",
        }
        assert set(flags.keys()) == expected_keys

    def test_all_compliance_flags_are_bool(self, dosimetry_report):
        """All values in get_compliance_flags() should be booleans."""
        flags = dosimetry_report.get_compliance_flags()
        for key, value in flags.items():
            assert isinstance(value, bool), (
                "Flag '%s' has type %s, expected bool" % (key, type(value).__name__)
            )

    def test_localizer_available_false_without_localizer(self, dosimetry_report):
        """localizer_available should be False when no localizer provided."""
        flags = dosimetry_report.get_compliance_flags()
        assert flags["localizer_available"] is False

    def test_localizer_available_true_with_localizer(
        self, ssde_series_result, dw_series_result, localizer_dw_result
    ):
        """localizer_available should be True when DwLocalizerResult is provided."""
        report = DosimetryReport(
            ssde_result=ssde_series_result,
            dw_result=dw_series_result,
            localizer_result=localizer_dw_result,
        )
        flags = report.get_compliance_flags()
        assert flags["localizer_available"] is True


class TestDosimetryReportJSON:
    """Test JSON serialization."""

    def test_to_json_creates_file(self, dosimetry_report, tmp_path):
        """to_json() should create the output file (atomic write)."""
        output_path = tmp_path / "report.json"
        dosimetry_report.to_json(output_path)
        assert output_path.is_file()

    def test_json_file_loads_correctly(self, dosimetry_report, tmp_path):
        """Written JSON file should load successfully with json.load()."""
        output_path = tmp_path / "report.json"
        dosimetry_report.to_json(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert "ssde_result" in data


class TestDosimetryReportSummary:
    """Test summary logging."""

    def test_print_summary_runs(self, dosimetry_report):
        """print_summary() should run without raising."""
        dosimetry_report.print_summary()


class TestDosimetryReportWithLocalizer:
    """Test report with localizer data."""

    def test_with_localizer_runs(
        self, ssde_series_result, dw_series_result, localizer_dw_result
    ):
        """DosimetryReport with localizer should instantiate without error."""
        report = DosimetryReport(
            ssde_result=ssde_series_result,
            dw_result=dw_series_result,
            localizer_result=localizer_dw_result,
        )
        assert report is not None
        assert report.localizer_result is not None
