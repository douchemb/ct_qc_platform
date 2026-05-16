# -*- coding: utf-8 -*-
"""
tests/test_localizer_parser.py — Tests for CT Localizer Parser.

Validates the AAPM TG-220 Appendix A LPV calibration pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.dosimetry.localizer_parser import (
    LocalizerParser,
    LocalizerData,
    LocalizerMetadata,
    LocalizerNotFoundError,
    LocalizerCalibrationError,
)


class TestLocalizerParserInstantiation:
    """Test that LocalizerParser instantiates correctly."""

    def test_instantiates_without_error(self, localizer_parser_instance):
        """LocalizerParser should instantiate without error."""
        assert localizer_parser_instance is not None
        assert isinstance(localizer_parser_instance, LocalizerParser)


class TestLocalizerParsing:
    """Test the full parse pipeline."""

    def test_parse_runs_without_error(self, localizer_parser_instance, saved_localizer_file):
        """parser.parse(saved_localizer_file) should run without error."""
        result = localizer_parser_instance.parse(saved_localizer_file)
        assert result is not None
        assert isinstance(result, LocalizerData)

    def test_image_orientation_is_ap(self, parsed_localizer_data):
        """Synthetic localizer has dominant x-component → AP orientation."""
        assert parsed_localizer_data.metadata.image_orientation == "AP"

    def test_air_reference_row_in_expected_region(self, parsed_localizer_data):
        """Air reference row should be within top or bottom air reference region."""
        row = parsed_localizer_data.air_reference_row
        # Air rows are 0–49 and 460–511, but we exclude 5% margin (25 rows)
        # So valid range is rows 25–486 approximately
        # The air reference should be found in the top block (rows 25–49)
        # or bottom block (rows 460–486)
        assert (25 <= row <= 51) or (460 <= row <= 511), (
            "Air reference row %d not in expected air region" % row
        )

    def test_lpv_calibrated_min_bound(self, parsed_localizer_data):
        """Calibrated LPV minimum should be >= 0.001 (clipping applied)."""
        assert parsed_localizer_data.lpv_calibrated.min() >= 0.001

    def test_lpv_calibrated_max_bound(self, parsed_localizer_data):
        """Calibrated LPV maximum should be <= 1.0 (clipping applied)."""
        assert parsed_localizer_data.lpv_calibrated.max() <= 1.0

    def test_water_eq_path_length_non_negative(self, parsed_localizer_data):
        """Water-equivalent path length should have no negative values."""
        assert parsed_localizer_data.water_eq_path_length_cm.min() >= 0.0

    def test_body_mask_has_detected_body(self, parsed_localizer_data):
        """Body mask should detect some body pixels."""
        assert parsed_localizer_data.body_mask.sum() > 0

    def test_body_mask_dtype_is_bool(self, parsed_localizer_data):
        """Body mask should be boolean array."""
        assert parsed_localizer_data.body_mask.dtype == bool

    def test_mean_path_length_in_body_physical_range(self, parsed_localizer_data):
        """Mean water-equivalent path length in body should be between 5.0 and 35.0 cm."""
        body_paths = parsed_localizer_data.water_eq_path_length_cm[
            parsed_localizer_data.body_mask
        ]
        if body_paths.size > 0:
            mean_path = float(np.mean(body_paths))
            assert 5.0 <= mean_path <= 35.0, (
                "Mean path length %.2f cm outside physical range [5, 35]" % mean_path
            )

    def test_lpv_shape_matches_input(self, parsed_localizer_data):
        """Calibrated LPV shape should match input dimensions (512, 256)."""
        assert parsed_localizer_data.lpv_calibrated.shape == (512, 256)


class TestLocalizerFindInDirectory:
    """Test find_localizer_in_directory."""

    def test_raises_not_found_in_axial_directory(
        self, localizer_parser_instance, temp_dicom_dir
    ):
        """Should raise LocalizerNotFoundError when directory has only axial CT files."""
        with pytest.raises(LocalizerNotFoundError):
            localizer_parser_instance.find_localizer_in_directory(temp_dicom_dir)


class TestLocalizerSerialization:
    """Test to_dict serialization."""

    def test_to_dict_returns_dict(self, parsed_localizer_data):
        """to_dict() should return a dict."""
        result = parsed_localizer_data.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_contains_no_ndarray(self, parsed_localizer_data):
        """to_dict() should contain no numpy.ndarray values (all converted to lists)."""
        result = parsed_localizer_data.to_dict()

        def _check_no_ndarray(obj, path=""):
            if isinstance(obj, np.ndarray):
                pytest.fail("Found numpy.ndarray at path: %s" % path)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _check_no_ndarray(v, path="%s.%s" % (path, k))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _check_no_ndarray(v, path="%s[%d]" % (path, i))

        _check_no_ndarray(result, "root")
