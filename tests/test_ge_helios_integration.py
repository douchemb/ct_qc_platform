# -*- coding: utf-8 -*-
"""
tests/test_ge_helios_integration.py
=====================================
Integration tests for Phase 5: GE Discovery RT DICOM compatibility
and GE Helios QA Phantom automatic geometry detection.

All tests use synthetic fixtures — no real GE DICOM files required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian

from config import CONFIG
from core.dicom_loader import DicomLoader


# ═══════════════════════════════════════════════════════════════════
# GE DICOM Loader Tests
# ═══════════════════════════════════════════════════════════════════

class TestGEDicomLoader:
    """Tests for GE Discovery RT DICOM compatibility in DicomLoader."""

    def test_ge_loader_resolves_rescale_from_private_tag(
        self, dicom_loader_instance, synthetic_ge_ct_dicom
    ):
        """to_hu_array succeeds and water mean HU is within ±10 of 0.0."""
        hu_array = dicom_loader_instance.to_hu_array(synthetic_ge_ct_dicom)
        assert hu_array.dtype == np.float32
        # Water region around center — exclude insert areas
        water_roi = hu_array[230:280, 230:280]
        assert abs(float(np.mean(water_roi))) < 10.0

    def test_ge_loader_handles_missing_standard_rescale_tags(
        self, dicom_loader_instance, synthetic_ge_ct_dicom
    ):
        """Verify standard RescaleSlope is absent; to_hu_array still works."""
        assert not hasattr(synthetic_ge_ct_dicom, "RescaleSlope")
        # Should not raise
        hu_array = dicom_loader_instance.to_hu_array(synthetic_ge_ct_dicom)
        assert hu_array is not None

    def test_ge_loader_string_slice_thickness(
        self, dicom_loader_instance, synthetic_ge_ct_dicom
    ):
        """extract_metadata parses '3.0 mm' SliceThickness correctly."""
        metadata = dicom_loader_instance.extract_metadata(synthetic_ge_ct_dicom)
        assert metadata.slice_thickness_mm == pytest.approx(3.0, abs=0.01)

    def test_ge_loader_string_kvp(
        self, dicom_loader_instance, synthetic_ge_ct_dicom
    ):
        """extract_metadata parses string '120.0' KVP correctly."""
        metadata = dicom_loader_instance.extract_metadata(synthetic_ge_ct_dicom)
        assert metadata.kvp == pytest.approx(120.0, abs=0.01)

    def test_ge_localizer_detection_via_image_type(
        self, dicom_loader_instance, synthetic_ge_localizer_dicom
    ):
        """GE localizer detected via ImageType despite standard CT SOP UID."""
        assert synthetic_ge_localizer_dicom.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2"
        assert dicom_loader_instance.is_localizer(synthetic_ge_localizer_dicom) is True

    def test_ge_standard_ct_is_not_localizer(
        self, dicom_loader_instance, synthetic_ge_ct_dicom
    ):
        """Axial CT image is not detected as a localizer."""
        assert dicom_loader_instance.is_localizer(synthetic_ge_ct_dicom) is False

    def test_ge_load_file_succeeds(self, dicom_loader_instance, saved_ge_dicom_file):
        """load_file returns a dataset without raising."""
        ds = dicom_loader_instance.load_file(saved_ge_dicom_file)
        assert ds is not None
        assert ds.Modality == "CT"

    def test_ge_instance_number_zero_fallback(
        self, dicom_loader_instance, tmp_path, caplog
    ):
        """3 GE CT files with InstanceNumber=0 — load_directory succeeds with fallback."""
        for i in range(3):
            file_meta = pydicom.Dataset()
            file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            filename = str(tmp_path / ("ge_slice_%03d.dcm" % i))
            ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ds.SOPInstanceUID = pydicom.uid.generate_uid()
            ds.Modality = "CT"
            ds.InstanceNumber = 0  # GE bug
            ds.SliceLocation = float(i * 3.0)
            ds.RescaleSlope = 1.0
            ds.RescaleIntercept = -1024.0
            ds.Rows = 64
            ds.Columns = 64
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            arr = np.full((64, 64), 1024, dtype=np.uint16)
            ds.PixelData = arr.tobytes()
            ds.save_as(filename)

        import logging
        with caplog.at_level(logging.WARNING):
            datasets = dicom_loader_instance.load_directory(tmp_path)

        assert len(datasets) == 3
        assert any("InstanceNumber" in msg for msg in caplog.messages)

    def test_ge_resolve_rescale_logs_warning_on_fallback(
        self, dicom_loader_instance, synthetic_ge_ct_dicom, caplog
    ):
        """GE private tag fallback logs a WARNING."""
        import logging
        with caplog.at_level(logging.WARNING):
            dicom_loader_instance.to_hu_array(synthetic_ge_ct_dicom)
        assert any(
            "GE private tag" in msg or "default" in msg.lower()
            for msg in caplog.messages
        )


# ═══════════════════════════════════════════════════════════════════
# Helios Geometry Detection Tests
# ═══════════════════════════════════════════════════════════════════

class TestHeliosGeometryDetection:
    """Tests for HeliosPhantomDetector automatic geometry detection."""

    def test_helios_detector_instantiates(self, helios_detector_instance):
        """HeliosPhantomDetector(CONFIG) does not raise."""
        assert helios_detector_instance is not None

    def test_helios_detect_returns_geometry_object(
        self, helios_detector_instance, synthetic_helios_hu_array
    ):
        """detect() returns a HeliosGeometry instance."""
        from modules.image_qc.phantom_geometry import HeliosGeometry
        hu_array, spacing = synthetic_helios_hu_array
        geometry = helios_detector_instance.detect(hu_array, spacing)
        assert isinstance(geometry, HeliosGeometry)

    def test_helios_center_detection_accuracy(
        self, helios_detector_instance, synthetic_helios_hu_array
    ):
        """Detected center within 5 px of true center (258, 255)."""
        hu_array, spacing = synthetic_helios_hu_array
        geometry = helios_detector_instance.detect(hu_array, spacing)
        assert abs(geometry.center_row - 258.0) < 5.0
        assert abs(geometry.center_col - 255.0) < 5.0

    def test_helios_radius_detection_accuracy(
        self, helios_detector_instance, synthetic_helios_hu_array
    ):
        """Detected radius within 5% of expected ~102.3 px."""
        hu_array, spacing = synthetic_helios_hu_array
        geometry = helios_detector_instance.detect(hu_array, spacing)
        expected_radius = 100.0 / 0.977  # ~102.3 px
        assert geometry.phantom_radius_px == pytest.approx(expected_radius, rel=0.05)

    def test_helios_all_seven_rois_present(
        self, helios_detector_instance, synthetic_helios_hu_array
    ):
        """geometry.roi_descriptors has exactly 7 keys."""
        hu_array, spacing = synthetic_helios_hu_array
        geometry = helios_detector_instance.detect(hu_array, spacing)
        expected_keys = {"water", "air", "acrylic", "ldpe", "polystyrene", "delrin", "teflon"}
        assert set(geometry.roi_descriptors.keys()) == expected_keys

    def test_helios_all_rois_within_image_bounds(
        self, helios_detector_instance, synthetic_helios_hu_array
    ):
        """All ROIs are within [0, 512] bounds."""
        hu_array, spacing = synthetic_helios_hu_array
        geometry = helios_detector_instance.detect(hu_array, spacing)
        for name, roi in geometry.roi_descriptors.items():
            assert roi.row_start >= 0, "ROI '%s' row_start < 0" % name
            assert roi.col_start >= 0, "ROI '%s' col_start < 0" % name
            assert roi.row_end <= 512, "ROI '%s' row_end > 512" % name
            assert roi.col_end <= 512, "ROI '%s' col_end > 512" % name

    def test_helios_water_roi_at_center(
        self, helios_detector_instance, synthetic_helios_hu_array
    ):
        """Water ROI center is within 5 px of detected phantom center."""
        hu_array, spacing = synthetic_helios_hu_array
        geometry = helios_detector_instance.detect(hu_array, spacing)
        water_roi = geometry.roi_descriptors["water"]
        water_center_row = water_roi.row_start + water_roi.height_px / 2.0
        water_center_col = water_roi.col_start + water_roi.width_px / 2.0
        assert abs(water_center_row - geometry.center_row) < 5.0
        assert abs(water_center_col - geometry.center_col) < 5.0

    def test_helios_air_insert_at_top(
        self, helios_detector_instance, synthetic_helios_hu_array
    ):
        """Air insert (0°) is above phantom center (smaller row index)."""
        hu_array, spacing = synthetic_helios_hu_array
        geometry = helios_detector_instance.detect(hu_array, spacing)
        air_roi = geometry.roi_descriptors["air"]
        air_center_row = air_roi.row_start + air_roi.height_px // 2
        assert air_center_row < geometry.center_row

    def test_helios_detection_quality_is_good_or_degraded(
        self, helios_detector_instance, synthetic_helios_hu_array
    ):
        """Detection quality is 'good' or 'degraded' on valid phantom."""
        hu_array, spacing = synthetic_helios_hu_array
        geometry = helios_detector_instance.detect(hu_array, spacing)
        assert geometry.detection_quality in ("good", "degraded")

    def test_helios_detect_from_volume(
        self, helios_detector_instance, volumetric_result
    ):
        """detect_from_volume does not raise on generic volumetric_result."""
        from modules.image_qc.phantom_geometry import HeliosGeometry
        geometry = helios_detector_instance.detect_from_volume(volumetric_result)
        assert isinstance(geometry, HeliosGeometry)

    def test_helios_roi_area_minimum(
        self, helios_detector_instance, synthetic_helios_hu_array
    ):
        """All 7 ROIs have area >= 400 px (TG-66 Section 5.1 minimum)."""
        hu_array, spacing = synthetic_helios_hu_array
        geometry = helios_detector_instance.detect(hu_array, spacing)
        for name, roi in geometry.roi_descriptors.items():
            assert roi.area_px >= 400, (
                "ROI '%s' area = %d < 400 px minimum" % (name, roi.area_px)
            )


# ═══════════════════════════════════════════════════════════════════
# Configuration Tests
# ═══════════════════════════════════════════════════════════════════

class TestHeliosConfiguration:
    """Tests for Phase 5 configuration additions."""

    def test_helios_phantom_config_in_config(self):
        assert hasattr(CONFIG, "helios_phantom")

    def test_helios_phantom_materials_count(self):
        assert len(CONFIG.helios_phantom.phantom_materials) == 7

    def test_helios_phantom_material_names(self):
        names = {m[0] for m in CONFIG.helios_phantom.phantom_materials}
        expected = {"water", "air", "acrylic", "ldpe", "polystyrene", "delrin", "teflon"}
        assert names == expected

    def test_helios_nominal_hu_values_physical(self):
        """All nominal HU values are in physical CT range [-1100, 1500]."""
        for mat in CONFIG.helios_phantom.phantom_materials:
            assert -1100 <= mat[1] <= 1500, (
                "Material '%s' has HU = %d outside physical range" % (mat[0], mat[1])
            )

    def test_helios_red_values_physical(self):
        """All RED values are in physical range [0.0, 2.5]."""
        for mat in CONFIG.helios_phantom.phantom_materials:
            assert 0.0 <= mat[2] <= 2.5, (
                "Material '%s' has RED = %.3f outside physical range" % (mat[0], mat[2])
            )

    def test_ge_scanner_config_in_config(self):
        assert hasattr(CONFIG, "ge_scanner")

    def test_ge_scanner_private_tag_correct(self):
        assert CONFIG.ge_scanner.ge_private_rescale_slope_tag == (0x0009, 0x100D)

    def test_phantom_config_json_loadable(self):
        """phantom_config.json exists and is valid JSON."""
        config_path = Path(__file__).parent.parent / "phantom_config.json"
        assert config_path.is_file(), "phantom_config.json not found at %s" % config_path
        with open(config_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert "active_phantom" in loaded
        assert "active_scanner" in loaded

    def test_phantom_config_json_active_phantom_valid(self):
        config_path = Path(__file__).parent.parent / "phantom_config.json"
        with open(config_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["active_phantom"] in ("ge_helios", "catphan_504")


# ═══════════════════════════════════════════════════════════════════
# HeliosPhantomDetector Unit Method Tests
# ═══════════════════════════════════════════════════════════════════

class TestHeliosDetectorUnitMethods:
    """Unit tests for individual HeliosPhantomDetector methods."""

    def test_nominal_geometry_fallback(self, helios_detector_instance):
        """_nominal_geometry returns correct values for 512x512 at 0.977 mm/px."""
        center_row, center_col, radius_px = helios_detector_instance._nominal_geometry(
            np.zeros((512, 512)), (0.977, 0.977)
        )
        assert center_row == pytest.approx(256.0, abs=0.1)
        assert center_col == pytest.approx(256.0, abs=0.1)
        expected_radius = 100.0 / 0.977
        assert radius_px == pytest.approx(expected_radius, abs=0.1)

    def test_compute_roi_descriptors_angles(self, helios_detector_instance):
        """Verify air insert (0°) is directly above center, polystyrene (180°) below."""
        rois = helios_detector_instance._compute_roi_descriptors(
            center_row=256.0, center_col=256.0,
            insert_radius_px=59.4, roi_radius_px=10.2,
            pixel_spacing_mm=(0.977, 0.977), image_shape=(512, 512),
        )
        # Air (0°) should be centered at col ≈ 256
        air_center_col = rois["air"].col_start + rois["air"].width_px // 2
        assert abs(air_center_col - 256) <= 2

        # Polystyrene (180°) should be below center
        poly_center_row = rois["polystyrene"].row_start + rois["polystyrene"].height_px // 2
        assert poly_center_row > 256

    def test_validate_geometry_good(self, helios_detector_instance):
        """Good geometry: center near image center, correct radius."""
        quality, notes = helios_detector_instance._validate_geometry(
            256.0, 256.0, 102.3, (512, 512), (0.977, 0.977)
        )
        assert quality == "good"

    def test_validate_geometry_failed_bad_radius(self, helios_detector_instance):
        """Failed geometry: obviously wrong radius."""
        quality, notes = helios_detector_instance._validate_geometry(
            256.0, 256.0, 10.0, (512, 512), (0.977, 0.977)
        )
        assert quality == "failed"
