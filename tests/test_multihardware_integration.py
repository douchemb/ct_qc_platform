"""
tests/test_multihardware_integration.py
========================================
Integration tests for Phase 5: Multi-Hardware & Multi-Phantom Integration.

All tests use synthetic DICOM fixtures — no real scanner files required.
Tests validate the complete adapter chain from DICOM metadata to
correctly positioned ROIDescriptor objects.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# SCANNER PROFILE REGISTRY TESTS (Step 5.1)
# ═══════════════════════════════════════════════════════════════════════════

class TestScannerProfileRegistry:

    def test_registry_loads_without_error(self, scanner_profile_registry):
        assert scanner_profile_registry is not None

    def test_registry_has_siemens_profile(self, scanner_profile_registry):
        profiles = scanner_profile_registry.available_profiles
        assert "siemens_somatom_gosim" in profiles

    def test_registry_has_ge_profile(self, scanner_profile_registry):
        assert "ge_discovery_rt" in scanner_profile_registry.available_profiles

    def test_registry_has_generic_profile(self, scanner_profile_registry):
        assert "generic" in scanner_profile_registry.available_profiles

    def test_detects_siemens_by_model(
        self, scanner_profile_registry, synthetic_siemens_ct_dicom
    ):
        profile = scanner_profile_registry.detect(synthetic_siemens_ct_dicom)
        assert profile.profile_id == "siemens_somatom_gosim"

    def test_detects_siemens_is_siemens(
        self, scanner_profile_registry, synthetic_siemens_ct_dicom
    ):
        profile = scanner_profile_registry.detect(synthetic_siemens_ct_dicom)
        assert profile.is_siemens() is True
        assert profile.is_ge() is False

    def test_siemens_uses_standard_rescale(
        self, scanner_profile_registry, synthetic_siemens_ct_dicom
    ):
        profile = scanner_profile_registry.detect(synthetic_siemens_ct_dicom)
        assert profile.rescale_method == "standard"

    def test_siemens_no_private_tags(
        self, scanner_profile_registry, synthetic_siemens_ct_dicom
    ):
        profile = scanner_profile_registry.detect(synthetic_siemens_ct_dicom)
        assert profile.private_tags_enabled is False

    def test_siemens_no_instance_number_bug(
        self, scanner_profile_registry, synthetic_siemens_ct_dicom
    ):
        profile = scanner_profile_registry.detect(synthetic_siemens_ct_dicom)
        assert profile.instance_number_bug is False

    def test_siemens_dlp_source_is_rdsr(
        self, scanner_profile_registry, synthetic_siemens_ct_dicom
    ):
        profile = scanner_profile_registry.detect(synthetic_siemens_ct_dicom)
        assert profile.dlp_source == "rdsr"

    def test_detects_ge_scanner(self, scanner_profile_registry, synthetic_ge_ct_dicom):
        profile = scanner_profile_registry.detect(synthetic_ge_ct_dicom)
        assert profile.profile_id == "ge_discovery_rt"
        assert profile.is_ge() is True

    def test_ge_private_tags_enabled(self, scanner_profile_registry, synthetic_ge_ct_dicom):
        profile = scanner_profile_registry.detect(synthetic_ge_ct_dicom)
        assert profile.private_tags_enabled is True

    def test_ge_instance_number_bug_flag(self, scanner_profile_registry, synthetic_ge_ct_dicom):
        profile = scanner_profile_registry.detect(synthetic_ge_ct_dicom)
        assert profile.instance_number_bug is True

    def test_unknown_scanner_returns_generic(self, scanner_profile_registry):
        """Non-strict mode must return generic profile for unknown scanners."""
        import pydicom
        from pydicom.dataset import Dataset
        ds = Dataset()
        ds.Manufacturer = "UNKNOWN_VENDOR"
        ds.ManufacturerModelName = "PHANTOM_SCANNER_9000"
        profile = scanner_profile_registry.detect(ds)
        assert profile.is_generic() is True

    def test_get_profile_by_id(self, scanner_profile_registry):
        p = scanner_profile_registry.get_profile("siemens_somatom_gosim")
        assert p.display_name is not None

    def test_get_invalid_profile_raises_key_error(self, scanner_profile_registry):
        with pytest.raises(KeyError):
            scanner_profile_registry.get_profile("nonexistent_profile_xyz")

    def test_profile_to_dict_is_serializable(
        self, scanner_profile_registry, synthetic_siemens_ct_dicom
    ):
        profile = scanner_profile_registry.detect(synthetic_siemens_ct_dicom)
        d = profile.to_dict()
        assert isinstance(d, dict)
        json.dumps(d)  # must be JSON-serializable


# ═══════════════════════════════════════════════════════════════════════════
# PHANTOM ADAPTER TESTS (Step 5.2)
# ═══════════════════════════════════════════════════════════════════════════

class TestPhantomAdapterFactory:

    def test_factory_instantiates(self, phantom_adapter_factory):
        assert phantom_adapter_factory is not None

    def test_detects_siemens_water_phantom(
        self, phantom_adapter_factory, synthetic_siemens_ct_dicom
    ):
        from modules.image_qc.phantom_adapters import SiemensWaterPhantomAdapter
        adapter = phantom_adapter_factory.create(synthetic_siemens_ct_dicom)
        assert isinstance(adapter, SiemensWaterPhantomAdapter)

    def test_unknown_series_returns_generic(self, phantom_adapter_factory):
        import pydicom
        from pydicom.dataset import Dataset
        from modules.image_qc.phantom_adapters import GenericPhantomAdapter
        ds = Dataset()
        ds.SeriesDescription = "UNKNOWN_PROTOCOL_XYZ"
        ds.ProtocolName      = "UNKNOWN"
        adapter = phantom_adapter_factory.create(ds)
        assert isinstance(adapter, GenericPhantomAdapter)

    def test_override_phantom_id(self, phantom_adapter_factory, synthetic_siemens_ct_dicom):
        from modules.image_qc.phantom_adapters import HeliosQAPhantomAdapter
        adapter = phantom_adapter_factory.create(
            synthetic_siemens_ct_dicom,
            override_phantom_id="ge_helios_qa"
        )
        assert isinstance(adapter, HeliosQAPhantomAdapter)


class TestSiemensWaterPhantomAdapter:

    @pytest.fixture
    def adapter(self):
        from modules.image_qc.phantom_adapters import SiemensWaterPhantomAdapter
        return SiemensWaterPhantomAdapter()

    @pytest.fixture
    def water_hu_array(self):
        """512×512 water phantom HU image — uniform 0 HU, circular body."""
        arr = np.full((512, 512), -1000.0, dtype=np.float32)  # air background
        cy, cx = 256, 256
        for r in range(512):
            for c in range(512):
                if (r - cy) ** 2 + (c - cx) ** 2 < 100 ** 2:
                    arr[r, c] = 0.0  # water phantom body
        return arr

    def test_has_5_rois(self, adapter, water_hu_array):
        rois = adapter.get_roi_descriptors(water_hu_array, (0.977, 0.977))
        assert len(rois) == 5

    def test_roi_labels_correct(self, adapter, water_hu_array):
        rois = adapter.get_roi_descriptors(water_hu_array, (0.977, 0.977))
        expected = {"center", "peripheral_12", "peripheral_3",
                    "peripheral_6", "peripheral_9"}
        assert set(rois.keys()) == expected

    def test_all_rois_within_image_bounds(self, adapter, water_hu_array):
        rois = adapter.get_roi_descriptors(water_hu_array, (0.977, 0.977))
        for label, roi in rois.items():
            assert roi.row_start >= 0, "%s: row_start < 0" % label
            assert roi.col_start >= 0, "%s: col_start < 0" % label
            assert roi.row_end <= 512, "%s: row_end > 512" % label
            assert roi.col_end <= 512, "%s: col_end > 512" % label

    def test_center_roi_near_image_center(self, adapter, water_hu_array):
        rois = adapter.get_roi_descriptors(water_hu_array, (0.977, 0.977))
        center_roi = rois["center"]
        roi_center_r = center_roi.row_start + center_roi.height_px // 2
        roi_center_c = center_roi.col_start + center_roi.width_px  // 2
        assert abs(roi_center_r - 256) < 10, "Center ROI row not near image center"
        assert abs(roi_center_c - 256) < 10, "Center ROI col not near image center"

    def test_has_no_edge_insert(self, adapter):
        assert adapter.has_edge_insert() is False

    def test_has_no_density_inserts(self, adapter):
        assert adapter.has_density_inserts() is False

    def test_material_references_has_water(self, adapter):
        refs = adapter.get_material_references()
        assert "water" in refs
        assert refs["water"].nominal_hu == pytest.approx(0.0, abs=0.1)

    def test_phantom_id_correct(self, adapter):
        assert adapter.phantom_id == "siemens_water_phantom"


class TestHeliosQAPhantomAdapter:

    @pytest.fixture
    def adapter(self):
        from modules.image_qc.phantom_adapters import HeliosQAPhantomAdapter
        return HeliosQAPhantomAdapter()

    @pytest.fixture
    def helios_hu_array(self):
        """512×512 Helios phantom HU image."""
        arr = np.full((512, 512), -1000.0, dtype=np.float32)
        cy, cx = 256, 256
        for r in range(512):
            for c in range(512):
                if (r - cy) ** 2 + (c - cx) ** 2 < 102 ** 2:
                    arr[r, c] = 0.0
        return arr

    def test_has_7_rois(self, adapter, helios_hu_array):
        rois = adapter.get_roi_descriptors(helios_hu_array, (0.977, 0.977))
        assert len(rois) == 7

    def test_roi_labels_include_all_materials(self, adapter, helios_hu_array):
        rois = adapter.get_roi_descriptors(helios_hu_array, (0.977, 0.977))
        expected = {"water", "air", "acrylic", "ldpe", "polystyrene", "delrin", "teflon"}
        assert set(rois.keys()) == expected

    def test_all_rois_within_bounds(self, adapter, helios_hu_array):
        rois = adapter.get_roi_descriptors(helios_hu_array, (0.977, 0.977))
        for label, roi in rois.items():
            assert roi.row_start >= 0
            assert roi.col_start >= 0
            assert roi.row_end <= 512
            assert roi.col_end <= 512

    def test_water_roi_at_center(self, adapter, helios_hu_array):
        rois = adapter.get_roi_descriptors(helios_hu_array, (0.977, 0.977))
        w = rois["water"]
        w_center_r = w.row_start + w.height_px // 2
        w_center_c = w.col_start + w.width_px  // 2
        assert abs(w_center_r - 256) < 10
        assert abs(w_center_c - 256) < 10

    def test_has_edge_insert(self, adapter):
        assert adapter.has_edge_insert() is True

    def test_has_density_inserts(self, adapter):
        assert adapter.has_density_inserts() is True

    def test_material_references_count(self, adapter):
        refs = adapter.get_material_references()
        assert len(refs) == 7

    def test_teflon_nominal_hu(self, adapter):
        refs = adapter.get_material_references()
        assert refs["teflon"].nominal_hu == pytest.approx(990.0, abs=1.0)

    def test_water_red_equals_one(self, adapter):
        refs = adapter.get_material_references()
        assert refs["water"].red == pytest.approx(1.000, abs=0.001)

    def test_all_red_values_physical(self, adapter):
        refs = adapter.get_material_references()
        for name, ref in refs.items():
            assert 0.0 <= ref.red <= 2.5, "%s: RED %s outside physical range" % (name, ref.red)


# ═══════════════════════════════════════════════════════════════════════════
# SPATIAL SORT ENGINE TESTS (Step 5.3)
# ═══════════════════════════════════════════════════════════════════════════

class TestSpatialSortEngine:

    def test_engine_instantiates(self, spatial_sort_engine):
        assert spatial_sort_engine is not None

    def test_sort_by_slice_location(self, spatial_sort_engine, temp_dicom_dir):
        from core.dicom_loader import DicomLoader
        from config import CONFIG
        loader = DicomLoader(CONFIG.dicom)
        datasets = loader.load_directory(temp_dicom_dir)
        result = spatial_sort_engine.sort(datasets)
        assert result.sort_method in (
            "SliceLocation", "ImagePositionPatient", "InstanceNumber"
        )
        assert len(result.datasets) == len(datasets)

    def test_sort_produces_monotonic_z(self, spatial_sort_engine, temp_dicom_dir):
        from core.dicom_loader import DicomLoader
        from config import CONFIG
        loader   = DicomLoader(CONFIG.dicom)
        datasets = loader.load_directory(temp_dicom_dir)
        result   = spatial_sort_engine.sort(datasets)
        z        = result.z_positions_mm
        diffs    = [z[i+1] - z[i] for i in range(len(z)-1)]
        # All diffs should have the same sign (monotonic)
        if len(diffs) > 1:
            assert all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs), \
                "Z-positions are not monotonic: %s" % z

    def test_sort_disordered_slices_by_instance_number(self):
        """Sort scrambled InstanceNumber returns correct order."""
        from core.spatial_sort import SpatialSortEngine
        import pydicom
        from pydicom.dataset import Dataset
        engine = SpatialSortEngine(preferred_tag="InstanceNumber")
        datasets = []
        for inst in [3, 1, 5, 2, 4]:
            ds = Dataset()
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ds.Modality = "CT"
            ds.InstanceNumber = inst
            ds.SliceLocation  = float(inst * 3)
            datasets.append(ds)
        result = engine.sort(datasets)
        instance_numbers = [
            getattr(ds, "InstanceNumber", 0) for ds in result.datasets
        ]
        assert instance_numbers == [1, 2, 3, 4, 5]

    def test_sort_with_disordered_dicom_dir(
        self, spatial_sort_engine, dicom_dir_with_spatial_disorder
    ):
        from core.dicom_loader import DicomLoader
        from config import CONFIG
        loader   = DicomLoader(CONFIG.dicom)
        datasets = loader.load_directory(dicom_dir_with_spatial_disorder)
        result   = spatial_sort_engine.sort(datasets)
        assert len(result.datasets) == 5
        z = result.z_positions_mm
        diffs = [z[i+1] - z[i] for i in range(len(z)-1)]
        assert all(d >= 0 for d in diffs), "Not monotonic after sort: %s" % z

    def test_sort_result_has_correct_fields(self, spatial_sort_engine, temp_dicom_dir):
        from core.dicom_loader import DicomLoader
        from config import CONFIG
        loader   = DicomLoader(CONFIG.dicom)
        datasets = loader.load_directory(temp_dicom_dir)
        result   = spatial_sort_engine.sort(datasets)
        assert isinstance(result.is_ascending, bool)
        assert isinstance(result.has_gaps, bool)
        assert isinstance(result.has_duplicates, bool)
        assert isinstance(result.warnings, list)

    def test_sort_result_to_dict_serializable(self, spatial_sort_engine, temp_dicom_dir):
        from core.dicom_loader import DicomLoader
        from config import CONFIG
        loader   = DicomLoader(CONFIG.dicom)
        datasets = loader.load_directory(temp_dicom_dir)
        result   = spatial_sort_engine.sort(datasets)
        d = result.to_dict()
        json.dumps(d)  # must be JSON-serializable

    def test_sort_empty_list_raises(self, spatial_sort_engine):
        from core.spatial_sort import SpatialSortError
        with pytest.raises(SpatialSortError):
            spatial_sort_engine.sort([])

    def test_sort_all_identical_instance_numbers_falls_back(self):
        """GE firmware bug: all InstanceNumber=0 must trigger fallback."""
        from core.spatial_sort import SpatialSortEngine
        from pydicom.dataset import Dataset
        engine = SpatialSortEngine(preferred_tag="InstanceNumber")
        datasets = []
        for i in range(3):
            ds = Dataset()
            ds.SOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"
            ds.Modality       = "CT"
            ds.InstanceNumber = 0       # GE bug: all zero
            ds.SliceLocation  = float(i * 3)  # correct spatial info
            datasets.append(ds)
        result = engine.sort(datasets)
        # Must have fallen back to SliceLocation or filename
        assert result.sort_method != "InstanceNumber"
        assert len(result.warnings) > 0

    def test_duplicate_z_detection(self):
        """Two slices at same z-position should be flagged."""
        from core.spatial_sort import SpatialSortEngine
        from pydicom.dataset import Dataset
        engine = SpatialSortEngine(preferred_tag="SliceLocation", strict=False)
        datasets = []
        for z in [0.0, 3.0, 3.0, 6.0]:  # duplicate at z=3.0
            ds = Dataset()
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ds.Modality    = "CT"
            ds.SliceLocation = z
            ds.InstanceNumber = int(z)
            datasets.append(ds)
        result = engine.sort(datasets)
        assert result.has_duplicates is True
        assert len(result.duplicate_indices) > 0


# ═══════════════════════════════════════════════════════════════════════════
# DOSE METADATA EXTRACTOR TESTS (Step 5.4)
# ═══════════════════════════════════════════════════════════════════════════

class TestDoseMetadataExtractor:

    @pytest.fixture
    def siemens_profile(self, scanner_profile_registry):
        return scanner_profile_registry.get_profile("siemens_somatom_gosim")

    @pytest.fixture
    def ge_profile(self, scanner_profile_registry):
        return scanner_profile_registry.get_profile("ge_discovery_rt")

    @pytest.fixture
    def siemens_extractor(self, siemens_profile):
        from core.dose_metadata_extractor import DoseMetadataExtractor
        return DoseMetadataExtractor(siemens_profile)

    @pytest.fixture
    def ge_extractor(self, ge_profile):
        from core.dose_metadata_extractor import DoseMetadataExtractor
        return DoseMetadataExtractor(ge_profile)

    def test_extracts_ctdivol_from_standard_tag(
        self, siemens_extractor, synthetic_siemens_ct_dicom
    ):
        meta = siemens_extractor.extract(synthetic_siemens_ct_dicom)
        assert meta.has_ctdi_vol is True
        assert meta.ctdi_vol_mgy == pytest.approx(14.2, abs=0.01)
        assert meta.source == "standard_tag"
        assert meta.confidence == "high"

    def test_siemens_missing_ctdivol_returns_none(self, siemens_extractor):
        from pydicom.dataset import Dataset
        ds = Dataset()
        ds.Modality = "CT"
        # No CTDIvol tag
        meta = siemens_extractor.extract(ds)
        assert meta.has_ctdi_vol is False
        assert meta.ctdi_vol_mgy is None
        assert meta.source == "none"

    def test_ge_extracts_ctdivol_standard_tag(self, ge_extractor, synthetic_ge_ct_dicom):
        """GE dataset with standard CTDIvol tag present."""
        from pydicom.dataset import Dataset
        ds = Dataset()
        ds.Modality  = "CT"
        ds.CTDIvol   = 12.5   # standard tag present
        meta = ge_extractor.extract(ds)
        assert meta.has_ctdi_vol is True
        assert meta.ctdi_vol_mgy == pytest.approx(12.5, abs=0.01)

    def test_dose_metadata_to_dict_serializable(
        self, siemens_extractor, synthetic_siemens_ct_dicom
    ):
        meta = siemens_extractor.extract(synthetic_siemens_ct_dicom)
        d = meta.to_dict()
        assert isinstance(d, dict)
        json.dumps(d, default=str)  # must be JSON-serializable

    def test_dose_metadata_has_ctdi_vol_property(
        self, siemens_extractor, synthetic_siemens_ct_dicom
    ):
        meta = siemens_extractor.extract(synthetic_siemens_ct_dicom)
        assert meta.has_ctdi_vol is True

    def test_cross_validate_consistent(self, siemens_extractor):
        from core.dose_metadata_extractor import DoseMetadata
        m1 = DoseMetadata(12.5, None, "standard_tag", "high")
        m2 = DoseMetadata(12.4, None, "rdsr", "high")
        cv = siemens_extractor.cross_validate(m1, m2, tolerance_pct=5.0)
        assert cv.consistent is True
        assert cv.discrepancy_pct < 5.0

    def test_cross_validate_inconsistent(self, siemens_extractor):
        from core.dose_metadata_extractor import DoseMetadata
        m1 = DoseMetadata(12.5, None, "standard_tag", "high")
        m2 = DoseMetadata(15.0, None, "rdsr", "high")
        cv = siemens_extractor.cross_validate(m1, m2, tolerance_pct=5.0)
        assert cv.consistent is False
        assert cv.warning_message is not None
        assert cv.discrepancy_pct > 5.0

    def test_cross_validate_one_source_missing(self, siemens_extractor):
        from core.dose_metadata_extractor import DoseMetadata
        m1 = DoseMetadata(12.5, None, "standard_tag", "high")
        m2 = DoseMetadata(None, None, "none", "unavailable")
        cv = siemens_extractor.cross_validate(m1, m2)
        assert cv.consistent is False
        assert cv.discrepancy_pct is None

    def test_cross_validate_to_dict_serializable(self, siemens_extractor):
        from core.dose_metadata_extractor import DoseMetadata
        m1 = DoseMetadata(12.5, None, "standard_tag", "high")
        m2 = DoseMetadata(12.4, None, "rdsr", "high")
        cv = siemens_extractor.cross_validate(m1, m2)
        json.dumps(cv.to_dict(), default=str)
