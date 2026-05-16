# -*- coding: utf-8 -*-
"""tests/test_dicom_loader.py — DICOM loader tests."""

from pathlib import Path

import numpy as np
import pytest

from core.dicom_loader import (
    DicomLoader, DicomLoadError, DicomMetadata, SliceRangeError,
)
from config import CONFIG


class TestIsCTImage:
    def test_synthetic_is_ct(self, dicom_loader_instance, synthetic_ct_dicom):
        assert dicom_loader_instance.is_ct_image(synthetic_ct_dicom) is True

    def test_synthetic_is_not_localizer(self, dicom_loader_instance, synthetic_ct_dicom):
        assert dicom_loader_instance.is_localizer(synthetic_ct_dicom) is False


class TestHUConversion:
    def test_hu_array_dtype(self, dicom_loader_instance, synthetic_ct_dicom):
        hu = dicom_loader_instance.to_hu_array(synthetic_ct_dicom)
        assert hu.dtype == np.float32

    def test_hu_array_shape(self, dicom_loader_instance, synthetic_ct_dicom):
        hu = dicom_loader_instance.to_hu_array(synthetic_ct_dicom)
        assert hu.shape == (512, 512)

    def test_central_hu_near_zero(self, dicom_loader_instance, synthetic_ct_dicom):
        """A water-only region (far from inserts) should be near 0 HU."""
        hu = dicom_loader_instance.to_hu_array(synthetic_ct_dicom)
        # Use rows 400-420, cols 400-420 — far from acrylic center (256,256)
        # and far from air insert (100,100)
        region = hu[400:420, 400:420]
        assert abs(float(np.mean(region))) < 5.0


class TestMetadataExtraction:
    def test_kvp(self, dicom_loader_instance, synthetic_ct_dicom):
        meta = dicom_loader_instance.extract_metadata(synthetic_ct_dicom)
        assert meta.kvp == 120.0

    def test_ctdi_vol(self, dicom_loader_instance, synthetic_ct_dicom):
        meta = dicom_loader_instance.extract_metadata(synthetic_ct_dicom)
        assert meta.ctdi_vol == 12.5

    def test_returns_dicom_metadata(self, dicom_loader_instance, synthetic_ct_dicom):
        meta = dicom_loader_instance.extract_metadata(synthetic_ct_dicom)
        assert isinstance(meta, DicomMetadata)


class TestFileLoading:
    def test_load_saved_file(self, dicom_loader_instance, saved_dicom_file):
        ds = dicom_loader_instance.load_file(saved_dicom_file)
        assert ds is not None

    def test_load_nonexistent_raises(self, dicom_loader_instance):
        with pytest.raises(DicomLoadError):
            dicom_loader_instance.load_file(Path("nonexistent.dcm"))


class TestDirectoryLoading:
    def test_load_directory_count(self, dicom_loader_instance, temp_dicom_dir):
        datasets = dicom_loader_instance.load_directory(temp_dicom_dir)
        assert len(datasets) == 5

    def test_load_directory_sorted(self, dicom_loader_instance, temp_dicom_dir):
        datasets = dicom_loader_instance.load_directory(temp_dicom_dir)
        instance_numbers = [int(ds.InstanceNumber) for ds in datasets]
        assert instance_numbers == [1, 2, 3, 4, 5]


class TestPixelSpacing:
    def test_pixel_spacing(self, dicom_loader_instance, synthetic_ct_dicom):
        spacing = dicom_loader_instance.get_pixel_spacing_mm(synthetic_ct_dicom)
        assert spacing == pytest.approx((0.977, 0.977))


class TestSliceInventory:
    def test_inventory_count(self, dicom_loader_instance, temp_dicom_dir):
        inv = dicom_loader_instance.get_slice_inventory(temp_dicom_dir)
        assert len(inv) == 5

    def test_inventory_keys(self, dicom_loader_instance, temp_dicom_dir):
        inv = dicom_loader_instance.get_slice_inventory(temp_dicom_dir)
        for entry in inv:
            assert "index" in entry
            assert "instance_number" in entry
            assert "file_path" in entry


class TestSliceRange:
    def test_range_1_to_3(self, dicom_loader_instance, temp_dicom_dir):
        datasets = dicom_loader_instance.load_slice_range(temp_dicom_dir, 1, 3)
        assert len(datasets) == 3

    def test_range_2_to_4_instance_numbers(self, dicom_loader_instance, temp_dicom_dir):
        datasets = dicom_loader_instance.load_slice_range(temp_dicom_dir, 2, 4)
        nums = [int(ds.InstanceNumber) for ds in datasets]
        assert nums == [2, 3, 4]

    def test_range_all(self, dicom_loader_instance, temp_dicom_dir):
        datasets = dicom_loader_instance.load_slice_range(temp_dicom_dir, 1, 5)
        assert len(datasets) == 5

    def test_range_start_zero_raises(self, dicom_loader_instance, temp_dicom_dir):
        with pytest.raises(SliceRangeError):
            dicom_loader_instance.load_slice_range(temp_dicom_dir, 0, 3)

    def test_range_start_gt_end_raises(self, dicom_loader_instance, temp_dicom_dir):
        with pytest.raises(SliceRangeError):
            dicom_loader_instance.load_slice_range(temp_dicom_dir, 4, 2)

    def test_range_exceeds_total_raises(self, dicom_loader_instance, temp_dicom_dir):
        with pytest.raises(SliceRangeError):
            dicom_loader_instance.load_slice_range(temp_dicom_dir, 1, 99)
