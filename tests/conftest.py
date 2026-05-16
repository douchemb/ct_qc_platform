# -*- coding: utf-8 -*-
"""
tests/conftest.py — Shared pytest fixtures for CT QC Platform tests.

Provides synthetic DICOM datasets, temporary directories, and
pre-configured analyzer instances for all test modules.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom.uid
import pytest

from config import CONFIG
from core.dicom_loader import DicomLoader
from modules.image_qc.roi_stats import (
    PhantomROIAnalyzer, ROIDescriptor, VolumetricQCResult,
)


@pytest.fixture
def synthetic_ct_dicom(tmp_path) -> Dataset:
    """Creates a minimal valid in-memory CT Dataset.

    512×512 uint16 pixel array:
    - Base: 1024 everywhere (maps to 0 HU with RescaleIntercept=-1024)
    - Gaussian noise: std=5 HU
    - Circular acrylic insert at center: radius 60 px → 1144 (+120 HU)
    - Air insert at (100, 100): radius 20 px → 24 (-1000 HU)
    """
    filename = str(tmp_path / "synthetic.dcm")
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    # DICOM tags
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.Modality = "CT"
    ds.SeriesDescription = "QA_HEAD_STD"
    ds.AcquisitionDate = "20240115"
    ds.PatientID = "QA_PHANTOM_001"
    ds.InstanceNumber = 1
    ds.KVP = 120.0
    ds.XRayTubeCurrent = 200.0
    ds.ExposureTime = 500
    ds.SliceThickness = 3.0
    ds.PixelSpacing = [0.977, 0.977]
    ds.CTDIvol = 12.5
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Build pixel array
    rng = np.random.default_rng(42)
    # Start with water background at 1024 (maps to 0 HU)
    base = np.full((512, 512), 1024, dtype=np.float64)

    # Add Gaussian noise: std=5 HU simulates realistic CT noise
    noise = rng.normal(0, 5, (512, 512))
    pixels = base + noise

    # Acrylic insert at center: base 1144 + noise (maps to +120 HU)
    cy, cx = 256, 256
    yy, xx = np.ogrid[:512, :512]
    acrylic_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= 60 ** 2
    pixels[acrylic_mask] = 1144 + noise[acrylic_mask]

    # Air insert at (100, 100): base 24 + noise (maps to -1000 HU)
    air_mask = ((yy - 100) ** 2 + (xx - 100) ** 2) <= 20 ** 2
    pixels[air_mask] = 24 + noise[air_mask]

    # Clip to uint16 range [0, 65535] before storing
    pixels = np.clip(pixels, 0, 65535).astype(np.uint16)
    ds.PixelData = pixels.tobytes()

    return ds


@pytest.fixture
def saved_dicom_file(synthetic_ct_dicom, tmp_path) -> Path:
    """Saves synthetic_ct_dicom to disk and returns the path."""
    path = tmp_path / "test_slice.dcm"
    synthetic_ct_dicom.save_as(str(path))
    return path


@pytest.fixture
def temp_dicom_dir(tmp_path) -> Path:
    """Creates 5 CT Dataset copies in tmp_path with varying metadata."""
    base_date = datetime(2024, 1, 15)

    for i in range(1, 6):
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        filename = str(tmp_path / f"slice_{i:03d}.dcm")
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)

        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.Modality = "CT"
        ds.SeriesDescription = "QA_HEAD_STD"
        ds.PatientID = "QA_PHANTOM_001"
        ds.InstanceNumber = i
        ds.KVP = 120.0
        ds.XRayTubeCurrent = 200.0
        ds.ExposureTime = 500
        ds.SliceThickness = 3.0
        ds.PixelSpacing = [0.977, 0.977]
        ds.CTDIvol = 12.5
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.Rows = 512
        ds.Columns = 512
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        # Increment date by 30 days per slice
        acq_date = base_date + timedelta(days=30 * (i - 1))
        ds.AcquisitionDate = acq_date.strftime("%Y%m%d")

        # Slice location: 3mm apart
        ds.SliceLocation = float(i * 3.0)

        # Build pixel array (same pattern, different seed per slice for noise)
        rng = np.random.default_rng(42 + i)
        base = np.full((512, 512), 1024, dtype=np.float64)
        noise = rng.normal(0, 5, (512, 512))
        pixels = base + noise

        # Acrylic insert at center with noise
        yy, xx = np.ogrid[:512, :512]
        acrylic_mask = ((yy - 256) ** 2 + (xx - 256) ** 2) <= 60 ** 2
        pixels[acrylic_mask] = 1144 + noise[acrylic_mask]

        # Air insert with noise
        air_mask = ((yy - 100) ** 2 + (xx - 100) ** 2) <= 20 ** 2
        pixels[air_mask] = 24 + noise[air_mask]

        pixels = np.clip(pixels, 0, 65535).astype(np.uint16)
        ds.PixelData = pixels.tobytes()
        ds.save_as(filename)

    return tmp_path


@pytest.fixture
def standard_rois() -> list[ROIDescriptor]:
    """Returns 5 ROIDescriptor objects for a 512×512 image."""
    return [
        ROIDescriptor("center_water", 226, 226, 60, 60),
        ROIDescriptor("peripheral_12", 80, 236, 40, 40),
        ROIDescriptor("peripheral_3", 236, 390, 40, 40),
        ROIDescriptor("peripheral_6", 390, 236, 40, 40),
        ROIDescriptor("peripheral_9", 236, 80, 40, 40),
    ]


@pytest.fixture
def dicom_loader_instance() -> DicomLoader:
    """Returns DicomLoader configured with CONFIG.dicom."""
    return DicomLoader(config=CONFIG.dicom)


@pytest.fixture
def roi_analyzer_instance(dicom_loader_instance) -> PhantomROIAnalyzer:
    """Returns PhantomROIAnalyzer with injected DicomLoader."""
    return PhantomROIAnalyzer(dicom_loader=dicom_loader_instance, config=CONFIG.image_qc)


@pytest.fixture
def volumetric_result(roi_analyzer_instance, temp_dicom_dir, standard_rois) -> VolumetricQCResult:
    """Pre-computed VolumetricQCResult covering all 5 slices.

    Shared across Phase 2 test modules to avoid re-running analysis.
    """
    return roi_analyzer_instance.analyze_volume(
        temp_dicom_dir, standard_rois, start_slice=1, end_slice=5,
    )


# ═══════════════════════════════════════════════════════════════════
# Phase 3 — Dosimetry Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_localizer_dicom(tmp_path) -> Dataset:
    """Creates a minimal valid CT localizer Dataset.

    512×256 pixel array (512 rows = axial positions, 256 cols = transverse):
    - Air reference rows 0–49 and 460–511: uniform value 4000
    - Patient rows 50–459: Gaussian attenuation profile per row
    - Stored as uint16
    """
    filename = str(tmp_path / "localizer.dcm")
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2.1"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2.1"
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.Modality = "CT"
    ds.SeriesDescription = "CT_LOCALIZER_AP"
    ds.AcquisitionDate = "20240115"
    ds.PatientID = "QA_PHANTOM_001"
    ds.InstanceNumber = 1
    ds.KVP = 120.0
    ds.PixelSpacing = [1.0, 1.0]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.Rows = 512
    ds.Columns = 256
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    # Build pixel array
    pixels = np.zeros((512, 256), dtype=np.float64)

    # Air reference rows: uniform 4000 (maximum transmission)
    pixels[0:50, :] = 4000.0
    pixels[460:512, :] = 4000.0

    # Patient rows 50–459: Gaussian attenuation profile
    # I(j) = 4000 × exp(-0.15 × t(j))
    # where t(j) = 20 × exp(-((j-128)/40)²) — simulates patient cross-section
    j_cols = np.arange(256)
    for i in range(50, 460):
        t_j = 20.0 * np.exp(-((j_cols - 128.0) / 40.0) ** 2)
        pixels[i, :] = 4000.0 * np.exp(-0.15 * t_j)

    # Clip to uint16 range and store
    pixels = np.clip(pixels, 0, 65535).astype(np.uint16)
    ds.PixelData = pixels.tobytes()

    return ds


@pytest.fixture
def saved_localizer_file(synthetic_localizer_dicom, tmp_path) -> Path:
    """Saves synthetic_localizer_dicom to disk. Returns path."""
    path = tmp_path / "localizer.dcm"
    synthetic_localizer_dicom.save_as(str(path))
    return path


@pytest.fixture
def localizer_parser_instance(dicom_loader_instance):
    """Returns LocalizerParser configured with CONFIG.dosimetry."""
    from modules.dosimetry.localizer_parser import LocalizerParser
    return LocalizerParser(dicom_loader=dicom_loader_instance, config=CONFIG.dosimetry)


@pytest.fixture
def parsed_localizer_data(localizer_parser_instance, saved_localizer_file):
    """Returns parsed LocalizerData from the synthetic localizer."""
    return localizer_parser_instance.parse(saved_localizer_file)


@pytest.fixture
def dw_calculator_instance():
    """Returns DwCalculator configured with CONFIG.dosimetry."""
    from modules.dosimetry.dw_calculator import DwCalculator
    return DwCalculator(config=CONFIG.dosimetry)


@pytest.fixture
def dw_series_result(dw_calculator_instance, volumetric_result):
    """Returns DwSeriesResult from volumetric_result."""
    return dw_calculator_instance.compute_from_volume(volumetric_result)


@pytest.fixture
def ssde_calculator_instance():
    """Returns SSDECalculator configured with CONFIG.dosimetry."""
    from modules.dosimetry.ssde_calculator import SSDECalculator
    return SSDECalculator(config=CONFIG.dosimetry)


# ═══════════════════════════════════════════════════════════════════
# Phase 4 — Predictive Maintenance Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def populated_archive(tmp_path):
    """Creates a MetricsArchive with 8 synthetic degrading sessions.

    Sessions span 8 months (one per month from 2024-01-15 onward).
    Metrics show realistic degradation trends:
      center_water_std_hu: 3.2 → +0.2/month
      nps_peak_frequency_lpmm: 0.28 → -0.01/month
      mtf_50_lpmm: 0.55 → -0.02/month
      hu_linearity_max_deviation_hu: 1.5 → +0.2/month
      ed_soft_tissue_slope: 1.002 → +0.003/month
      ed_bone_slope: 0.998 → -0.003/month
    """
    import uuid
    from modules.predictive.metrics_archive import MetricsArchive, QCSessionRecord

    archive_path = tmp_path / "test_archive.json"
    archive = MetricsArchive(archive_path)

    for i in range(8):
        month = i + 1
        date_str = "2024-%02d-15" % month
        record = QCSessionRecord(
            session_id=str(uuid.uuid4()),
            session_date=date_str,
            session_timestamp="2024-%02d-15T10:00:00Z" % month,
            scanner_id="TEST_SCANNER",
            operator_id="TEST_OP",
            center_water_mean_hu=0.1,
            center_water_std_hu=3.2 + 0.2 * i,
            center_water_variance_hu=(3.2 + 0.2 * i) ** 2,
            nps_peak_frequency_lpmm=0.28 - 0.01 * i,
            nps_peak_value_hu2mm2=0.5,
            nps_integral_hu2mm2=1.2,
            mtf_50_lpmm=0.55 - 0.02 * i,
            mtf_10_lpmm=0.25,
            hu_linearity_max_deviation_hu=1.5 + 0.2 * i,
            hu_linearity_r_squared=0.998,
            ed_soft_tissue_slope=1.002 + 0.003 * i,
            ed_bone_slope=0.998 - 0.003 * i,
            ed_max_red_deviation=0.01,
            ctdi_vol_mgy=12.5,
            ssde_mean_mgy=14.2,
            dw_mean_cm=23.4,
            effective_dose_msv=5.1,
            all_image_qc_passed=True,
            all_dosimetry_computed=True,
            ed_calibration_passed=True,
            notes="Synthetic session %d" % (i + 1),
            schema_version="1.0",
        )
        archive.append_session(record)

    return archive


@pytest.fixture
def fitted_trend_results(populated_archive):
    """Returns dict[str, TrendModelResult] from fit_all_metrics."""
    from modules.predictive.trend_model import QCTrendModel
    model = QCTrendModel(CONFIG.predictive)
    return model.fit_all_metrics(populated_archive)


@pytest.fixture
def maintenance_alert_fixture(fitted_trend_results, populated_archive):
    """Returns MaintenanceAlert from generate_maintenance_alert."""
    from modules.predictive.failure_predictor import FailurePredictor
    predictor = FailurePredictor(CONFIG.predictive)
    return predictor.generate_maintenance_alert(
        fitted_trend_results, populated_archive, "TEST_SCANNER",
    )


# ═══════════════════════════════════════════════════════════════════
# Phase 5 — GE Discovery RT / Helios Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_ge_ct_dicom(tmp_path) -> Dataset:
    """
    Simulates a GE Discovery RT CT image with GE-specific DICOM quirks:
    - Standard RescaleSlope (0028,1053) is ABSENT
    - GE private tag (0009,100D) carries slope = 1.0
    - SliceThickness written as string "3.0 mm" (GE firmware quirk)
    - KVP written as string "120.0" (some GE versions)
    - InstanceNumber = 0 (GE firmware bug affecting some series)
    - CTDIvol is absent from standard tag
    """
    import math
    from pydicom.dataset import FileMetaDataset

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

    ds.SOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.Modality       = "CT"
    ds.Manufacturer   = "GE MEDICAL SYSTEMS"
    ds.ManufacturerModelName = "Discovery RT"

    # GE quirk: NO standard RescaleSlope or RescaleIntercept

    # GE private tag (0009,100D) carries the slope
    block = ds.private_block(0x0009, "GE_CT_NEXT", create=True)
    block.add_new(0x0D, "DS", "1.0")   # slope = 1.0

    # GE quirk: SliceThickness as string with unit — use LO VR to bypass pydicom DS validation
    ds.add_new(0x00180050, "LO", "3.0 mm")

    # GE quirk: KVP as string — use LO VR
    ds.add_new(0x00180060, "LO", "120.0")

    # GE quirk: InstanceNumber = 0
    ds.InstanceNumber     = 0
    ds.AcquisitionDate    = "20240115"
    ds.SeriesDescription  = "GE_HELIOS_QA"
    ds.PatientID          = "GE_PHANTOM_001"
    ds.PixelSpacing       = [0.977, 0.977]
    ds.SliceLocation      = 0.0
    ds.XRayTubeCurrent    = 200
    ds.ExposureTime       = 500
    ds.BitsAllocated      = 16
    ds.BitsStored         = 16
    ds.HighBit            = 15
    ds.PixelRepresentation = 0
    ds.Rows               = 512
    ds.Columns            = 512
    ds.SamplesPerPixel    = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Synthetic 512×512 phantom image: water background + 2 inserts
    rng  = np.random.default_rng(42)
    arr  = np.full((512, 512), 1024, dtype=np.uint16)  # water: 1024 → 0 HU
    noise = rng.integers(-5, 6, (512, 512), dtype=np.int16)
    arr = (arr.astype(np.int32) + noise).clip(0, 65535).astype(np.uint16)

    # Air insert at top (angle 0°, radius ~59 px from center)
    rr, cc = 256 - 59, 256
    for i in range(max(0, rr - 12), min(512, rr + 12)):
        for j in range(max(0, cc - 12), min(512, cc + 12)):
            if (i - rr)**2 + (j - cc)**2 < 144:
                arr[i, j] = 24     # -1000 HU after rescale

    # Teflon insert (angle 300°)
    angle_rad = math.radians(300)
    tr = int(256 - 59 * math.cos(angle_rad))
    tc = int(256 + 59 * math.sin(angle_rad))
    for i in range(max(0, tr - 12), min(512, tr + 12)):
        for j in range(max(0, tc - 12), min(512, tc + 12)):
            if (i - tr)**2 + (j - tc)**2 < 144:
                arr[i, j] = 2014   # +990 HU after rescale

    ds.PixelData = arr.tobytes()
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    return ds


@pytest.fixture
def saved_ge_dicom_file(synthetic_ge_ct_dicom, tmp_path) -> Path:
    """Saves synthetic_ge_ct_dicom to disk. Returns the path."""
    path = tmp_path / "ge_test_slice.dcm"
    synthetic_ge_ct_dicom.save_as(str(path), write_like_original=False)
    return path


@pytest.fixture
def synthetic_ge_localizer_dicom() -> Dataset:
    """
    Creates a localizer dataset using GE's non-standard method:
    - SOPClassUID = standard CT UID (NOT the localizer UID)
    - ImageType = ["ORIGINAL", "PRIMARY", "LOCALIZER"]
    - SeriesDescription = "Topogram 0.6"
    """
    from pydicom.dataset import FileMetaDataset

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

    ds.SOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"  # NOT the localizer UID
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.Modality       = "CT"
    ds.ImageType      = ["ORIGINAL", "PRIMARY", "LOCALIZER"]
    ds.SeriesDescription = "Topogram 0.6"
    ds.AcquisitionDate   = "20240115"
    ds.PatientID         = "GE_PHANTOM_001"
    ds.InstanceNumber    = 1
    ds.KVP               = 120.0
    ds.PixelSpacing      = [1.0, 1.0]
    ds.RescaleSlope      = 1.0
    ds.RescaleIntercept  = 0.0
    ds.Rows              = 512
    ds.Columns           = 256
    ds.BitsAllocated     = 16
    ds.BitsStored        = 16
    ds.HighBit           = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel     = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Simple localizer pixel data
    pixels = np.zeros((512, 256), dtype=np.float64)
    pixels[0:50, :] = 4000.0
    pixels[460:512, :] = 4000.0
    j_cols = np.arange(256)
    for i in range(50, 460):
        t_j = 20.0 * np.exp(-((j_cols - 128.0) / 40.0) ** 2)
        pixels[i, :] = 4000.0 * np.exp(-0.15 * t_j)
    pixels = np.clip(pixels, 0, 65535).astype(np.uint16)
    ds.PixelData = pixels.tobytes()
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    return ds


@pytest.fixture
def helios_detector_instance():
    """Returns a configured HeliosPhantomDetector instance."""
    from modules.image_qc.phantom_geometry import HeliosPhantomDetector
    return HeliosPhantomDetector(CONFIG)


@pytest.fixture
def synthetic_helios_hu_array():
    """
    Returns (hu_array, pixel_spacing_mm) simulating a Helios phantom cross-section.
    - Background: -1000 HU (air outside phantom)
    - Phantom body (radius ~102 px): 0 HU (water)
    - Center at (258, 255) — deliberately off-center
    - Air insert at top: -1000 HU
    - Teflon insert at 300°: +990 HU
    - Pixel spacing: (0.977, 0.977) mm
    """
    import math

    pixel_spacing_mm = (0.977, 0.977)
    hu_array = np.full((512, 512), -1000.0, dtype=np.float32)

    cx, cy = 258, 255  # deliberately off-center
    phantom_radius_px = int(100.0 / pixel_spacing_mm[0])  # ~102 px

    # Phantom body: water = 0 HU
    yy, xx = np.ogrid[:512, :512]
    body_mask = ((yy - cx)**2 + (xx - cy)**2) <= phantom_radius_px**2
    hu_array[body_mask] = 0.0

    # Air insert at top (angle 0°)
    insert_r_px = int(58.0 / pixel_spacing_mm[0])  # ~59 px
    air_row = cx - insert_r_px
    air_col = cy
    air_mask = ((yy - air_row)**2 + (xx - air_col)**2) <= 12**2
    hu_array[air_mask] = -1000.0

    # Teflon insert at 300°
    angle_rad = math.radians(300)
    tef_row = int(cx - insert_r_px * math.cos(angle_rad))
    tef_col = int(cy + insert_r_px * math.sin(angle_rad))
    tef_mask = ((yy - tef_row)**2 + (xx - tef_col)**2) <= 12**2
    hu_array[tef_mask] = 990.0

    return hu_array, pixel_spacing_mm


# ═══════════════════════════════════════════════════════════════════
# Phase 5 — Multi-Hardware & Multi-Phantom Integration Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def scanner_profile_registry():
    """ScannerProfileRegistry loaded from scanner_profiles.yaml."""
    from core.scanner_profiles import ScannerProfileRegistry
    return ScannerProfileRegistry()


@pytest.fixture
def phantom_adapter_factory():
    """PhantomAdapterFactory loaded from phantom_profiles.yaml."""
    from modules.image_qc.phantom_adapters import PhantomAdapterFactory
    return PhantomAdapterFactory()


@pytest.fixture
def synthetic_siemens_ct_dicom():
    """
    Synthetic CT dataset simulating a Siemens SOMATOM go.Sim.
    Has standard DICOM tags (no private GE tags).
    """
    from pydicom.dataset import FileMetaDataset

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

    ds.SOPClassUID           = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID        = pydicom.uid.generate_uid()
    ds.Modality              = "CT"
    ds.Manufacturer          = "SIEMENS"
    ds.ManufacturerModelName = "SOMATOM go.Sim"
    ds.SeriesDescription     = "Water Phantom QA"
    ds.ProtocolName          = "QA"
    ds.PatientID             = "SIEMENS_QA_001"
    ds.AcquisitionDate       = "20240201"
    ds.InstanceNumber        = 1
    ds.SliceLocation         = 0.0
    ds.SliceThickness        = 3.0
    ds.KVP                   = 120.0
    ds.RescaleSlope          = 1.0
    ds.RescaleIntercept      = -1024.0
    ds.PixelSpacing          = [0.977, 0.977]
    ds.CTDIvol               = 14.2   # standard tag present on Siemens
    ds.BitsAllocated         = 16
    ds.BitsStored            = 16
    ds.HighBit               = 15
    ds.PixelRepresentation   = 0
    ds.Rows                  = 512
    ds.Columns               = 512
    ds.SamplesPerPixel       = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Synthetic 512×512 water phantom image
    rng = np.random.default_rng(42)
    arr = np.full((512, 512), 1024, dtype=np.uint16)
    arr = arr + rng.integers(-5, 6, (512, 512), dtype=np.int16).astype(np.uint16)
    ds.PixelData = arr.tobytes()
    ds.is_implicit_VR   = False
    ds.is_little_endian = True
    return ds


@pytest.fixture
def dicom_dir_with_spatial_disorder(tmp_path):
    """
    Creates a directory of 5 CT slices with deliberately scrambled
    InstanceNumber values to test spatial sort robustness.
    SliceLocation values are set correctly for validation.
    """
    instance_order = [3, 1, 5, 2, 4]  # scrambled

    for idx, inst in enumerate(instance_order):
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

        filename = str(tmp_path / ("slice_%03d.dcm" % idx))
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)

        ds.SOPClassUID           = "1.2.840.10008.5.1.4.1.1.2"
        ds.SOPInstanceUID        = pydicom.uid.generate_uid()
        ds.Modality              = "CT"
        ds.Manufacturer          = "SIEMENS"
        ds.ManufacturerModelName = "SOMATOM go.Sim"
        ds.SeriesDescription     = "QA_HEAD_STD"
        ds.PatientID             = "QA_PHANTOM_001"
        ds.AcquisitionDate       = "20240115"
        ds.InstanceNumber        = inst         # scrambled
        ds.SliceLocation         = float(idx * 3)  # correct spatial order by index
        ds.ImagePositionPatient  = [0.0, 0.0, float(idx * 3)]
        ds.SliceThickness        = 3.0
        ds.KVP                   = 120.0
        ds.RescaleSlope          = 1.0
        ds.RescaleIntercept      = -1024.0
        ds.PixelSpacing          = [0.977, 0.977]
        ds.BitsAllocated         = 16
        ds.BitsStored            = 16
        ds.HighBit               = 15
        ds.PixelRepresentation   = 0
        ds.Rows                  = 512
        ds.Columns               = 512
        ds.SamplesPerPixel       = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        rng = np.random.default_rng(42 + idx)
        base = np.full((512, 512), 1024, dtype=np.float64)
        noise = rng.normal(0, 5, (512, 512))
        pixels = np.clip(base + noise, 0, 65535).astype(np.uint16)
        ds.PixelData = pixels.tobytes()
        ds.save_as(filename)

    return tmp_path


@pytest.fixture
def spatial_sort_engine():
    from core.spatial_sort import SpatialSortEngine
    return SpatialSortEngine(preferred_tag="SliceLocation")


# ═══════════════════════════════════════════════════════════════════
# Phase 6 — Comprehensive Metrics Engine Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def basic_metrics_engine():
    from modules.image_qc.basic_metrics import BasicMetricsEngine
    return BasicMetricsEngine()


@pytest.fixture
def siemens_water_adapter():
    from modules.image_qc.phantom_adapters import SiemensWaterPhantomAdapter
    return SiemensWaterPhantomAdapter()


@pytest.fixture
def helios_adapter():
    from modules.image_qc.phantom_adapters import HeliosQAPhantomAdapter
    return HeliosQAPhantomAdapter()


@pytest.fixture
def advanced_metrics_engine():
    """AdvancedMetricsEngine with no calculators injected — for routing tests."""
    from modules.image_qc.advanced_metrics_engine import AdvancedMetricsEngine
    return AdvancedMetricsEngine()


@pytest.fixture
def advanced_metrics_engine_full():
    """AdvancedMetricsEngine with all calculators injected."""
    from modules.image_qc.advanced_metrics_engine import AdvancedMetricsEngine
    from modules.image_qc.nps_calculator import NPSCalculator
    from modules.image_qc.mtf_calculator import MTFCalculator
    from modules.image_qc.hu_linearity import HULinearityAnalyzer
    from modules.dosimetry.dw_calculator import DwCalculator
    from modules.dosimetry.ssde_calculator import SSDECalculator
    from config import CONFIG
    return AdvancedMetricsEngine(
        nps_calculator  = NPSCalculator(CONFIG.image_qc),
        mtf_calculator  = MTFCalculator(CONFIG.image_qc),
        hu_analyzer     = HULinearityAnalyzer(CONFIG.image_qc),
        dw_calculator   = DwCalculator(CONFIG.dosimetry),
        ssde_calculator = SSDECalculator(CONFIG.dosimetry),
    )


@pytest.fixture
def metadata_miner_siemens(scanner_profile_registry):
    from core.metadata_miner import MetadataMiner
    profile = scanner_profile_registry.get_profile("siemens_somatom_gosim")
    return MetadataMiner(profile)


@pytest.fixture
def volumetric_result_siemens():
    """Synthetic VolumetricQCResult for 5 Siemens CT slices — built directly."""
    import numpy as np
    from modules.image_qc.roi_stats import (
        VolumetricQCResult, VolumetricROIStat,
        SliceAnalysisResult, ROIStatistics,
    )
    from core.dicom_loader import DicomMetadata

    rng = np.random.default_rng(42)
    n_slices = 5
    roi_labels = ["center", "peripheral_12", "peripheral_3", "peripheral_6", "peripheral_9"]
    hu_arrays = []
    slice_results = []

    for i in range(n_slices):
        # Synthetic HU array: water (~0 HU) with small noise
        arr = rng.normal(loc=0.0, scale=3.0, size=(512, 512)).astype(np.float32)
        hu_arrays.append(arr)

        metadata = DicomMetadata(
            sop_instance_uid=f"1.2.3.4.{i}",
            series_description="Water Phantom QA",
            acquisition_date="20240201",
            patient_id="SIEMENS_QA_001",
            kvp=120.0, mas=200.0,
            slice_thickness_mm=3.0,
            pixel_spacing_mm=(0.977, 0.977),
            rescale_slope=1.0, rescale_intercept=-1024.0,
            instance_number=i + 1,
            slice_location=float((i + 1) * 3),
            image_position_patient=[0.0, 0.0, float((i + 1) * 3)],
            ctdi_vol=14.2,
            reconstruction_kernel="Br40",
            x_ray_tube_current=250.0,
        )

        roi_results = []
        for label in roi_labels:
            mean_hu = float(rng.normal(0.0, 0.5))
            std_hu  = float(rng.uniform(2.5, 3.5))
            roi_results.append(ROIStatistics(
                roi_label=label, slice_file=f"siemens_{i+1:03d}.dcm",
                acquisition_date="20240201",
                mean_hu=mean_hu, std_hu=std_hu,
                variance_hu=std_hu ** 2,
                min_hu=mean_hu - 3 * std_hu, max_hu=mean_hu + 3 * std_hu,
                snr=abs(mean_hu) / max(std_hu, 1e-9),
                skewness=0.01, kurtosis=-0.02,
                n_pixels=2500,
                roi_row_start=200, roi_col_start=200,
                roi_height_px=50, roi_width_px=50,
            ))
        slice_results.append(SliceAnalysisResult(
            metadata=metadata, roi_results=roi_results,
            analysis_timestamp="2024-02-01T12:00:00", source_file=f"siemens_{i+1:03d}.dcm",
        ))

    # Build volumetric stats
    volumetric_stats = {}
    for label in roi_labels:
        means = [sr.get_roi_stat(label).mean_hu for sr in slice_results]
        stds  = [sr.get_roi_stat(label).std_hu  for sr in slice_results]
        vars_ = [sr.get_roi_stat(label).variance_hu for sr in slice_results]
        snrs  = [sr.get_roi_stat(label).snr for sr in slice_results]
        mins  = [sr.get_roi_stat(label).min_hu for sr in slice_results]
        maxs  = [sr.get_roi_stat(label).max_hu for sr in slice_results]

        volumetric_stats[label] = VolumetricROIStat(
            roi_label=label, n_slices=n_slices,
            mean_hu_mean=float(np.mean(means)), mean_hu_std=float(np.std(means)),
            std_hu_mean=float(np.mean(stds)), std_hu_std=float(np.std(stds)),
            variance_hu_mean=float(np.mean(vars_)), variance_hu_std=float(np.std(vars_)),
            min_hu_overall=float(min(mins)), max_hu_overall=float(max(maxs)),
            snr_mean=float(np.mean(snrs)),
            passes_tg66=float(np.mean(stds)) <= 5.0,
        )

    return VolumetricQCResult(
        series_description="Water Phantom QA",
        acquisition_date="20240201",
        start_slice=1, end_slice=5,
        n_slices_selected=5, n_slices_processed=5,
        pixel_spacing_mm=(0.977, 0.977),
        slice_thickness_mm=3.0,
        slice_results=slice_results,
        volumetric_stats=volumetric_stats,
        hu_arrays=hu_arrays,
    )
