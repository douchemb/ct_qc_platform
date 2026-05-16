# -*- coding: utf-8 -*-
"""
core/dicom_loader.py — DICOM Loading Engine & Metadata Extractor.

This is the ONLY file in the entire project that imports pydicom.
All other modules receive either a pydicom.Dataset or a numpy.ndarray —
they never call pydicom.dcmread directly.

Standards:
    - DICOM PS3.3: Information Object Definitions
    - DICOM PS3.4: Service Class Specifications (SOP Class UIDs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
from pydicom.dataset import Dataset

from config import DicomConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Custom Exceptions
# ═══════════════════════════════════════════════════════════════════

class DicomLoadError(Exception):
    """Raised when a DICOM file cannot be read or is structurally invalid."""


class DicomModalityError(DicomLoadError):
    """Raised when a DICOM file is not a CT image."""


class MissingRescaleTagsError(DicomLoadError):
    """
    Raised when RescaleSlope or RescaleIntercept tags are absent.
    HU conversion is impossible without these mandatory tags.
    """


class PixelDataError(DicomLoadError):
    """Raised when pixel data is absent, malformed, or physically implausible."""


class MissingPixelSpacingError(DicomLoadError):
    """Raised when neither PixelSpacing nor ImagerPixelSpacing tags are present."""


class SliceRangeError(ValueError):
    """
    Raised when a requested slice range is invalid.
    Covers: start < 1, start > end, end > total slices.
    """


class InsufficientSlicesError(ValueError):
    """Raised when too few slices are available for a valid analysis."""


# ═══════════════════════════════════════════════════════════════════
# Metadata Dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DicomMetadata:
    """Structured metadata extracted from a CT DICOM dataset.

    Mandatory fields raise during extraction if absent.
    Optional fields have safe defaults and never raise.
    """
    sop_instance_uid: str
    series_description: str
    acquisition_date: str
    patient_id: str
    kvp: float
    mas: float
    slice_thickness_mm: float
    pixel_spacing_mm: tuple[float, float]   # (row_spacing, col_spacing)
    rescale_slope: float
    rescale_intercept: float
    instance_number: int
    slice_location: Optional[float]         # SliceLocation tag in mm, None if absent
    image_position_patient: Optional[list[float]]  # [x, y, z] in mm
    ctdi_vol: Optional[float]
    reconstruction_kernel: Optional[str]
    x_ray_tube_current: float               # mA

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# DICOM Loader Class
# ═══════════════════════════════════════════════════════════════════

class DicomLoader:
    """Central DICOM I/O engine. Loads files, extracts arrays and metadata.

    Parameters
    ----------
    config : DicomConfig
        DICOM-specific configuration from CONFIG.dicom.
    """

    def __init__(self, config: DicomConfig) -> None:
        self._config = config

    # ─────────────────────────────────────────────────────────────
    # File Loading
    # ─────────────────────────────────────────────────────────────

    def load_file(self, path: Path, require_ct: bool = True) -> Dataset:
        """Load a single DICOM file from disk.

        Parameters
        ----------
        path : Path
            Absolute or relative path to the DICOM file.
        require_ct : bool
            If True, raises DicomModalityError for non-CT files.

        Returns
        -------
        pydicom.Dataset
            The loaded DICOM dataset.

        Raises
        ------
        DicomLoadError
            If the file cannot be read or is not a valid DICOM file.
        DicomModalityError
            If require_ct is True and the file is not a CT image.
        MissingRescaleTagsError
            If mandatory rescale tags are absent.
        """
        path = Path(path).resolve()

        if not path.is_file():
            raise DicomLoadError("File does not exist: %s" % path)

        try:
            ds = pydicom.dcmread(str(path), force=False, stop_before_pixels=False)
        except pydicom.errors.InvalidDicomError as exc:
            raise DicomLoadError(
                "Invalid DICOM file '%s': %s" % (path, exc)
            ) from exc
        except Exception as exc:
            raise DicomLoadError(
                "Failed to read DICOM file '%s': %s" % (path, exc)
            ) from exc

        if require_ct and not self.is_ct_image(ds):
            raise DicomModalityError(
                "File '%s' is not a CT image. Modality='%s', SOPClassUID='%s'"
                % (path, getattr(ds, "Modality", "N/A"),
                   getattr(ds, "SOPClassUID", "N/A"))
            )

        # GE Discovery RT compatibility: do NOT reject files missing standard
        # rescale tags — the GE fallback chain in _resolve_ge_rescale handles it.
        # Reference: GE DICOM Conformance Statement Discovery RT Rev 3
        if not hasattr(ds, "RescaleSlope") or not hasattr(ds, "RescaleIntercept"):
            logger.debug(
                "Standard RescaleSlope/RescaleIntercept absent on '%s' — "
                "GE fallback chain will be used during HU conversion.", path.name
            )

        logger.debug("Loaded DICOM file: %s", path.name)
        return ds

    def load_directory(
        self,
        dir_path: Path,
        glob: str = "*.dcm",
        sort_by: str = "InstanceNumber",
    ) -> list[Dataset]:
        """Load all DICOM CT files from a directory.

        Parameters
        ----------
        dir_path : Path
            Directory containing DICOM files.
        glob : str
            Glob pattern for file matching (default: ``*.dcm``).
        sort_by : str
            DICOM tag name used for sorting (default: ``InstanceNumber``).

        Returns
        -------
        list[Dataset]
            Datasets sorted by the specified tag in acquisition order.
        """
        dir_path = Path(dir_path).resolve()
        if not dir_path.is_dir():
            raise DicomLoadError("Directory does not exist: %s" % dir_path)

        files = sorted(dir_path.glob(glob))
        logger.info("Found %d files matching '%s' in %s", len(files), glob, dir_path)

        datasets: list[Dataset] = []
        ct_accepted = 0
        non_ct_skipped = 0
        errors_skipped = 0

        for fpath in files:
            try:
                ds = pydicom.dcmread(str(fpath), force=False, stop_before_pixels=False)
            except Exception:
                logger.debug("Skipping non-DICOM file: %s", fpath.name)
                errors_skipped += 1
                continue

            if not self.is_ct_image(ds):
                logger.debug("Skipping non-CT file: %s (Modality=%s)",
                             fpath.name, getattr(ds, "Modality", "N/A"))
                non_ct_skipped += 1
                continue

            datasets.append(ds)
            ct_accepted += 1

        # GE compatibility: resolve sort key before sorting
        sort_by = self._resolve_sort_key(sort_by, datasets)

        # Sort by the specified DICOM tag
        datasets.sort(key=lambda d: getattr(d, sort_by, 0))

        # ── Spatial sort integration — Phase 5 Step 5.3 ──────────────────────
        # Apply SpatialSortEngine to guarantee anatomically correct slice order.
        # The preferred tag comes from the detected ScannerProfile (if available)
        # or defaults to SliceLocation.
        # Reference: DICOM PS 3.3 C.7.6.2 — Image Plane Module
        if datasets:
            from core.spatial_sort import SpatialSortEngine
            preferred_tag = getattr(self, "_sort_preferred_tag", sort_by)
            sorter = SpatialSortEngine(preferred_tag=preferred_tag)
            sort_result = sorter.sort(datasets)
            datasets = sort_result.datasets
            if sort_result.warnings:
                for w in sort_result.warnings:
                    logger.warning("SpatialSort: %s", w)
            logger.info(
                "Spatial sort complete: %d slices, method='%s', ascending=%s",
                len(datasets), sort_result.sort_method, sort_result.is_ascending
            )
        # ── End spatial sort ──────────────────────────────────────────────────

        logger.info(
            "DICOM load summary — accepted: %d CT, skipped: %d non-CT, "
            "errors: %d, total: %d",
            ct_accepted, non_ct_skipped, errors_skipped, len(files),
        )

        return datasets

    def load_slice_range(
        self,
        dir_path: Path,
        start_slice: int,
        end_slice: int,
        sort_by: str = "InstanceNumber",
        glob: str = "*.dcm",
    ) -> list[Dataset]:
        """Load a specific range of CT slices from a directory.

        Uses 1-based indexing (slice 1 = first acquired slice, matching
        clinical convention).

        Parameters
        ----------
        dir_path : Path
            Directory containing DICOM files.
        start_slice : int
            First slice to include (1-based, inclusive).
        end_slice : int
            Last slice to include (1-based, inclusive).
        sort_by : str
            DICOM tag name for sorting.
        glob : str
            Glob pattern for file matching.

        Returns
        -------
        list[Dataset]
            Selected slice range, sorted by sort_by.

        Raises
        ------
        SliceRangeError
            If range is invalid (start < 1, start > end, end > total).
        """
        # Validate range parameters
        if start_slice < 1:
            raise SliceRangeError(
                "start_slice must be >= 1, got %d" % start_slice
            )
        if start_slice > end_slice:
            raise SliceRangeError(
                "start_slice %d > end_slice %d" % (start_slice, end_slice)
            )

        # Load all CT slices first (sort key fallback handled inside load_directory)
        all_datasets = self.load_directory(dir_path, glob=glob, sort_by=sort_by)
        total = len(all_datasets)

        if end_slice > total:
            raise SliceRangeError(
                "end_slice %d exceeds total slice count %d" % (end_slice, total)
            )

        # Convert 1-based clinical indexing to 0-based Python indexing
        # Slice 1 → index 0, slice N → index N-1
        selected = all_datasets[start_slice - 1 : end_slice]  # 1-based to 0-based

        logger.info(
            "Loaded slice range [%d, %d] — %d slices selected from %d total",
            start_slice, end_slice, len(selected), total,
        )

        return selected

    def get_slice_inventory(
        self,
        dir_path: Path,
        glob: str = "*.dcm",
    ) -> list[dict]:
        """Build a lightweight inventory of all CT slices in a directory.

        Parameters
        ----------
        dir_path : Path
            Directory containing DICOM files.
        glob : str
            Glob pattern for file matching.

        Returns
        -------
        list[dict]
            One dict per slice with keys: index, instance_number,
            slice_location, acquisition_date, series_description, file_path.
        """
        datasets = self.load_directory(dir_path, glob=glob, sort_by="InstanceNumber")

        inventory = []
        for idx, ds in enumerate(datasets, start=1):  # 1-based position
            entry = {
                "index": idx,
                "instance_number": int(getattr(ds, "InstanceNumber", 0)),
                "slice_location": float(getattr(ds, "SliceLocation", 0.0))
                    if hasattr(ds, "SliceLocation") else None,
                "acquisition_date": str(getattr(ds, "AcquisitionDate", "")),
                "series_description": str(getattr(ds, "SeriesDescription", "")),
                "file_path": str(getattr(ds, "filename", "")),
            }
            inventory.append(entry)

        logger.info("Slice inventory: %d CT slices found", len(inventory))
        return inventory

    # ─────────────────────────────────────────────────────────────
    # Pixel Array Conversion
    # ─────────────────────────────────────────────────────────────

    def to_hu_array(self, ds: Dataset) -> np.ndarray:
        """Convert stored pixel values to Hounsfield Units.

        Applies the mandatory DICOM affine transform:
            HU = RescaleSlope × StoredPixelValue + RescaleIntercept

        Parameters
        ----------
        ds : Dataset
            A loaded DICOM dataset with pixel data.

        Returns
        -------
        np.ndarray
            Float32 array of Hounsfield Unit values.

        Raises
        ------
        MissingRescaleTagsError
            If RescaleSlope or RescaleIntercept are missing.
        PixelDataError
            If >1% of pixels fall outside the configured HU bounds.
        """
        # Resolve rescale parameters via GE fallback chain
        # Reference: GE DICOM Conformance Statement Discovery RT Rev 3
        slope, intercept = self._resolve_ge_rescale(ds)

        # Extract pixel array
        pixel_array = ds.pixel_array

        # Handle multi-frame datasets — extract first frame only
        if pixel_array.ndim > 2:
            logger.warning(
                "Multi-frame dataset detected (%d frames). "
                "Extracting first frame only.",
                pixel_array.shape[0],
            )
            pixel_array = pixel_array[0]

        # Apply HU transform: HU = slope * SV + intercept
        # Reference: DICOM PS3.3 C.11.1.1.2
        hu_array = slope * pixel_array.astype(np.float32) + intercept

        # Validate pixel range
        total_pixels = hu_array.size
        out_of_range = np.sum(
            (hu_array < self._config.min_pixel_value)
            | (hu_array > self._config.max_pixel_value)
        )
        out_of_range_pct = (out_of_range / total_pixels) * 100.0

        if out_of_range_pct > 1.0:
            raise PixelDataError(
                "%.2f%% of pixels (%d / %d) fall outside the valid HU range "
                "[%.0f, %.0f]. This may indicate corrupted pixel data or "
                "incorrect rescale parameters."
                % (out_of_range_pct, out_of_range, total_pixels,
                   self._config.min_pixel_value, self._config.max_pixel_value)
            )

        logger.debug(
            "HU conversion complete — shape: %s, range: [%.1f, %.1f] HU",
            hu_array.shape, float(hu_array.min()), float(hu_array.max()),
        )

        return hu_array

    # ─────────────────────────────────────────────────────────────
    # Metadata Extraction
    # ─────────────────────────────────────────────────────────────

    def extract_metadata(self, ds: Dataset) -> DicomMetadata:
        """Extract structured metadata from a DICOM dataset.

        Mandatory tags raise if absent. Optional tags use safe defaults
        and log DEBUG messages — never raise on a missing optional tag.

        Parameters
        ----------
        ds : Dataset
            A loaded DICOM dataset.

        Returns
        -------
        DicomMetadata
            Populated metadata dataclass.
        """
        # ── Mandatory fields ──
        sop_instance_uid = getattr(ds, "SOPInstanceUID", None)
        if sop_instance_uid is None:
            raise DicomLoadError("Missing mandatory tag: SOPInstanceUID")

        # GE Discovery RT compatibility: resolve rescale via fallback chain
        # instead of raising MissingRescaleTagsError
        slope, intercept = self._resolve_ge_rescale(ds)

        # ── Optional fields with safe defaults ──
        series_description = str(getattr(ds, "SeriesDescription", ""))
        acquisition_date = str(getattr(ds, "AcquisitionDate", "19000101"))
        patient_id = str(getattr(ds, "PatientID", "PHANTOM"))
        # GE compatibility: parse numeric tags that may have unit suffixes
        kvp = self._parse_ge_numeric_tag(getattr(ds, "KVP", None), default=0.0)
        slice_thickness_mm = self._parse_ge_numeric_tag(
            getattr(ds, "SliceThickness", None), default=0.0
        )
        instance_number = int(getattr(ds, "InstanceNumber", 0))
        x_ray_tube_current = float(getattr(ds, "XRayTubeCurrent", 0.0))

        # Compute mAs = XRayTubeCurrent × ExposureTime / 1000.0
        raw_exposure_time = getattr(ds, "ExposureTime", None)
        exposure_time = self._parse_ge_numeric_tag(raw_exposure_time, default=0.0)
        if x_ray_tube_current > 0 and exposure_time > 0:
            mas = float(x_ray_tube_current) * exposure_time / 1000.0
        else:
            mas = 0.0

        # Extract pixel spacing as tuple
        raw_spacing = getattr(ds, "PixelSpacing", None)
        if raw_spacing is not None:
            if hasattr(raw_spacing, "__iter__"):
                pixel_spacing_mm = (float(raw_spacing[0]), float(raw_spacing[1]))
            else:
                pixel_spacing_mm = (float(raw_spacing), float(raw_spacing))
        else:
            raw_ips = getattr(ds, "ImagerPixelSpacing", None)
            if raw_ips is not None:
                pixel_spacing_mm = (float(raw_ips[0]), float(raw_ips[1]))
            else:
                pixel_spacing_mm = (1.0, 1.0)
                logger.debug("Neither PixelSpacing nor ImagerPixelSpacing present — using default (1.0, 1.0)")

        # SliceLocation — optional, may be absent
        slice_location = getattr(ds, "SliceLocation", None)
        if slice_location is not None:
            slice_location = float(slice_location)

        # ImagePositionPatient — optional, may be absent
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            image_position_patient = [float(v) for v in ipp]
        else:
            image_position_patient = None

        # GE CTDIvol fallback chain
        # Standard: (0018,9345) CTDIvol
        ctdi_vol = getattr(ds, "CTDIvol", None)
        # GE fallback: ExposureDoseSequence (0018,9328) — present on GE AW reconstructions
        # Reference: GE DICOM Conformance Statement Discovery RT Rev 3
        if ctdi_vol is None:
            try:
                eds = ds[0x0018, 0x9328].value
                if eds:
                    ctdi_vol = float(eds[0][0x0018, 0x9345].value)
                    logger.debug("CTDIvol resolved from GE ExposureDoseSequence: %.2f mGy", ctdi_vol)
            except (KeyError, AttributeError, IndexError, TypeError):
                pass
        if ctdi_vol is not None:
            ctdi_vol = float(ctdi_vol)
        else:
            logger.debug("CTDIvol tag absent on this dataset — SSDE calculation will be skipped")

        # Reconstruction kernel — optional
        reconstruction_kernel = getattr(ds, "ConvolutionKernel", None)
        if reconstruction_kernel is not None:
            reconstruction_kernel = str(reconstruction_kernel)

        return DicomMetadata(
            sop_instance_uid=str(sop_instance_uid),
            series_description=series_description,
            acquisition_date=acquisition_date,
            patient_id=patient_id,
            kvp=kvp,
            mas=mas,
            slice_thickness_mm=slice_thickness_mm,
            pixel_spacing_mm=pixel_spacing_mm,
            rescale_slope=slope,
            rescale_intercept=intercept,
            instance_number=instance_number,
            slice_location=slice_location,
            image_position_patient=image_position_patient,
            ctdi_vol=ctdi_vol,
            reconstruction_kernel=reconstruction_kernel,
            x_ray_tube_current=x_ray_tube_current,
        )

    # ─────────────────────────────────────────────────────────────
    # Classification Helpers
    # ─────────────────────────────────────────────────────────────

    def is_ct_image(self, ds: Dataset) -> bool:
        """Check if a dataset represents a CT image.

        Returns True if Modality == "CT" OR SOPClassUID matches the
        CT Image Storage SOP Class. Both are checked because some
        scanners write non-standard Modality tags.

        Parameters
        ----------
        ds : Dataset
            A loaded DICOM dataset.

        Returns
        -------
        bool
        """
        modality = getattr(ds, "Modality", "")
        sop_uid = getattr(ds, "SOPClassUID", "")

        return (
            str(modality).upper() == self._config.expected_modality
            or str(sop_uid) == self._config.ct_sop_class_uid
        )

    def is_localizer(self, ds: Dataset) -> bool:
        """Check if a dataset is a CT Localizer Radiograph (Scout/Topogram).

        Required by Phase 3 for SSDE calculation.
        GE Discovery RT topograms may not carry the standard localizer SOP
        Class UID but consistently write "LOCALIZER" in ImageType.
        Reference: GE DICOM Conformance Statement Discovery RT, Table A.2

        Parameters
        ----------
        ds : Dataset
            A loaded DICOM dataset.

        Returns
        -------
        bool
        """
        # Condition 1: standard localizer SOP Class UID
        sop_uid = str(getattr(ds, "SOPClassUID", ""))
        if sop_uid == self._config.ct_localizer_sop_class_uid:
            return True

        # Condition 2 & 3: check ImageType tag for "LOCALIZER" string
        # GE-specific detection: ImageType tag contains "LOCALIZER" string
        # Reference: GE DICOM Conformance Statement Discovery RT, Table A.2
        # ImageType on GE localizers: ["ORIGINAL","PRIMARY","LOCALIZER"] or similar
        image_type_values = getattr(ds, "ImageType", [])
        if isinstance(image_type_values, str):
            image_type_values = [image_type_values]
        if isinstance(image_type_values, (list, pydicom.multival.MultiValue)):
            if any("LOCALIZER" in str(v).upper() for v in image_type_values):
                return True

        # Condition 3: SeriesDescription fallback
        series_desc = str(getattr(ds, "SeriesDescription", "")).upper()
        if "LOCALIZER" in series_desc:
            return True

        return False

    def get_pixel_spacing_mm(self, ds: Dataset) -> tuple[float, float]:
        """Extract pixel spacing from a dataset.

        Tries PixelSpacing first, then ImagerPixelSpacing as fallback.

        Parameters
        ----------
        ds : Dataset
            A loaded DICOM dataset.

        Returns
        -------
        tuple[float, float]
            (row_spacing_mm, col_spacing_mm)

        Raises
        ------
        MissingPixelSpacingError
            If neither PixelSpacing nor ImagerPixelSpacing is present.
        """
        # Primary: PixelSpacing (0028,0030)
        ps = getattr(ds, "PixelSpacing", None)
        if ps is not None:
            return (float(ps[0]), float(ps[1]))

        # Fallback: ImagerPixelSpacing (0018,1164)
        ips = getattr(ds, "ImagerPixelSpacing", None)
        if ips is not None:
            logger.debug("Using ImagerPixelSpacing as fallback for PixelSpacing")
            return (float(ips[0]), float(ips[1]))

        raise MissingPixelSpacingError(
            "Neither PixelSpacing nor ImagerPixelSpacing tag is present. "
            "SOPInstanceUID: %s" % getattr(ds, "SOPInstanceUID", "unknown")
        )

    # ─────────────────────────────────────────────────────────────
    # GE Discovery RT Compatibility Helpers
    # ─────────────────────────────────────────────────────────────

    def _resolve_ge_rescale(self, ds: Dataset) -> tuple[float, float]:
        """
        Resolves RescaleSlope and RescaleIntercept for GE scanners that omit
        the standard DICOM tags (0028,1053) and (0028,1052).

        GE-specific fallback chain (in priority order):
        1. Standard tags (0028,1053) / (0028,1052) — used if present
        2. GE private tag (0009,100D) — GE raw calibration slope
           Present on GE Discovery RT images reconstructed in raw mode.
           Reference: GE DICOM Conformance Statement, Discovery RT, Rev 3.
        3. RealWorldValueMappingSequence (0040,9096) — IEC standard alternative
           GE firmware >= 27.x writes calibration here when standard tags absent.
        4. Safe default slope=1.0, intercept=-1024.0 with WARNING log.
           This default is physically reasonable for CT (water=0 HU after -1024).
           A WARNING is logged so the physicist is never silently misled.

        Returns (slope, intercept) as floats.
        """
        # Priority 1: standard DICOM tags — fast path
        slope     = getattr(ds, "RescaleSlope",     None)
        intercept = getattr(ds, "RescaleIntercept", None)
        if slope is not None and intercept is not None:
            return float(slope), float(intercept)

        # Priority 2: GE private tag (0009,100D) — raw calibration slope
        # GE encodes only slope here; intercept defaults to -1024 for CT
        # Reference: GE DICOM Conformance Statement Discovery RT Rev 3, Table C.7-11b
        try:
            ge_slope = ds[0x0009, 0x100D].value
            logger.warning(
                "Standard RescaleSlope absent — using GE private tag (0009,100D): slope=%.4f",
                float(ge_slope)
            )
            return float(ge_slope), -1024.0
        except (KeyError, AttributeError):
            pass

        # Priority 3: RealWorldValueMappingSequence (0040,9096)
        # GE firmware >= 27.x encodes calibration here per IEC 62B/1016/CDV
        try:
            rwvms = ds[0x0040, 0x9096].value
            if rwvms:
                item  = rwvms[0]
                slope = float(item[0x0040, 0x9225].value)  # RealWorldValueSlope
                inter = float(item[0x0040, 0x9224].value)  # RealWorldValueIntercept
                logger.warning(
                    "Standard RescaleSlope absent — using RealWorldValueMappingSequence: "
                    "slope=%.4f intercept=%.4f", slope, inter
                )
                return slope, inter
        except (KeyError, AttributeError, IndexError):
            pass

        # Priority 4: safe default — always log prominently
        logger.warning(
            "No rescale calibration tags found (standard or GE-specific). "
            "Applying default slope=1.0 intercept=-1024.0. "
            "Verify pixel values are in HU before interpreting results. "
            "This is a known issue with some GE Discovery RT firmware versions."
        )
        return 1.0, -1024.0

    def _parse_ge_numeric_tag(self, raw_value, default: float = 0.0) -> float:
        """
        Safely parses a DICOM numeric tag that GE may encode as a string
        with a unit suffix (e.g., "5.0 mm", "120.0 kVp").
        Strips non-numeric suffixes before conversion.
        Returns default if parsing fails entirely.
        """
        if raw_value is None:
            return default
        try:
            return float(raw_value)
        except (ValueError, TypeError):
            # Strip trailing non-numeric characters (units, spaces)
            cleaned = str(raw_value).strip().split()[0]
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                logger.debug(
                    "Could not parse GE numeric tag value '%s', using default %.1f",
                    raw_value, default
                )
                return default

    def set_sort_tag(self, tag: str) -> None:
        """
        Sets the preferred spatial sort tag from a ScannerProfile.
        Called by PipelineOrchestrator after scanner detection.
        """
        self._sort_preferred_tag = tag
        logger.debug("DicomLoader: sort tag set to '%s'", tag)

    def _resolve_sort_key(
        self, sort_by: str, datasets: list[Dataset]
    ) -> str:
        """
        Resolves the sort key for a list of datasets.
        GE Discovery RT images sometimes have InstanceNumber set to 0 for
        all slices in a series (a known GE bug on certain protocol
        reconstructions). When all InstanceNumber values are identical,
        sorting by InstanceNumber produces undefined order.
        Reference: GE DICOM Conformance Statement Discovery RT, known issue Rev 3 §4.2
        """
        if sort_by == "InstanceNumber" and len(datasets) > 1:
            instance_numbers = [
                int(getattr(ds, "InstanceNumber", 0)) for ds in datasets
            ]
            if len(set(instance_numbers)) == 1:
                logger.warning(
                    "All InstanceNumber values are identical (%d) — GE known issue. "
                    "Falling back to SliceLocation sort order.",
                    instance_numbers[0]
                )
                return "SliceLocation"
        return sort_by
