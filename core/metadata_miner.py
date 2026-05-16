"""
core/metadata_miner.py
========================
DICOM Metadata Mining — Automated extraction of CTDIvol, DLP,
scan protocol parameters, and cross-validation from all available sources.

The MetadataMiner consolidates information from:
  1. Axial CT image headers (per-slice metadata)
  2. Siemens RDSR objects (separate DICOM SR files)
  3. GE private tag sequences
  4. Computed/derived values (DLP = SSDE × scan_length)

Reference: DICOM PS 3.3 C.8.15 — CT Image Module;
           DICOM PS 3.3 C.7.6.3 — Image Pixel Module;
           Siemens RDSR: DICOM PS 3.17 Annex GG.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

from core.scanner_profiles import ScannerProfile
from core.dose_metadata_extractor import DoseMetadataExtractor, DoseMetadata

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class ScanProtocolMetadata:
    """CT scan acquisition and reconstruction parameters extracted from DICOM."""
    kvp: float
    mas_mean: float
    slice_thickness_mm: float
    reconstruction_kernel: Optional[str]
    field_of_view_mm: Optional[float]
    n_slices: int
    scan_length_mm: float
    rotation_time_s: Optional[float]

    def to_dict(self) -> dict:
        return {
            "kvp":                   self.kvp,
            "mas_mean":              round(self.mas_mean, 1),
            "slice_thickness_mm":    self.slice_thickness_mm,
            "reconstruction_kernel": self.reconstruction_kernel,
            "field_of_view_mm":      self.field_of_view_mm,
            "n_slices":              self.n_slices,
            "scan_length_mm":        round(self.scan_length_mm, 1),
            "rotation_time_s":       self.rotation_time_s,
        }


@dataclass
class MinedMetadata:
    """Complete metadata mining result for one CT series."""
    scanner_id: str
    acquisition_date: str
    series_description: str
    scanner_profile_id: str

    dose: DoseMetadata
    protocol: ScanProtocolMetadata

    cross_validation: Optional["DoseCrossValidation"] = None
    ctdivol_consistent_across_slices: bool = True
    ctdivol_slice_range_mgy: Optional[tuple[float, float]] = None

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scanner_id":          self.scanner_id,
            "acquisition_date":    self.acquisition_date,
            "series_description":  self.series_description,
            "scanner_profile_id":  self.scanner_profile_id,
            "dose":                self.dose.to_dict(),
            "protocol":            self.protocol.to_dict(),
            "cross_validation":    self.cross_validation.to_dict() if self.cross_validation else None,
            "ctdivol_consistent":  self.ctdivol_consistent_across_slices,
            "ctdivol_slice_range": list(self.ctdivol_slice_range_mgy) if self.ctdivol_slice_range_mgy else None,
            "warnings":            self.warnings,
        }


# ── Miner ──────────────────────────────────────────────────────────────────

class MetadataMiner:
    """
    Extracts and consolidates all clinically relevant metadata from a
    list of CT DICOM datasets representing one series.
    """

    CTDIVOL_CONSISTENCY_THRESHOLD_PCT = 5.0

    def __init__(self, scanner_profile: ScannerProfile) -> None:
        self._profile   = scanner_profile
        self._extractor = DoseMetadataExtractor(scanner_profile)

    def mine(
        self, datasets: list, scanner_id: str = "UNKNOWN", rdsr_dataset=None,
    ) -> MinedMetadata:
        """Mines metadata from all datasets in the series."""
        if not datasets:
            raise ValueError("Cannot mine metadata from an empty dataset list.")

        warnings: list[str] = []
        first_ds = datasets[0]

        dose_meta = self._extract_consensus_ctdivol(datasets, warnings)

        cross_val = None
        if rdsr_dataset is not None and self._profile.dlp_source == "rdsr":
            try:
                rdsr_meta = self._extractor.extract_from_rdsr(rdsr_dataset)
                cross_val = self._extractor.cross_validate(dose_meta, rdsr_meta)
                if not cross_val.consistent:
                    warnings.append(
                        f"CTDIvol cross-validation: DISCREPANCY "
                        f"{cross_val.discrepancy_pct:.1f}%"
                    )
            except Exception as exc:
                warnings.append(f"RDSR cross-validation failed: {exc}")

        protocol = self._extract_protocol(datasets, warnings)
        ctdivol_consistent, ctdivol_range = self._check_ctdivol_consistency(datasets, warnings)

        return MinedMetadata(
            scanner_id=scanner_id,
            acquisition_date=str(getattr(first_ds, "AcquisitionDate", "19000101")),
            series_description=str(getattr(first_ds, "SeriesDescription", "")),
            scanner_profile_id=self._profile.profile_id,
            dose=dose_meta,
            protocol=protocol,
            cross_validation=cross_val,
            ctdivol_consistent_across_slices=ctdivol_consistent,
            ctdivol_slice_range_mgy=ctdivol_range,
            warnings=warnings,
        )

    def _extract_consensus_ctdivol(
        self, datasets: list, warnings: list[str],
    ) -> DoseMetadata:
        """Extracts CTDIvol from all slices and returns the consensus (median) value."""
        ctdivol_values = []
        sources        = []

        for ds in datasets:
            meta = self._extractor.extract(ds)
            if meta.has_ctdi_vol:
                ctdivol_values.append(meta.ctdi_vol_mgy)
                sources.append(meta.source)

        if not ctdivol_values:
            return DoseMetadata(
                ctdi_vol_mgy=None, dlp_mgy_cm=None,
                source="none", confidence="unavailable",
                extraction_notes="CTDIvol absent from all slices.",
            )

        consensus_ctdivol = median(ctdivol_values)
        consensus_source  = max(set(sources), key=sources.count)

        first_meta = self._extractor.extract(datasets[0])

        return DoseMetadata(
            ctdi_vol_mgy=consensus_ctdivol,
            dlp_mgy_cm=first_meta.dlp_mgy_cm,
            source=consensus_source,
            confidence="high" if consensus_source == "standard_tag" else "medium",
            extraction_notes=(
                f"Consensus from {len(ctdivol_values)}/{len(datasets)} slices. "
                f"Range: [{min(ctdivol_values):.3f}, {max(ctdivol_values):.3f}] mGy."
            ),
        )

    def _extract_protocol(
        self, datasets: list, warnings: list[str],
    ) -> ScanProtocolMetadata:
        """Extracts acquisition and reconstruction parameters from DICOM headers."""
        first = datasets[0]
        last  = datasets[-1]

        kvp = float(getattr(first, "KVP", 0.0) or 0.0)
        if kvp == 0.0:
            warnings.append("KVP tag absent or zero.")

        mas_values = []
        for ds in datasets:
            tube_current  = float(getattr(ds, "XRayTubeCurrent", 0.0) or 0.0)
            exposure_time = float(getattr(ds, "ExposureTime",     0.0) or 0.0)
            if tube_current > 0 and exposure_time > 0:
                mas_values.append(tube_current * exposure_time / 1000.0)
        mas_mean = float(sum(mas_values) / len(mas_values)) if mas_values else 0.0

        slice_thickness = float(getattr(first, "SliceThickness", 0.0) or 0.0)

        kernel = getattr(first, "ConvolutionKernel", None)
        if kernel is not None:
            kernel = str(kernel).strip()

        fov = None
        try:
            fov_raw = getattr(first, "ReconstructionDiameter", None)
            if fov_raw is not None:
                fov = float(fov_raw)
        except (TypeError, ValueError):
            pass

        scan_length = 0.0
        try:
            z_first = float(first.ImagePositionPatient[2])
            z_last  = float(last.ImagePositionPatient[2])
            scan_length = abs(z_last - z_first) + slice_thickness
        except (AttributeError, IndexError, TypeError, ValueError):
            scan_length = len(datasets) * slice_thickness

        rotation_time = None
        try:
            for tag_name in ["RevolutionTime", "RotationTime"]:
                val = getattr(first, tag_name, None)
                if val is not None:
                    rotation_time = float(val)
                    break
        except (TypeError, ValueError):
            pass

        return ScanProtocolMetadata(
            kvp=kvp, mas_mean=mas_mean, slice_thickness_mm=slice_thickness,
            reconstruction_kernel=kernel, field_of_view_mm=fov,
            n_slices=len(datasets), scan_length_mm=scan_length,
            rotation_time_s=rotation_time,
        )

    def _check_ctdivol_consistency(
        self, datasets: list, warnings: list[str],
    ) -> tuple[bool, Optional[tuple[float, float]]]:
        """Checks whether CTDIvol varies significantly across slices."""
        values = []
        for ds in datasets:
            meta = self._extractor.extract(ds)
            if meta.has_ctdi_vol:
                values.append(meta.ctdi_vol_mgy)

        if len(values) < 2:
            return True, None

        v_min  = min(values)
        v_max  = max(values)
        v_mean = sum(values) / len(values)

        if v_mean == 0:
            return True, (v_min, v_max)

        range_pct = (v_max - v_min) / v_mean * 100.0

        if range_pct > self.CTDIVOL_CONSISTENCY_THRESHOLD_PCT:
            warnings.append(
                f"CTDIvol varies {range_pct:.1f}% across {len(values)} slices "
                f"(range: {v_min:.3f}–{v_max:.3f} mGy)."
            )
            return False, (v_min, v_max)

        return True, (v_min, v_max)
