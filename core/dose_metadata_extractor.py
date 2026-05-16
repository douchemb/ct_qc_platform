"""
core/dose_metadata_extractor.py
=================================
DICOM Dose Metadata Extraction — CTDIvol and DLP.

Extracts CTDIvol and DLP from scanner-specific DICOM locations.
Implements cross-validation between header tags and RDSR data.
Provides a unified DoseMetadata object regardless of scanner brand.

Siemens SOMATOM:
  CTDIvol: (0018,9345) standard tag, also in RDSR SR document
  DLP:     Radiation Dose SR (RDSR) — separate DICOM object with SOP Class
           1.2.840.10008.5.1.4.1.1.88.67 (Enhanced SR)

GE Discovery RT:
  CTDIvol: (0018,9345) → fallback (0018,9328) ExposureDoseSequence
  DLP:     Not reliably stored — calculated from SSDE_mean × scan_length

Reference: DICOM PS 3.3 C.8.15.3 — CT Acquisition;
           IEC 60601-2-44:2009 — CTDIvol definition;
           AAPM Report 96 — DLP definition;
           Siemens RDSR format: DICOM PS 3.17 Annex GG.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.scanner_profiles import ScannerProfile

logger = logging.getLogger(__name__)


# ── Custom exceptions ──────────────────────────────────────────────────────

class MissingCTDIvolError(ValueError):
    """
    Raised when CTDIvol cannot be extracted from any available source.
    SSDE calculation is impossible without CTDIvol.
    """


class DoseMetadataExtractionError(RuntimeError):
    """Raised when dose metadata extraction fails unexpectedly."""


# ── Result dataclasses ─────────────────────────────────────────────────────

@dataclass
class DoseMetadata:
    """
    Unified dose metadata extracted from DICOM, regardless of scanner brand.

    source indicates where CTDIvol was found:
      "standard_tag"      — (0018,9345) directly
      "ge_sequence"       — GE ExposureDoseSequence fallback
      "rdsr"              — Siemens RDSR object
      "none"              — not found anywhere

    confidence:
      "high"              — from standard tag or verified RDSR
      "medium"            — from secondary/fallback tag
      "estimated"         — calculated from other values
      "unavailable"       — not found
    """
    ctdi_vol_mgy: Optional[float]
    dlp_mgy_cm: Optional[float]
    source: str
    confidence: str
    extraction_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "ctdi_vol_mgy":    self.ctdi_vol_mgy,
            "dlp_mgy_cm":      self.dlp_mgy_cm,
            "source":          self.source,
            "confidence":      self.confidence,
            "extraction_notes": self.extraction_notes,
        }

    @property
    def has_ctdi_vol(self) -> bool:
        return self.ctdi_vol_mgy is not None

    @property
    def has_dlp(self) -> bool:
        return self.dlp_mgy_cm is not None


@dataclass
class DoseCrossValidation:
    """
    Cross-validation result comparing CTDIvol from two independent sources.
    Used to detect firmware bugs and reconstruction parameter inconsistencies.
    """
    ctdivol_source1_mgy: float
    ctdivol_source2_mgy: Optional[float]
    source1_name: str
    source2_name: str
    discrepancy_pct: Optional[float]
    consistent: bool        # True if discrepancy < 5%
    warning_message: Optional[str]

    def to_dict(self) -> dict:
        return {
            "ctdivol_source1_mgy":  self.ctdivol_source1_mgy,
            "ctdivol_source2_mgy":  self.ctdivol_source2_mgy,
            "source1":              self.source1_name,
            "source2":              self.source2_name,
            "discrepancy_pct":      self.discrepancy_pct,
            "consistent":           self.consistent,
            "warning":              self.warning_message,
        }


# ── Extractor ──────────────────────────────────────────────────────────────

class DoseMetadataExtractor:
    """
    Extracts CTDIvol and DLP from DICOM datasets using scanner-specific
    fallback chains defined in the ScannerProfile.

    Usage:
        extractor = DoseMetadataExtractor(scanner_profile)
        dose_meta = extractor.extract(ds)
        if dose_meta.has_ctdi_vol:
            ssde = f_factor * dose_meta.ctdi_vol_mgy
    """

    # Standard CTDIvol tag — IEC 60601-2-44:2009 / DICOM (0018,9345)
    _CTDIVOL_STANDARD_TAG = (0x0018, 0x9345)

    # GE ExposureDoseSequence tag — (0018,9328)
    _GE_EXPOSURE_DOSE_SEQ_TAG = (0x0018, 0x9328)

    # DLP standard tag — DICOM (0018,9094)
    _DLP_STANDARD_TAG = (0x0018, 0x9094)

    def __init__(self, scanner_profile: ScannerProfile) -> None:
        self._profile = scanner_profile

    def extract(self, ds: "pydicom.Dataset") -> DoseMetadata:
        """
        Extracts CTDIvol and DLP using the scanner-profile-specific chain.
        Never raises — returns DoseMetadata with source="none" if all fail.
        """
        ctdi_vol     = None
        dlp          = None
        source       = "none"
        confidence   = "unavailable"
        notes        = []

        # ── CTDIvol extraction ─────────────────────────────────────────
        # Step 1: Standard tag (0018,9345)
        ctdi_val = self._try_standard_ctdivol(ds)
        if ctdi_val is not None:
            ctdi_vol   = ctdi_val
            source     = "standard_tag"
            confidence = "high"
            logger.debug(
                "CTDIvol extracted from standard tag (0018,9345): %.3f mGy", ctdi_vol
            )

        # Step 2: GE-specific fallback (ExposureDoseSequence)
        if ctdi_vol is None and self._profile.private_tags_enabled:
            ctdi_val = self._try_ge_exposure_dose_sequence(ds)
            if ctdi_val is not None:
                ctdi_vol   = ctdi_val
                source     = "ge_sequence"
                confidence = "medium"
                notes.append(
                    "CTDIvol from GE ExposureDoseSequence (0018,9328) — "
                    "verify against scanner console."
                )
                logger.debug(
                    "CTDIvol extracted from GE ExposureDoseSequence: %.3f mGy", ctdi_vol
                )

        if ctdi_vol is None:
            notes.append(
                "CTDIvol not found in any standard or scanner-specific tag. "
                "SSDE calculation will not be possible."
            )
            logger.warning(
                "CTDIvol extraction failed for scanner profile '%s'. "
                "Neither standard tag nor GE fallback yielded a value.",
                self._profile.profile_id
            )

        # ── DLP extraction ─────────────────────────────────────────────
        if self._profile.dlp_source == "header":
            dlp = self._try_standard_dlp(ds)
            if dlp is not None:
                logger.debug(
                    "DLP extracted from standard header tag: %.2f mGy·cm", dlp
                )

        elif self._profile.dlp_source == "rdsr":
            # Siemens RDSR — cannot be parsed from the axial image dataset alone.
            # RDSR is a separate DICOM object. Log that it requires external lookup.
            notes.append(
                "DLP source is RDSR (Siemens). RDSR parsing requires the "
                "separate SR DICOM object. DLP set to None — "
                "use DoseMetadataExtractor.extract_from_rdsr() if RDSR is available."
            )
            logger.info(
                "DLP source is RDSR for profile '%s'. "
                "Separate RDSR object required for DLP extraction.",
                self._profile.profile_id
            )

        elif self._profile.dlp_source == "calculated":
            # GE: DLP calculated after SSDE is known — placeholder here
            notes.append(
                "DLP will be calculated as SSDE_mean × scan_length_cm "
                "after dosimetry computation."
            )

        return DoseMetadata(
            ctdi_vol_mgy=ctdi_vol,
            dlp_mgy_cm=dlp,
            source=source,
            confidence=confidence,
            extraction_notes=" | ".join(notes) if notes else "",
        )

    def extract_from_rdsr(self, rdsr_ds: "pydicom.Dataset") -> DoseMetadata:
        """
        Extracts CTDIvol and DLP from a Siemens RDSR DICOM object.
        The RDSR is a separate dataset (SOP Class 1.2.840.10008.5.1.4.1.1.88.67).
        This method is called when the RDSR file is explicitly provided.

        Reference: DICOM PS 3.17 Annex GG — Radiation Dose SR.
        """
        ctdi_vol = None
        dlp      = None
        notes    = []

        # RDSR content is in the ContentSequence
        try:
            content_seq = getattr(rdsr_ds, "ContentSequence", None)
            if content_seq is None:
                notes.append("RDSR ContentSequence absent.")
                return DoseMetadata(
                    ctdi_vol_mgy=None, dlp_mgy_cm=None,
                    source="rdsr", confidence="unavailable",
                    extraction_notes=" | ".join(notes)
                )

            # Walk the SR tree looking for CTDIvol and DLP concept names
            # Concept Name Code: (113830, DCM, "Mean CTDIvol")
            # Concept Name Code: (113838, DCM, "DLP")
            ctdi_vol, dlp = self._walk_rdsr_tree(content_seq)

            if ctdi_vol is not None:
                logger.info(
                    "CTDIvol extracted from Siemens RDSR: %.3f mGy", ctdi_vol
                )
            if dlp is not None:
                logger.info(
                    "DLP extracted from Siemens RDSR: %.2f mGy·cm", dlp
                )

        except Exception as exc:
            notes.append("RDSR parsing error: %s" % exc)
            logger.error("RDSR extraction failed: %s", exc, exc_info=True)

        confidence = "high" if ctdi_vol is not None else "unavailable"
        return DoseMetadata(
            ctdi_vol_mgy=ctdi_vol,
            dlp_mgy_cm=dlp,
            source="rdsr",
            confidence=confidence,
            extraction_notes=" | ".join(notes),
        )

    def cross_validate(
        self,
        meta_from_header: DoseMetadata,
        meta_from_rdsr: DoseMetadata,
        tolerance_pct: float = 5.0,
    ) -> DoseCrossValidation:
        """
        Compares CTDIvol from two independent sources (header vs RDSR).
        A discrepancy > 5% indicates a firmware bug or reconstruction mismatch.

        Reference: Clinical standard for dose consistency verification —
                   IPEM Report 91 Section 4.3.
        """
        if not meta_from_header.has_ctdi_vol or not meta_from_rdsr.has_ctdi_vol:
            return DoseCrossValidation(
                ctdivol_source1_mgy=meta_from_header.ctdi_vol_mgy or 0.0,
                ctdivol_source2_mgy=meta_from_rdsr.ctdi_vol_mgy,
                source1_name="header",
                source2_name="rdsr",
                discrepancy_pct=None,
                consistent=False,
                warning_message="Cannot cross-validate — one or both sources unavailable.",
            )

        v1 = meta_from_header.ctdi_vol_mgy
        v2 = meta_from_rdsr.ctdi_vol_mgy
        discrepancy = abs(v1 - v2) / max(abs(v1), 1e-6) * 100.0
        consistent  = discrepancy < tolerance_pct

        warning = None
        if not consistent:
            warning = (
                "CTDIvol discrepancy %.1f%% > %.1f%% tolerance. "
                "Header: %.3f mGy | RDSR: %.3f mGy. "
                "Possible firmware bug or post-hoc reconstruction with different parameters. "
                "Use RDSR value for dosimetric reporting (higher reliability)."
                % (discrepancy, tolerance_pct, v1, v2)
            )
            logger.warning("Dose cross-validation: %s", warning)

        return DoseCrossValidation(
            ctdivol_source1_mgy=v1,
            ctdivol_source2_mgy=v2,
            source1_name=meta_from_header.source,
            source2_name=meta_from_rdsr.source,
            discrepancy_pct=discrepancy,
            consistent=consistent,
            warning_message=warning,
        )

    # ── Private helpers ────────────────────────────────────────────────

    def _try_standard_ctdivol(self, ds) -> Optional[float]:
        """Reads CTDIvol from the standard DICOM tag (0018,9345)."""
        try:
            val = getattr(ds, "CTDIvol", None)
            if val is not None:
                return float(val)
            # Also try direct tag access
            if self._CTDIVOL_STANDARD_TAG in ds:
                return float(ds[self._CTDIVOL_STANDARD_TAG].value)
        except (TypeError, ValueError, AttributeError):
            pass
        return None

    def _try_ge_exposure_dose_sequence(self, ds) -> Optional[float]:
        """
        GE-specific fallback: reads CTDIvol from ExposureDoseSequence (0018,9328).
        Reference: GE DICOM Conformance Statement Discovery RT Rev 3, Table C.8-26.
        """
        try:
            seq = ds[self._GE_EXPOSURE_DOSE_SEQ_TAG].value
            if seq:
                item = seq[0]
                ctdi = getattr(item, "CTDIvol", None)
                if ctdi is not None:
                    return float(ctdi)
                # Try the standard tag within the sequence item
                if self._CTDIVOL_STANDARD_TAG in item:
                    return float(item[self._CTDIVOL_STANDARD_TAG].value)
        except (KeyError, AttributeError, IndexError, TypeError):
            pass
        return None

    def _try_standard_dlp(self, ds) -> Optional[float]:
        """Reads DLP from the standard DICOM tag (0018,9094) if present."""
        try:
            if self._DLP_STANDARD_TAG in ds:
                return float(ds[self._DLP_STANDARD_TAG].value)
        except (KeyError, TypeError, ValueError):
            pass
        return None

    def _walk_rdsr_tree(
        self, content_seq, depth: int = 0
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Recursively walks the RDSR ContentSequence to find CTDIvol and DLP.
        Stops recursing at depth > 10 to prevent infinite loops on malformed RDSR.
        """
        if depth > 10:
            return None, None

        ctdi_vol = None
        dlp      = None

        for item in content_seq:
            try:
                concept = getattr(item, "ConceptNameCodeSequence", None)
                if concept and len(concept) > 0:
                    code_val = concept[0].CodeValue

                    # (113830, DCM, "Mean CTDIvol")
                    if code_val == "113830" and ctdi_vol is None:
                        measured = getattr(item, "MeasuredValueSequence", None)
                        if measured and len(measured) > 0:
                            ctdi_vol = float(measured[0].NumericValue)

                    # (113838, DCM, "DLP")
                    elif code_val == "113838" and dlp is None:
                        measured = getattr(item, "MeasuredValueSequence", None)
                        if measured and len(measured) > 0:
                            dlp = float(measured[0].NumericValue)

                # Recurse into nested content
                nested = getattr(item, "ContentSequence", None)
                if nested:
                    sub_ctdi, sub_dlp = self._walk_rdsr_tree(nested, depth + 1)
                    if ctdi_vol is None:
                        ctdi_vol = sub_ctdi
                    if dlp is None:
                        dlp = sub_dlp

            except (AttributeError, TypeError, ValueError, IndexError):
                continue

        return ctdi_vol, dlp
