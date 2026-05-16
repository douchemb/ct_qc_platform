"""
core/scanner_profiles.py
=========================
Scanner Hardware Profile System.

Provides automatic detection of CT scanner type from DICOM metadata
and returns a ScannerProfile object that configures all downstream
parsing behaviour. No if/else chains in clinical code — routing is
declarative via scanner_profiles.yaml.

The profile system is the single point of truth for:
  - How to extract RescaleSlope/Intercept (standard vs GE private chain)
  - Where to find CTDIvol and DLP (standard tag vs RDSR vs sequence)
  - How to detect localizer images (SOP UID vs ImageType string)
  - Which spatial sort tag to prefer (SliceLocation vs ImagePositionPatient)
  - Which phantom adapter to use by default

Usage:
    registry = ScannerProfileRegistry()
    profile  = registry.detect(ds)
    # profile.rescale_method, profile.sort_preferred_tag, etc.

Reference: DICOM PS 3.3 — Information Object Definitions;
           Siemens DICOM Conformance Statement SOMATOM go.Sim Rev 2;
           GE DICOM Conformance Statement Discovery RT Rev 3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Path to the YAML configuration file at project root
_PROFILES_YAML = Path(__file__).parent.parent / "scanner_profiles.yaml"


# ── Custom exceptions ──────────────────────────────────────────────────────

class ScannerProfileError(RuntimeError):
    """Raised when the profile registry cannot be loaded or parsed."""


class UnknownScannerError(RuntimeError):
    """
    Raised when no profile matches the scanner and strict mode is enabled.
    In non-strict mode, the generic profile is returned instead.
    """


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CTDIvolTagConfig:
    """DICOM tag addresses for CTDIvol extraction, per scanner profile."""
    primary: tuple[int, int]              # e.g., (0x0018, 0x9345)
    secondary: Optional[tuple[int, int]]  # fallback tag, or None


@dataclass(frozen=True)
class ScannerProfile:
    """
    Immutable hardware profile for one CT scanner family.
    Consumed by DicomLoader, DoseMetadataExtractor, and PhantomAdapterFactory.

    All tag addresses are stored as (group, element) integer tuples,
    matching pydicom's tag access convention: ds[group, element].
    """
    profile_id: str                     # e.g., "siemens_somatom_gosim"
    display_name: str                   # human-readable name for UI/logs
    manufacturer_match: list[str]       # Manufacturer tag values to match
    model_match: list[str]              # ManufacturerModelName values to match

    # DICOM parsing behaviour
    rescale_method: str                 # "standard" | "ge_private_chain"
    ctdivol_tags: CTDIvolTagConfig
    dlp_source: str                     # "rdsr" | "calculated" | "header"
    localizer_detection: str            # "sop_uid" | "image_type_string"
    sort_preferred_tag: str             # "SliceLocation" | "ImagePositionPatient"
    private_tags_enabled: bool
    instance_number_bug: bool           # True = GE bug, fallback sort needed

    # Clinical defaults
    default_phantom: str                # phantom profile ID
    effective_energy_kev: float         # μ_water reference energy
    notes: str = ""

    def is_siemens(self) -> bool:
        return "SIEMENS" in (m.upper() for m in self.manufacturer_match)

    def is_ge(self) -> bool:
        return any("GE" in m.upper() for m in self.manufacturer_match)

    def is_generic(self) -> bool:
        return self.profile_id == "generic"

    def to_dict(self) -> dict:
        return {
            "profile_id":           self.profile_id,
            "display_name":         self.display_name,
            "rescale_method":       self.rescale_method,
            "dlp_source":           self.dlp_source,
            "localizer_detection":  self.localizer_detection,
            "sort_preferred_tag":   self.sort_preferred_tag,
            "default_phantom":      self.default_phantom,
            "is_siemens":           self.is_siemens(),
            "is_ge":                self.is_ge(),
        }


# ── Registry ───────────────────────────────────────────────────────────────

class ScannerProfileRegistry:
    """
    Loads scanner profiles from scanner_profiles.yaml and detects
    the correct profile for a given pydicom Dataset.

    Detection algorithm (priority order):
      1. Match ManufacturerModelName against profile.model_match (case-insensitive)
      2. Match Manufacturer against profile.manufacturer_match (case-insensitive)
      3. Return generic profile if no match (never raises in non-strict mode)

    The registry is instantiated once and reused — yaml parsing is cached.
    """

    def __init__(
        self,
        profiles_yaml: Path = _PROFILES_YAML,
        strict: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        profiles_yaml : Path
            Path to the scanner_profiles.yaml file.
        strict : bool
            If True, raises UnknownScannerError when no profile matches.
            If False (default), returns the generic profile as fallback.
        """
        self._strict = strict
        self._profiles: list[ScannerProfile] = []
        self._generic: Optional[ScannerProfile] = None
        self._load_profiles(profiles_yaml)
        logger.info(
            "ScannerProfileRegistry loaded %d profiles from %s",
            len(self._profiles), profiles_yaml.name
        )

    def _load_profiles(self, yaml_path: Path) -> None:
        """Parses scanner_profiles.yaml and builds ScannerProfile objects."""
        if not yaml_path.exists():
            raise ScannerProfileError(
                "Scanner profiles file not found: %s. "
                "Ensure scanner_profiles.yaml is present at the project root."
                % yaml_path
            )
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ScannerProfileError(
                "Failed to parse %s: %s" % (yaml_path, exc)
            ) from exc

        profiles_raw = raw.get("profiles", {})
        for profile_id, cfg in profiles_raw.items():
            # Parse CTDIvol tag config
            primary_raw   = cfg["ctdivol_tags"]["primary"]
            secondary_raw = cfg["ctdivol_tags"].get("secondary")

            ctdivol_cfg = CTDIvolTagConfig(
                primary=tuple(primary_raw) if primary_raw else (0x0018, 0x9345),
                secondary=tuple(secondary_raw) if secondary_raw else None,
            )

            profile = ScannerProfile(
                profile_id=profile_id,
                display_name=cfg.get("display_name", profile_id),
                manufacturer_match=cfg.get("manufacturer_match", []),
                model_match=cfg.get("model_match", []),
                rescale_method=cfg.get("rescale_method", "standard"),
                ctdivol_tags=ctdivol_cfg,
                dlp_source=cfg.get("dlp_source", "header"),
                localizer_detection=cfg.get("localizer_detection", "sop_uid"),
                sort_preferred_tag=cfg.get("sort_preferred_tag", "SliceLocation"),
                private_tags_enabled=cfg.get("private_tags_enabled", False),
                instance_number_bug=cfg.get("instance_number_bug", False),
                default_phantom=cfg.get("default_phantom", "generic"),
                effective_energy_kev=float(cfg.get("effective_energy_kev", 70.0)),
                notes=cfg.get("notes", ""),
            )

            self._profiles.append(profile)
            if profile_id == "generic":
                self._generic = profile

        if self._generic is None:
            raise ScannerProfileError(
                "scanner_profiles.yaml must contain a 'generic' profile "
                "as the universal fallback."
            )

    def detect(self, ds: "pydicom.Dataset") -> ScannerProfile:
        """
        Detects the scanner profile for a given DICOM dataset.

        Reads Manufacturer (0008,0070) and ManufacturerModelName (0008,1090).
        Matching is case-insensitive and uses substring matching — a profile
        with model_match = ["SOMATOM go.Sim"] will match
        ManufacturerModelName = "SOMATOM go.Sim RT" as well.

        Returns the most specific match (model > manufacturer > generic).
        """
        manufacturer = str(getattr(ds, "Manufacturer", "")).upper().strip()
        model        = str(getattr(ds, "ManufacturerModelName", "")).upper().strip()

        logger.debug(
            "Detecting scanner profile for Manufacturer='%s' Model='%s'",
            manufacturer, model
        )

        # Priority 1: model name match (most specific)
        for profile in self._profiles:
            if profile.is_generic():
                continue
            for m in profile.model_match:
                if m.upper() in model or model in m.upper():
                    logger.info(
                        "Scanner profile matched by model: '%s' → %s",
                        model, profile.profile_id
                    )
                    return profile

        # Priority 2: manufacturer match
        for profile in self._profiles:
            if profile.is_generic():
                continue
            for m in profile.manufacturer_match:
                if m.upper() in manufacturer:
                    logger.info(
                        "Scanner profile matched by manufacturer: '%s' → %s",
                        manufacturer, profile.profile_id
                    )
                    return profile

        # Priority 3: generic fallback
        if self._strict:
            raise UnknownScannerError(
                "No scanner profile matched Manufacturer='%s' "
                "Model='%s'. Add a profile to scanner_profiles.yaml."
                % (manufacturer, model)
            )

        logger.warning(
            "No scanner profile matched Manufacturer='%s' Model='%s'. "
            "Using generic fallback profile. "
            "Add a specific profile to scanner_profiles.yaml for full compatibility.",
            manufacturer, model
        )
        return self._generic

    def get_profile(self, profile_id: str) -> ScannerProfile:
        """Returns a profile by its ID string. Raises KeyError if not found."""
        for p in self._profiles:
            if p.profile_id == profile_id:
                return p
        raise KeyError(
            "Scanner profile '%s' not found. "
            "Available: %s" % (profile_id, [p.profile_id for p in self._profiles])
        )

    @property
    def available_profiles(self) -> list[str]:
        """Returns list of all loaded profile IDs."""
        return [p.profile_id for p in self._profiles]
